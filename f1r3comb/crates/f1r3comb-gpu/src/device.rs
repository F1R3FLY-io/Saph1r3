//! The wgpu backend: a [`StepEngine`] whose find-and-select runs on a GPU.

use crate::wgsl::{self, meta, slot, REDEX_WORDS, WORKGROUP};
use f1r3comb_mat::State;
use f1r3comb_par::{choose, Config, Redex, Resolver, StepEngine, PAD};
use f1r3comb_term::{Shape, SHAPES};
use f1r3comb_term::rules::{rules, RULE_COUNT};
use std::collections::HashMap;

const SLOTS: u32 = 8;
const SLOT_STRIDE: u64 = 256;

/// Statistics for a device run.
#[derive(Clone, Debug, Default)]
pub struct DeviceStats {
    pub steps: u64,
    pub rounds: u64,
    pub dispatches: u64,
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipes: HashMap<&'static str, wgpu::ComputePipeline>,
    params: wgpu::Buffer,
    meta: wgpu::Buffer,
    cols: wgpu::Buffer,
    work: wgpu::Buffer,
    rdx: wgpu::Buffer,
    bind: Option<wgpu::BindGroup>,
    pub adapter_name: String,
    pub backend: String,
    pub stats: DeviceStats,
}

#[derive(Debug)]
pub struct GpuError(pub String);

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

fn bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn groups(n: u32) -> (u32, u32) {
    let g = n.div_ceil(WORKGROUP).max(1);
    if g <= 65535 {
        (g, 1)
    } else {
        (65535, g.div_ceil(65535))
    }
}

impl Gpu {
    /// Open the first adapter wgpu offers (a hardware GPU if present, else a
    /// software one such as lavapipe).
    pub fn new() -> Result<Gpu, GpuError> {
        pollster::block_on(Self::open())
    }

    async fn open() -> Result<Gpu, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| GpuError(format!("no GPU adapter: {e}")))?;
        let info = adapter.get_info();
        let al = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
            max_buffer_size: al.max_buffer_size,
            ..wgpu::Limits::downlevel_defaults()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("f1r3comb"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuError(format!("request_device: {e}")))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("f1r3comb kernels"),
            source: wgpu::ShaderSource::Wgsl(wgsl::kernels().into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("f1r3comb"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                storage(3, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("f1r3comb"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let mut pipes = HashMap::new();
        for e in wgsl::ENTRY_POINTS {
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(e),
                compilation_options: Default::default(),
                cache: None,
            });
            pipes.insert(*e, p);
        }
        let mk = |label, size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let params = mk("params", SLOT_STRIDE * SLOTS as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let meta_b = mk("meta", (meta::WORDS * 4) as u64, st);
        let cols = mk("cols", 1024, st);
        let work = mk("work", 1024, st);
        let rdx = mk("rdx", 1024, st);
        Ok(Gpu {
            device,
            queue,
            layout,
            pipes,
            params,
            meta: meta_b,
            cols,
            work,
            rdx,
            bind: None,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            stats: DeviceStats::default(),
        })
    }

    fn ensure(&mut self, which: u8, words: usize) {
        let need = ((words.max(4) * 4) as u64).next_power_of_two();
        let buf = match which {
            0 => &mut self.cols,
            1 => &mut self.work,
            _ => &mut self.rdx,
        };
        if buf.size() >= need {
            return;
        }
        *buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grown"),
            size: need,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.bind = None;
    }

    fn bind_group(&mut self) -> wgpu::BindGroup {
        if self.bind.is_none() {
            self.bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("f1r3comb"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.meta.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.cols.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.work.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.rdx.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.params,
                            offset: 0,
                            size: wgpu::BufferSize::new(16),
                        }),
                    },
                ],
            }));
        }
        self.bind.clone().unwrap()
    }

    fn set_params(&self, s: u32, p: [u32; 4]) {
        self.queue.write_buffer(&self.params, s as u64 * SLOT_STRIDE, &bytes(&p));
    }

    fn read(&self, buf: &wgpu::Buffer, word_off: usize, words: usize) -> Vec<u32> {
        if words == 0 {
            return Vec::new();
        }
        let size = (words * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, (word_off * 4) as u64, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait).expect("poll");
        let data = slice.get_mapped_range();
        let v = data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data);
        staging.unmap();
        v
    }

    /// Record dispatches: (entry, parameter slot, thread count).
    fn run(&mut self, seq: &[(&'static str, u32, u32)]) {
        let bg = self.bind_group();
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_bind_group(0, &bg, &[0]);
            for (e, s, n) in seq {
                pass.set_pipeline(&self.pipes[e]);
                pass.set_bind_group(0, &bg, &[s * SLOT_STRIDE as u32]);
                if *e == "k_scan" {
                    pass.dispatch_workgroups(1, 1, 1);
                } else {
                    if *n == 0 {
                        continue;
                    }
                    let (x, y) = groups(*n);
                    pass.dispatch_workgroups(x, y, 1);
                }
                self.stats.dispatches += 1;
            }
        }
        self.queue.submit([enc.finish()]);
    }

    /// Enumerate the redexes of `st` (canonical order) and, if `mis`, run
    /// the maximal-independent-set rounds. Returns redexes and decisions.
    pub fn step(&mut self, st: &State, seed: u64, step: u64, mis: bool) -> (Vec<Redex>, Vec<u32>) {
        // ---- host-side layout
        let mut m = vec![0u32; meta::WORDS];
        let mut cols: Vec<u32> = Vec::new();
        for s in Shape::ALL {
            let t = st.table(s);
            m[meta::LEN + s.ix()] = t.len;
            m[meta::BASE + s.ix()] = cols.len() as u32;
            for j in 0..s.arity() {
                cols.extend(t.cols[j].iter().map(|x| x.0));
            }
        }
        let mut acc = 0u32;
        for (r, spec) in rules().iter().enumerate() {
            m[meta::SEG + r] = acc;
            acc += st.table(spec.consumer).len;
        }
        m[meta::SEG + RULE_COUNT] = acc;
        let threads = acc;
        let mut refs = 0u32;
        for s in 0..SHAPES {
            m[meta::REF + s] = refs;
            refs += m[meta::LEN + s];
        }
        m[meta::REF + SHAPES] = refs;
        let ml = st.table(Shape::M).len;
        let ql = st.table(Shape::Q).len;
        let keys = st.table(Shape::M).cols[0]
            .iter()
            .chain(st.table(Shape::Q).cols[0].iter())
            .map(|x| x.0 + 1)
            .max()
            .unwrap_or(0);
        m[meta::KEYS] = keys;
        m[meta::SEED] = (seed >> 32) as u32;
        m[meta::SEED + 1] = seed as u32;
        m[meta::SEED + 2] = (step >> 32) as u32;
        m[meta::SEED + 3] = step as u32;
        // work sections
        let mut w = 0u32;
        let mut sec = |k: usize, n: u32, m: &mut Vec<u32>| {
            m[k] = w;
            w += n;
        };
        sec(meta::W_COUNTS, 2 * keys, &mut m);
        sec(meta::W_START, 2 * keys + 1, &mut m);
        sec(meta::W_BROWS, ml + ql, &mut m);
        sec(meta::W_CNT, threads, &mut m);
        sec(meta::W_OFF, threads + 1, &mut m);
        sec(meta::W_BEST, 3 * refs, &mut m);
        sec(meta::W_TAKEN, refs, &mut m);
        sec(meta::W_CTRL, 4, &mut m);
        self.ensure(0, cols.len());
        self.ensure(1, w as usize);
        self.queue.write_buffer(&self.meta, 0, &bytes(&m));
        if !cols.is_empty() {
            self.queue.write_buffer(&self.cols, 0, &bytes(&cols));
        }
        // ---- bucketing and counting
        self.set_params(slot::ZERO_COUNTS, [m[meta::W_COUNTS], 2 * keys, 0, 0]);
        self.set_params(slot::SCAN_BUCKETS, [m[meta::W_COUNTS], m[meta::W_START], 2 * keys, 0]);
        self.set_params(slot::SCAN_COUNTS, [m[meta::W_CNT], m[meta::W_OFF], threads, 0]);
        self.run(&[
            ("k_fill", slot::ZERO_COUNTS, 2 * keys),
            ("k_hist", slot::COUNT, ml + ql),
            ("k_scan", slot::SCAN_BUCKETS, 0),
            ("k_scatter", slot::COUNT, ml + ql),
            ("k_bsort", slot::COUNT, 2 * keys),
            ("k_count", slot::COUNT, threads),
            ("k_scan", slot::SCAN_COUNTS, 0),
        ]);
        let total = self.read(&self.work, (m[meta::W_OFF] + threads) as usize, 1)[0];
        if total == 0 {
            return (Vec::new(), Vec::new());
        }
        self.ensure(2, total as usize * REDEX_WORDS);
        self.set_params(slot::REDEXES, [total, 0, 0, 0]);
        self.run(&[("k_emit", slot::COUNT, threads)]);
        if mis {
            self.set_params(slot::FILL_BEST, [m[meta::W_BEST], 3 * refs, u32::MAX, 0]);
            self.set_params(slot::ZERO_TAKEN, [m[meta::W_TAKEN], refs, 0, 0]);
            self.set_params(slot::ZERO_CTRL, [m[meta::W_CTRL], 4, 0, 0]);
            loop {
                self.stats.rounds += 1;
                self.run(&[
                    ("k_fill", slot::FILL_BEST, 3 * refs),
                    ("k_fill", slot::ZERO_TAKEN, refs),
                    ("k_fill", slot::ZERO_CTRL, 4),
                    ("k_min_hi", slot::REDEXES, total),
                    ("k_min_lo", slot::REDEXES, total),
                    ("k_min_idx", slot::REDEXES, total),
                    ("k_take", slot::REDEXES, total),
                    ("k_exclude", slot::REDEXES, total),
                ]);
                if self.read(&self.work, m[meta::W_CTRL] as usize, 1)[0] == 0 {
                    break;
                }
            }
        }
        self.stats.steps += 1;
        let raw = self.read(&self.rdx, 0, total as usize * REDEX_WORDS);
        let mut rs = Vec::with_capacity(total as usize);
        let mut dec = Vec::with_capacity(total as usize);
        for c in raw.chunks_exact(REDEX_WORDS) {
            let np = rules()[c[0] as usize].premises.len();
            let mut prem = [PAD; 3];
            prem[..np].copy_from_slice(&c[2..2 + np]);
            rs.push(Redex { rule: c[0] as u8, consumer: c[1], prem });
            dec.push(c[7]);
        }
        (rs, dec)
    }

    /// The device's priorities for the last emitted redexes are also
    /// readable; exposed for conformance tests.
    pub fn priorities(&mut self, st: &State, seed: u64, step: u64) -> Vec<u64> {
        let (rs, _) = self.step(st, seed, step, false);
        let raw = self.read(&self.rdx, 0, rs.len() * REDEX_WORDS);
        raw.chunks_exact(REDEX_WORDS).map(|c| ((c[5] as u64) << 32) | c[6] as u64).collect()
    }
}

impl StepEngine for Gpu {
    fn find_and_select(&mut self, st: &State, cfg: &Config, step: u64) -> (Vec<Redex>, Vec<usize>) {
        let mis = cfg.resolver == Resolver::MaximalProgress;
        let (rs, dec) = self.step(st, cfg.seed, step, mis);
        let chosen = if mis {
            (0..rs.len()).filter(|i| dec[*i] == 1).collect()
        } else {
            choose(st, &rs, cfg, step)
        };
        (rs, chosen)
    }
}
