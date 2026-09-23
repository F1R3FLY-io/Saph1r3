//! Phase one (draft 3 Def. 6.3; F1R3Comb v0.6 §5.2, §8): a structural
//! homomorphism from the K1ndl1ng normal form to RC^ν. No name is
//! allocated: every apparatus name is a variable bound by the `New` of the
//! input that introduces it (Req. 5.4).
//!
//! The occurrence analysis is Table 1 with three repairs this compiler has
//! carried since v0.2 and the paper does not yet state (see README):
//! a chain's result is a *message*, so a subject position needs an adapter
//! (`br(out,c)` for a send subject, `bl(out,c)` for a receive subject) and a
//! payload position a relay (`fw(out, chan)`); links that act on public
//! names sit inside the innermost input enclosing their occurrence; and (o3)
//! against a dynamic channel relays through a proxy.
//!
//! Phase one also emits the **address supply** phase two needs: every drop
//! (o4) and every gate whose body contains an input posts one fresh name at
//! the static channel `r*`, `new a (m(r*, a))`. After phase two `a` is a leaf
//! of the instance's address, and the unit the drop or the gate releases
//! takes it as its own address. See the README for why the supply is per
//! release site rather than per unit.

use crate::{Diag, Severity};
use f1r3comb_ir::{IName, Node};
use f1r3comb_term::{Hash32, Shape};
use k1ndl1ng_norm::{Name, Node as SNode, Norm};
use std::collections::HashMap;

type Id = Vec<u32>;

fn child(id: &Id, k: u32) -> Id {
    let mut v = id.clone();
    v.push(k);
    v
}

/// `r*` as an IR name: `@k(@0)`.
pub fn r_star_ir() -> IName {
    IName::quote(Node::atom(Shape::K, vec![IName::quote(Node::Nil)]))
}

/// Call `f(binder_index, id_of_name_position, role)` for every bound-name
/// occurrence in `p` at depth `d`, in encoding order, including inside
/// quotations.
fn walk_bound(p: &Norm, d: u32, id: &Id, f: &mut dyn FnMut(i64, &Id, Role)) {
    match p.node() {
        SNode::Par(ps) => {
            for (i, q) in ps.iter().enumerate() {
                walk_bound(q, d, &child(id, i as u32), f);
            }
        }
        SNode::Send { chan, args, .. } => {
            walk_name(chan, d, &child(id, 0), Role::SendChan, f);
            if let Some(pay) = args.first() {
                let pid = child(id, 1);
                match pay.node() {
                    SNode::Eval(n @ Name::Bound(_)) => walk_name(n, d, &pid, Role::Payload, f),
                    _ => walk_bound(pay, d, &pid, f),
                }
            }
        }
        SNode::Receive { binds, body } => {
            if let Some(b) = binds.first() {
                walk_name(&b.chan, d, &child(id, 0), Role::RecvChan, f);
            }
            walk_bound(body, d + 1, &child(id, 1), f);
        }
        SNode::Eval(n) => walk_name(n, d, &child(id, 0), Role::Eval, f),
        _ => {}
    }
}

fn walk_name(n: &Name, d: u32, id: &Id, role: Role, f: &mut dyn FnMut(i64, &Id, Role)) {
    match n {
        Name::Bound(i) => f(d as i64 - 1 - *i as i64, id, role),
        Name::Quote(q) => walk_bound(q, d, &child(id, 0), f),
        _ => {}
    }
}

/// Binder indices `< limit` referenced from `p` (at depth `d`).
fn escaping(p: &Norm, d: u32, limit: u32) -> Vec<u32> {
    let mut v = Vec::new();
    walk_bound(p, d, &Vec::new(), &mut |b, _, _| {
        if b >= 0 && (b as u32) < limit {
            v.push(b as u32);
        }
    });
    v.sort_unstable();
    v.dedup();
    v
}

/// Does `p` contain an input anywhere, including under quotation?
pub fn contains_receive(p: &Norm) -> bool {
    let mut work = vec![p.clone()];
    while let Some(p) = work.pop() {
        match p.node() {
            SNode::Receive { .. } => return true,
            SNode::Par(ps) => work.extend(ps.iter().cloned()),
            SNode::Send { chan, args, .. } => {
                if let Name::Quote(q) = chan {
                    work.push(q.clone());
                }
                work.extend(args.iter().cloned());
            }
            SNode::Eval(Name::Quote(q)) => work.push(q.clone()),
            _ => {}
        }
    }
    false
}

/// Does `p` contain an input at its own level (not under quotation)? Such a
/// process, stored behind a gate, is a unit with bound names.
fn has_receive_at_top(p: &Norm) -> bool {
    match p.node() {
        SNode::Receive { .. } => true,
        SNode::Par(ps) => ps.iter().any(has_receive_at_top),
        _ => false,
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Role {
    SendChan,
    RecvChan,
    Payload,
    Eval,
}

#[derive(Clone)]
enum Render {
    Proxy(IName),
    Deleted,
    Relay(IName),
    Leaf(IName),
}

#[derive(Default)]
struct Ctx {
    renders: HashMap<Id, Render>,
    pending: HashMap<Id, Vec<Node>>,
}

enum Event {
    Direct(Role, Id, Id),
    Leaf(Id),
    Deliver(Id),
    Site(Role, Id, Norm, u32, Id),
}

/// Marking bookkeeping (Req. 5.17): each marked variable belongs to one
/// (receive, occurrence) flow.
#[derive(Default)]
pub struct Marks {
    pub group: HashMap<u32, (u32, u32)>,
    pub receives: u32,
}

#[derive(Default)]
pub struct Phase1 {
    next: u32,
    memo: HashMap<Hash32, Node>,
    pub diags: Vec<Diag>,
    pub marks: Marks,
}

impl Phase1 {
    fn err(&mut self, code: &'static str, message: String) {
        if !self.diags.iter().any(|d| d.code == code && d.message == message) {
            self.diags.push(Diag { code, severity: Severity::Error, span: None, message });
        }
    }

    fn fresh(&mut self, vars: &mut Vec<u32>) -> u32 {
        self.next += 1;
        vars.push(self.next);
        self.next
    }

    fn mark(&mut self, v: u32, recv: u32, group: u32) {
        self.marks.group.insert(v, (recv, group));
    }

    /// `[[P]]` for a closed `P`, memoised on the source normal form.
    pub fn closed(&mut self, p: &Norm) -> Node {
        let h = p.hash();
        if let Some(n) = self.memo.get(&h) {
            return n.clone();
        }
        let mut ctx = Ctx::default();
        let mut out = Vec::new();
        self.proc_(&mut ctx, p, &Vec::new(), 0, &mut out);
        let n = Node::par(out);
        self.memo.insert(h, n.clone());
        n
    }

    /// `[[@Q]] = @[[Q]]` for a closed `Q`.
    fn static_name(&mut self, q: &Norm) -> IName {
        IName::quote(self.closed(q))
    }

    fn name_at(&mut self, ctx: &Ctx, n: &Name, id: &Id, d: u32) -> IName {
        if let Some(Render::Proxy(c)) = ctx.renders.get(id) {
            return c.clone();
        }
        match n {
            Name::Quote(q) if escaping(q, d, d).is_empty() => self.static_name(q),
            _ => {
                self.err("comb-occ-unclassified", format!("internal: name position {id:?} has no rendering"));
                IName::quote(Node::Nil)
            }
        }
    }

    fn proc_(&mut self, ctx: &mut Ctx, p: &Norm, id: &Id, d: u32, out: &mut Vec<Node>) {
        match p.node() {
            SNode::Nil => {}
            SNode::Par(ps) => {
                for (i, q) in ps.iter().enumerate() {
                    self.proc_(ctx, q, &child(id, i as u32), d, out);
                }
            }
            SNode::Send { chan, args, .. } => {
                let target = self.name_at(ctx, chan, &child(id, 0), d);
                let pid = child(id, 1);
                match ctx.renders.get(&pid).cloned() {
                    Some(Render::Deleted) => {}
                    Some(Render::Relay(c)) => out.push(Node::atom(Shape::Fw, vec![c, target])),
                    Some(_) => self.err("comb-occ-unclassified", format!("internal: payload {pid:?}")),
                    None => {
                        let pay = &args[0];
                        if escaping(pay, d, d).is_empty() {
                            let v = self.static_name(pay);
                            out.push(Node::atom(Shape::M, vec![target, v]));
                        } else {
                            self.err("comb-occ-unclassified", format!("internal: dynamic payload {pid:?} unrendered"));
                        }
                    }
                }
            }
            SNode::Eval(_) => match ctx.renders.get(&child(id, 0)).cloned() {
                Some(Render::Proxy(c)) => out.push(Node::atom(Shape::E, vec![c])),
                _ => self.err("comb-occ-unclassified", format!("internal: drop at {id:?} unrendered")),
            },
            SNode::Receive { binds, body } => {
                let subj = self.name_at(ctx, &binds[0].chan, &child(id, 0), d);
                let n = self.receive(ctx, body, subj, id, d);
                out.push(n);
            }
            _ => self.err("comb-level", "internal: non-K0 node reached the lowering".into()),
        }
    }

    /// `[[for(y <- x){body}]]` at depth `d` (the receive's own binder index
    /// is `d`), with subject `x` already rendered.
    fn receive(&mut self, ctx: &mut Ctx, body: &Norm, subj: IName, id: &Id, d: u32) -> Node {
        let me = d;
        let recv = self.marks.receives;
        self.marks.receives += 1;
        let bid = child(id, 1);
        let mut events: Vec<Event> = Vec::new();
        self.events(body, d + 1, &bid, me, &bid, &mut events);
        let k = events.iter().filter(|e| !matches!(e, Event::Site(..))).count();

        let mut vars: Vec<u32> = Vec::new();
        let b = self.fresh(&mut vars);
        let c = self.fresh(&mut vars);
        let ps: Vec<u32> = (0..=k).map(|_| self.fresh(&mut vars)).collect();
        let qs: Vec<u32> = (1..k).map(|_| self.fresh(&mut vars)).collect();
        for (i, p) in ps.iter().enumerate().skip(1) {
            self.mark(*p, recv, i as u32);
        }
        let v = IName::Var;

        // --- phase 1 of the receive: occurrences, in encoding order (Req. 5.6)
        let mut links: Vec<Node> = Vec::new();
        let mut i = 0usize;
        let mut sites = Vec::new();
        let route = |ctx: &mut Ctx, links: &mut Vec<Node>, scope: &Id, a: Node| {
            if *scope == bid {
                links.push(a);
            } else {
                ctx.pending.entry(scope.clone()).or_default().push(a);
            }
        };
        for e in events {
            match e {
                Event::Direct(role, oid, scope) => {
                    i += 1;
                    let p = v(ps[i]);
                    let ci = self.fresh(&mut vars);
                    self.mark(ci, recv, i as u32);
                    match role {
                        Role::RecvChan => {
                            route(ctx, &mut links, &scope, Node::atom(Shape::Bl, vec![p, v(ci)]));
                            ctx.renders.insert(oid, Render::Proxy(v(ci)));
                        }
                        Role::SendChan => {
                            links.push(Node::atom(Shape::Br, vec![p, v(ci)]));
                            ctx.renders.insert(oid, Render::Proxy(v(ci)));
                        }
                        Role::Eval => {
                            links.push(Node::atom(Shape::Fw, vec![p, v(ci)]));
                            ctx.renders.insert(oid, Render::Proxy(v(ci)));
                            // the address the dropped unit will run at
                            let a = self.fresh(&mut vars);
                            links.push(Node::atom(Shape::M, vec![r_star_ir(), v(a)]));
                        }
                        Role::Payload => {
                            let mut cid = oid.clone();
                            *cid.last_mut().unwrap() = 0;
                            match self.static_chan_of(body, &bid, &cid, d + 1) {
                                Some(x) => {
                                    route(ctx, &mut links, &scope, Node::atom(Shape::Fw, vec![p, x]));
                                    ctx.renders.insert(oid, Render::Deleted);
                                }
                                None => {
                                    links.push(Node::atom(Shape::Fw, vec![p, v(ci)]));
                                    ctx.renders.insert(oid, Render::Relay(v(ci)));
                                }
                            }
                        }
                    }
                }
                Event::Leaf(oid) => {
                    i += 1;
                    ctx.renders.insert(oid, Render::Leaf(v(ps[i])));
                }
                Event::Deliver(oid) => {
                    i += 1;
                    let ci = self.fresh(&mut vars);
                    self.mark(ci, recv, i as u32);
                    links.push(Node::atom(Shape::Fw, vec![v(ps[i]), v(ci)]));
                    ctx.renders.insert(oid, Render::Leaf(v(ci)));
                }
                Event::Site(role, sid, q, sd, scope) => sites.push((role, sid, q, sd, scope)),
            }
        }
        debug_assert_eq!(i, k);

        // --- chains for the dynamic names this receive owns (Req. 8.4)
        for (role, sid, q, sd, scope) in sites {
            if contains_receive(&q) {
                self.err(
                    "comb-o5-unsupported",
                    "a quotation containing both an input and a bound name: rebuilding it at run time needs \
                     the phase-two image of the input, which no member of the constructor family can build"
                        .into(),
                );
                continue;
            }
            let qid = if role == Role::Payload { sid.clone() } else { child(&sid, 0) };
            let mut chain = Vec::new();
            let out_ch = self.build_proc(ctx, &mut vars, recv, &q, &qid, sd, sd, &mut chain);
            match role {
                Role::SendChan => {
                    let cc = self.fresh(&mut vars);
                    chain.push(Node::atom(Shape::Br, vec![out_ch, v(cc)]));
                    ctx.renders.insert(sid, Render::Proxy(v(cc)));
                }
                Role::RecvChan => {
                    let cc = self.fresh(&mut vars);
                    chain.push(Node::atom(Shape::Bl, vec![out_ch, v(cc)]));
                    ctx.renders.insert(sid, Render::Proxy(v(cc)));
                }
                Role::Payload => {
                    ctx.renders.insert(sid, Render::Relay(out_ch));
                }
                Role::Eval => self.err("comb-occ-unclassified", "internal: dynamic drop".into()),
            }
            for a in chain {
                route(ctx, &mut links, &scope, a);
            }
        }

        // --- the body, stored behind the gate
        let mut batoms = Vec::new();
        self.proc_(ctx, body, &bid, d + 1, &mut batoms);
        if let Some(extra) = ctx.pending.remove(&bid) {
            batoms.extend(extra);
        }
        if has_receive_at_top(body) {
            // the stored body is a unit with bound names: its address
            let a = self.fresh(&mut vars);
            links.push(Node::atom(Shape::M, vec![r_star_ir(), v(a)]));
        }
        let big_b = Node::par(batoms);

        // --- distributor and gate (Def. 6.2)
        let mut parts = Vec::new();
        if k == 0 {
            parts.push(Node::atom(Shape::S, vec![subj, v(b), v(c)]));
        } else {
            let q1 = if k == 1 { ps[1] } else { qs[0] };
            parts.push(Node::atom(Shape::D, vec![subj, v(ps[0]), v(q1)]));
            for j in 1..k {
                let from = qs[j - 1];
                let to = if j + 1 == k { ps[k] } else { qs[j] };
                parts.push(Node::atom(Shape::D, vec![v(from), v(ps[j]), v(to)]));
            }
            parts.push(Node::atom(Shape::S, vec![v(ps[0]), v(b), v(c)]));
        }
        parts.push(Node::q(v(b), big_b));
        parts.push(Node::atom(Shape::E, vec![v(c)]));
        parts.extend(links);
        Node::New { vars, body: Box::new(Node::par(parts)) }
    }

    fn static_chan_of(&mut self, body: &Norm, bid: &Id, cid: &Id, bd: u32) -> Option<IName> {
        let (chan, d) = find_chan(body, bid, cid, bd)?;
        match &chan {
            Name::Quote(q) if escaping(q, d, d).is_empty() => Some(self.static_name(q)),
            _ => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn events(&mut self, p: &Norm, d: u32, id: &Id, me: u32, scope: &Id, out: &mut Vec<Event>) {
        match p.node() {
            SNode::Par(ps) => {
                for (i, q) in ps.iter().enumerate() {
                    self.events(q, d, &child(id, i as u32), me, scope, out);
                }
            }
            SNode::Send { chan, args, .. } => {
                self.name_events(chan, d, &child(id, 0), Role::SendChan, me, scope, out);
                if let Some(pay) = args.first() {
                    let pid = child(id, 1);
                    match pay.node() {
                        SNode::Eval(n @ Name::Bound(_)) => self.name_events(n, d, &pid, Role::Payload, me, scope, out),
                        _ => self.quote_events(pay, d, &pid, &pid, Role::Payload, me, scope, out),
                    }
                }
            }
            SNode::Receive { binds, body } => {
                if let Some(b) = binds.first() {
                    self.name_events(&b.chan, d, &child(id, 0), Role::RecvChan, me, scope, out);
                }
                let bid = child(id, 1);
                self.events(body, d + 1, &bid, me, &bid, out);
            }
            SNode::Eval(n) => self.name_events(n, d, &child(id, 0), Role::Eval, me, scope, out),
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn name_events(&mut self, n: &Name, d: u32, id: &Id, role: Role, me: u32, scope: &Id, out: &mut Vec<Event>) {
        match n {
            Name::Bound(i) => {
                if d as i64 - 1 - *i as i64 == me as i64 {
                    out.push(Event::Direct(role, id.clone(), scope.clone()));
                }
            }
            Name::Quote(q) => self.quote_events(q, d, id, &child(id, 0), role, me, scope, out),
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn quote_events(&mut self, q: &Norm, d: u32, nid: &Id, qid: &Id, role: Role, me: u32, scope: &Id, out: &mut Vec<Event>) {
        let refs = escaping(q, d, d);
        if !refs.contains(&me) {
            return;
        }
        let owner = *refs.iter().max().unwrap();
        if owner == me {
            out.push(Event::Site(role, nid.clone(), q.clone(), d, scope.clone()));
        }
        let mut leaves = Vec::new();
        walk_bound(q, d, qid, &mut |b, lid, _| {
            if b == me as i64 {
                leaves.push(lid.clone());
            }
        });
        for lid in leaves {
            out.push(if owner == me { Event::Leaf(lid) } else { Event::Deliver(lid) });
        }
    }

    /// Build, from leaves, the channel on which `@[[p']]` will be delivered,
    /// `p'` being `p` with its leaves substituted (Prop. 3.12). `p` has no
    /// input (checked by the caller).
    #[allow(clippy::too_many_arguments)]
    fn build_proc(&mut self, ctx: &Ctx, vars: &mut Vec<u32>, recv: u32, p: &Norm, id: &Id, d: u32, limit: u32, links: &mut Vec<Node>) -> IName {
        if escaping(p, d, limit).is_empty() {
            let q = self.closed(p);
            return self.constant(vars, IName::quote(q), links);
        }
        match p.node() {
            SNode::Eval(Name::Bound(_)) => self.leaf(ctx, &child(id, 0)),
            SNode::Par(ps) => {
                let mut statics = Vec::new();
                let mut chans = Vec::new();
                for (i, q) in ps.iter().enumerate() {
                    if escaping(q, d, limit).is_empty() {
                        statics.push(self.closed(q));
                    } else {
                        chans.push(self.build_proc(ctx, vars, recv, q, &child(id, i as u32), d, limit, links));
                    }
                }
                let mut acc: Vec<IName> = Vec::new();
                if !statics.is_empty() {
                    let k = self.constant(vars, IName::quote(Node::par(statics)), links);
                    acc.push(k);
                }
                acc.extend(chans);
                let mut cur = acc[0].clone();
                for nxt in acc.into_iter().skip(1) {
                    let o = self.fresh(vars);
                    self.mark_join(o, &cur, &nxt);
                    links.push(Node::atom(Shape::ConsPar, vec![cur, nxt, IName::Var(o)]));
                    cur = IName::Var(o);
                }
                cur
            }
            SNode::Send { chan, args, .. } => {
                let a = self.build_name(ctx, vars, recv, chan, &child(id, 0), d, limit, links);
                let pid = child(id, 1);
                let pay = &args[0];
                let b = match pay.node() {
                    SNode::Eval(Name::Bound(_)) => self.leaf(ctx, &pid),
                    _ if escaping(pay, d, limit).is_empty() => {
                        let v = self.static_name(pay);
                        self.constant(vars, v, links)
                    }
                    _ => self.build_proc(ctx, vars, recv, pay, &pid, d, limit, links),
                };
                let o = self.fresh(vars);
                self.mark_join(o, &a, &b);
                links.push(Node::atom(Shape::ConsM, vec![a, b, IName::Var(o)]));
                IName::Var(o)
            }
            _ => {
                self.err("comb-occ-unclassified", format!("internal: cannot rebuild node at {id:?}"));
                IName::quote(Node::Nil)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_name(&mut self, ctx: &Ctx, vars: &mut Vec<u32>, recv: u32, n: &Name, id: &Id, d: u32, limit: u32, links: &mut Vec<Node>) -> IName {
        match n {
            Name::Bound(_) => self.leaf(ctx, id),
            Name::Quote(q) if escaping(q, d, limit).is_empty() => {
                let v = self.static_name(q);
                self.constant(vars, v, links)
            }
            Name::Quote(q) => self.build_proc(ctx, vars, recv, q, &child(id, 0), d, limit, links),
            _ => {
                self.err("comb-open", "internal: open name in a chain".into());
                IName::quote(Node::Nil)
            }
        }
    }

    /// A chain node's output belongs to the flow of its first marked input.
    fn mark_join(&mut self, o: u32, a: &IName, b: &IName) {
        let g = |n: &IName, m: &Marks| match n {
            IName::Var(x) => m.group.get(x).copied(),
            _ => None,
        };
        if let Some(gr) = g(a, &self.marks).or(g(b, &self.marks)) {
            self.marks.group.insert(o, gr);
        }
    }

    fn leaf(&mut self, ctx: &Ctx, id: &Id) -> IName {
        match ctx.renders.get(id) {
            Some(Render::Leaf(c)) => c.clone(),
            _ => {
                self.err("comb-occ-unclassified", format!("internal: leaf {id:?} has no channel"));
                IName::quote(Node::Nil)
            }
        }
    }

    /// A constant message `m(k, v)` on a fresh channel `k`.
    fn constant(&mut self, vars: &mut Vec<u32>, v: IName, links: &mut Vec<Node>) -> IName {
        let k = self.fresh(vars);
        links.push(Node::atom(Shape::M, vec![IName::Var(k), v]));
        IName::Var(k)
    }
}

/// Find the channel name of the send whose chan position is `cid`.
fn find_chan(p: &Norm, id: &Id, cid: &Id, d: u32) -> Option<(Name, u32)> {
    if !cid.starts_with(id) {
        return None;
    }
    let rest = &cid[id.len()..];
    let mut cur = p.clone();
    let mut depth = d;
    let mut k = 0usize;
    loop {
        let step = *rest.get(k)?;
        match cur.node().clone() {
            SNode::Par(ps) => {
                cur = ps.get(step as usize)?.clone();
                k += 1;
            }
            SNode::Send { chan, args, .. } => {
                if k + 1 == rest.len() && step == 0 {
                    return Some((chan, depth));
                }
                match (step, &chan) {
                    (0, Name::Quote(q)) => {
                        if rest.get(k + 1) != Some(&0) {
                            return None;
                        }
                        cur = q.clone();
                        k += 2;
                    }
                    (1, _) => {
                        cur = args.first()?.clone();
                        k += 1;
                    }
                    _ => return None,
                }
            }
            SNode::Receive { binds, body } => match step {
                0 => match &binds[0].chan {
                    Name::Quote(q) if rest.get(k + 1) == Some(&0) => {
                        cur = q.clone();
                        k += 2;
                    }
                    _ => return None,
                },
                1 => {
                    cur = body.clone();
                    depth += 1;
                    k += 1;
                }
                _ => return None,
            },
            SNode::Eval(Name::Quote(q)) => {
                cur = q;
                k += 2;
            }
            _ => return None,
        }
    }
}
