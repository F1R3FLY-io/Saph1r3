//! Printing a tree back to text. Iterative, per Req. 4.5.

use crate::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Style {
    /// One line, minimal parentheses.
    Compact,
    /// Indented, one top-level par component per line.
    Pretty,
}

enum Job {
    Proc(ProcId, u32),
    Name(NameId),
    Text(&'static str),
    Owned(String),
    Indent(u32),
}

pub fn print(ast: &Ast, style: Style) -> String {
    print_proc(ast, ast.root(), style)
}

pub fn print_proc(ast: &Ast, root: ProcId, style: Style) -> String {
    let mut out = String::new();
    let mut stack: Vec<Job> = vec![Job::Proc(root, 0)];
    while let Some(job) = stack.pop() {
        match job {
            Job::Text(s) => out.push_str(s),
            Job::Owned(s) => out.push_str(&s),
            Job::Indent(d) => {
                if style == Style::Pretty {
                    out.push('\n');
                    for _ in 0..d {
                        out.push_str("  ");
                    }
                }
            }
            Job::Name(n) => match ast.name(n) {
                NameNode::Var { id } => out.push_str(ast.text(id)),
                NameNode::Wild => out.push('_'),
                NameNode::Quote(p) => {
                    out.push_str("@{");
                    stack.push(Job::Text("}"));
                    stack.push(Job::Proc(p, 0));
                }
            },
            Job::Proc(p, depth) => match ast.proc(p) {
                ProcNode::Nil => out.push_str("Nil"),
                ProcNode::Wild => out.push('_'),
                ProcNode::Var { id } => out.push_str(ast.text(id)),
                ProcNode::Error { .. } => out.push_str("<error>"),
                ProcNode::Eval { name } => {
                    out.push('*');
                    stack.push(Job::Name(name));
                }
                ProcNode::Par(s) => {
                    let kids = ast.procs_of(s).to_vec();
                    let mut jobs = Vec::new();
                    for (i, k) in kids.iter().enumerate() {
                        if i > 0 {
                            jobs.push(Job::Text(" | "));
                            if style == Style::Pretty && depth == 0 {
                                jobs.push(Job::Indent(depth));
                            }
                        }
                        jobs.push(Job::Proc(*k, depth));
                    }
                    jobs.reverse();
                    stack.extend(jobs);
                }
                ProcNode::Send {
                    chan,
                    persistent,
                    args,
                } => {
                    let kids = ast.procs_of(args).to_vec();
                    let mut jobs = Vec::new();
                    jobs.push(Job::Name(chan));
                    jobs.push(Job::Text(if persistent { "!!(" } else { "!(" }));
                    for (i, k) in kids.iter().enumerate() {
                        if i > 0 {
                            jobs.push(Job::Text(", "));
                        }
                        jobs.push(Job::Proc(*k, depth));
                    }
                    jobs.push(Job::Text(")"));
                    jobs.reverse();
                    stack.extend(jobs);
                }
                ProcNode::New { names, body } => {
                    let ids: Vec<String> =
                        ast.syms_of(names).iter().map(|s| ast.text(*s).to_string()).collect();
                    let mut jobs = vec![
                        Job::Text("new "),
                        Job::Owned(ids.join(", ")),
                        Job::Text(" in {"),
                        Job::Indent(depth + 1),
                        Job::Proc(body, depth + 1),
                        Job::Indent(depth),
                        Job::Text("}"),
                    ];
                    jobs.reverse();
                    stack.extend(jobs);
                }
                ProcNode::Receive { receipt, body } => {
                    let r = ast.receipt(receipt);
                    let mut jobs = vec![Job::Text("for(")];
                    for (i, b) in ast.binds_of(r.binds).enumerate() {
                        if i > 0 {
                            jobs.push(Job::Text(" & "));
                        }
                        let bd = ast.bind(b);
                        for (j, pat) in ast.names_of(bd.pats).iter().enumerate() {
                            if j > 0 {
                                jobs.push(Job::Text(", "));
                            }
                            jobs.push(Job::Name(*pat));
                        }
                        jobs.push(Job::Owned(format!(" {} ", bd.kind.arrow())));
                        jobs.push(Job::Name(bd.chan));
                    }
                    jobs.push(Job::Text(") {"));
                    jobs.push(Job::Indent(depth + 1));
                    jobs.push(Job::Proc(body, depth + 1));
                    jobs.push(Job::Indent(depth));
                    jobs.push(Job::Text("}"));
                    jobs.reverse();
                    stack.extend(jobs);
                }
            },
        }
    }
    out
}
