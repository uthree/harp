use crate::backends::Renderer;
use crate::uop::{Op, UOp};
use log::debug;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt::Write;

#[derive(Debug)]
pub struct CStyleRenderer;

impl Renderer for CStyleRenderer {
    fn render(&self, uops: &[UOp]) -> String {
        let mut code = String::new();
        let mut context = CStyleRenderContext::new();
        context.collect_vars(uops);

        writeln!(&mut code, "#include <math.h>").unwrap();
        writeln!(&mut code, "void kernel_main(void** bufs, int* ints) {{").unwrap();

        // Buffer and integer arguments are cast to local variables for clarity.
        let mut sorted_args: Vec<_> = context.args.iter().collect();
        sorted_args.sort_by(|(a, _), (b, _)| {
            let a_is_data = a.starts_with("data");
            let b_is_data = b.starts_with("data");
            if a_is_data && b_is_data {
                let a_num: usize = a[4..].parse().unwrap();
                let b_num: usize = b[4..].parse().unwrap();
                a_num.cmp(&b_num)
            } else {
                a.cmp(b)
            }
        });
        let mut buf_idx = 0;
        let mut int_idx = 0;
        for (name, dtype) in sorted_args {
            if context.defined_vars.contains(name.as_str()) {
                continue;
            }
            if context.buffers.contains(name.as_str()) {
                writeln!(&mut code, "    {dtype}* {name} = bufs[{buf_idx}];").unwrap();
                buf_idx += 1;
            } else {
                writeln!(&mut code, "    int {name} = ints[{int_idx}];").unwrap();
                int_idx += 1;
            }
        }

        context.render_ops(&mut code, uops, 4);
        writeln!(&mut code, "}}").unwrap();
        debug!("Rendered C code:\n{code}");
        code
    }
}

struct CStyleRenderContext {

    args: FxHashMap<String, crate::dtype::DType>,
    buffers: FxHashSet<String>,
    defined_vars: FxHashSet<String>,
}

impl CStyleRenderContext {
    fn new() -> Self {
        Self {
            args: FxHashMap::default(),
            buffers: FxHashSet::default(),
            defined_vars: FxHashSet::default(),
        }
    }

    fn collect_vars(&mut self, uops: &[UOp]) {
        for uop in uops {
            // Collect variables from the operation itself
            if let Op::Var(name) = &uop.0.op {
                if !self.args.contains_key(name) {
                    self.args.insert(name.clone(), uop.0.dtype.clone());
                }
            }

            // Handle special cases for Load and Store to identify buffer variables
            if let Op::Load = &uop.0.op {
                if let Some(first_src) = uop.0.src.first() {
                    if let Op::Var(name) = &first_src.0.op {
                        self.buffers.insert(name.clone());
                    }
                }
            }

            match &uop.0.op {
                Op::Store => {
                    // The first source is the destination buffer/variable
                    if let Some(dest) = uop.0.src.first() {
                        if let Op::Var(name) = &dest.0.op {
                            self.buffers.insert(name.clone());
                            if !self.args.contains_key(name) {
                                self.args.insert(name.clone(), dest.0.dtype.clone());
                            }
                            // If it's a variable declaration, mark it as defined
                            if uop.0.src.len() == 2 {
                                self.defined_vars.insert(name.clone());
                            }
                        }
                    }
                    // The last source is the expression to be stored.
                    if let Some(expr) = uop.0.src.last() {
                        self.collect_vars_recursive(expr);
                    }
                }
                // For other operations, recursively collect from all sources.
                _ => {
                    for src in &uop.0.src {
                        self.collect_vars_recursive(src);
                    }
                }
            }
        }
    }

    fn collect_vars_recursive(&mut self, uop: &UOp) {
        // If the uop is a variable, add it to the arguments list.
        if let Op::Var(name) = &uop.0.op {
            if !self.args.contains_key(name) {
                self.args.insert(name.clone(), uop.0.dtype.clone());
            }
        }

        // If we encounter a Load operation, the first source is a buffer.
        if let Op::Load = &uop.0.op {
            if let Some(buffer_var) = uop.0.src.first() {
                if let Op::Var(name) = &buffer_var.0.op {
                    self.buffers.insert(name.clone());
                }
            }
        }

        // Recursively traverse the sources of the expression.
        for src in &uop.0.src {
            self.collect_vars_recursive(src);
        }
    }

    fn render_ops(&mut self, code: &mut String, uops: &[UOp], initial_indent: usize) {
        let mut indent = initial_indent;
        for uop in uops {
            let indent_str = " ".repeat(indent);
            match &uop.0.op {
                Op::LoopStart => {
                    let var = self.render_expr(&uop.0.src[0]);
                    let limit = self.render_expr(&uop.0.src[1]);
                    writeln!(
                        code,
                        "{indent_str}for (int {var} = 0; {var} < {limit}; ++{var}) {{"
                    )
                    .unwrap();
                    indent += 4;
                }
                Op::LoopEnd => {
                    indent -= 4;
                    writeln!(code, "{}}}", " ".repeat(indent)).unwrap();
                }
                Op::Store => {
                    if uop.0.src.len() == 2 {
                        let dest = self.render_expr(&uop.0.src[0]);
                        let value = self.render_expr(&uop.0.src[1]);
                        writeln!(
                            code,
                            "{}{} {} = {};",
                            indent_str,
                            uop.0.src[1].0.dtype,
                            dest,
                            value
                        )
                        .unwrap();
                    } else {
                        let dest = self.render_expr(&uop.0.src[0]);
                        let idx = self.render_expr(&uop.0.src[1]);
                        let value = self.render_expr(&uop.0.src[2]);
                        writeln!(code, "{indent_str}{dest}[{idx}] = {value};").unwrap();
                    }
                }
                Op::If => {
                    let condition = self.render_expr(&uop.0.src[0]);
                    writeln!(code, "{indent_str}if ({condition}) {{").unwrap();
                    // Note: This assumes the body of the if is the next instruction,
                    // which might require a more robust block handling mechanism
                    // (e.g., If/EndIf opcodes).
                    // self.render_ops(code, &uop.0.src[1..], indent + 4);
                    // For now, we assume a simple structure.
                }
                _ => panic!("Unexpected statement in kernel: {:?}", uop.0.op),
            }
        }
    }

    fn render_expr(&mut self, uop: &UOp) -> String {
        match &uop.0.op {
            Op::Add => format!("({} + {})", self.render_expr(&uop.0.src[0]), self.render_expr(&uop.0.src[1])),
            Op::Mul => format!("({} * {})", self.render_expr(&uop.0.src[0]), self.render_expr(&uop.0.src[1])),
            Op::Div => format!("({} / {})", self.render_expr(&uop.0.src[0]), self.render_expr(&uop.0.src[1])),
            Op::Recip => format!("(1.0f / {})", self.render_expr(&uop.0.src[0])),
            Op::Rem => format!("({} % {})", self.render_expr(&uop.0.src[0]), self.render_expr(&uop.0.src[1])),
            Op::Load => format!("{}[{}]", self.render_expr(&uop.0.src[0]), self.render_expr(&uop.0.src[1])),
            Op::Const(num) => format!("{num}"),
            Op::Var(name) => name.clone(),
            Op::Exp2 => format!("exp2({})", self.render_expr(&uop.0.src[0])),
            Op::Log2 => format!("log2({})", self.render_expr(&uop.0.src[0])),
            Op::Sin => format!("sin({})", self.render_expr(&uop.0.src[0])),
            Op::Sqrt => format!("sqrt({})", self.render_expr(&uop.0.src[0])),
            Op::Cast(_) => format!("({}){}", uop.0.dtype, self.render_expr(&uop.0.src[0])),
            _ => panic!("Cannot render {:?} as an expression", uop.0.op),
        }
    }
}
