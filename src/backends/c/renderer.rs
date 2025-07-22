use crate::backends::Renderer;
use crate::uop::{Op, UOp};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

#[derive(Debug)]
pub struct CStyleRenderer;

impl Renderer for CStyleRenderer {
    fn render(&self, uop: &UOp) -> String {
        let mut code = String::new();
        let mut renderer_impl = CStyleRendererImpl::new();
        renderer_impl.collect_vars(uop);

        writeln!(&mut code, "#include <math.h>").unwrap();
        writeln!(&mut code, "void kernel_main(void** bufs, int* ints) {{").unwrap();

        // バッファと整数引数をローカル変数にキャストしてわかりやすくする
        let mut sorted_args: Vec<_> = renderer_impl.args.iter().collect();
        sorted_args.sort_by_key(|(name, _)| (*name).clone());
        let mut buf_idx = 0;
        let mut int_idx = 0;
        for (name, dtype) in sorted_args {
            if renderer_impl.defined_vars.contains(name.as_str()) {
                continue;
            }
            if renderer_impl.buffers.contains(name.as_str()) {
                writeln!(&mut code, "    {}* {} = bufs[{}];", dtype, name, buf_idx).unwrap();
                buf_idx += 1;
            } else {
                writeln!(&mut code, "    int {} = ints[{}];", name, int_idx).unwrap();
                int_idx += 1;
            }
        }

        renderer_impl.render_op(&mut code, uop, 4);
        writeln!(&mut code, "}}").unwrap();
        code
    }
}

struct CStyleRendererImpl {
    args: HashMap<String, crate::dtype::DType>,
    buffers: HashSet<String>,
    defined_vars: HashSet<String>,
}

impl CStyleRendererImpl {
    fn new() -> Self {
        Self {
            args: HashMap::new(),
            buffers: HashSet::new(),
            defined_vars: HashSet::new(),
        }
    }

    fn collect_vars(&mut self, uop: &UOp) {
        if let Op::Var(name) = &uop.0.op {
            if !self.args.contains_key(name) {
                self.args.insert(name.clone(), uop.0.dtype.clone());
            }
        }
        match &uop.0.op {
            Op::Load | Op::Store => {
                if let Some(first_src) = uop.0.src.get(0) {
                    if let Op::Var(name) = &first_src.0.op {
                        self.buffers.insert(name.clone());
                    }
                }
            }
            _ => {}
        }
        if let Op::Store = &uop.0.op {
            if uop.0.src.len() == 2 {
                if let Some(dest) = uop.0.src.get(0) {
                    if let Op::Var(name) = &dest.0.op {
                        self.defined_vars.insert(name.clone());
                    }
                }
            }
        }
        for src in &uop.0.src {
            self.collect_vars(src);
        }
    }

    fn render_op(&mut self, code: &mut String, uop: &UOp, indent: usize) {
        let indent_str = " ".repeat(indent);
        match &uop.0.op {
            Op::Loop => {
                let limit = self.render_expr(&uop.0.src[0]);
                writeln!(
                    code,
                    "{}for (int i = 0; i < {}; ++i) {{",
                    indent_str, limit
                )
                .unwrap();
                self.render_op(code, &uop.0.src[1], indent + 4);
                writeln!(code, "{}}}", indent_str).unwrap();
            }
            Op::Block => {
                for stmt in &uop.0.src {
                    self.render_op(code, stmt, indent);
                }
            }
            Op::Store => {
                if uop.0.src.len() == 2 {
                    let dest = self.render_expr(&uop.0.src[0]);
                    let value = self.render_expr(&uop.0.src[1]);
                    writeln!(
                        code,
                        "{}{} {} = {};",
                        indent_str, uop.0.src[1].0.dtype, dest, value
                    )
                    .unwrap();
                } else {
                    let dest = self.render_expr(&uop.0.src[0]);
                    let idx = self.render_expr(&uop.0.src[1]);
                    let value = self.render_expr(&uop.0.src[2]);
                    writeln!(code, "{}{}[{}] = {};", indent_str, dest, idx, value).unwrap();
                }
            }
            Op::If => {
                let condition = self.render_expr(&uop.0.src[0]);
                writeln!(code, "{}if ({}) {{", indent_str, condition).unwrap();
                self.render_op(code, &uop.0.src[1], indent + 4);
                writeln!(code, "{}}}", indent_str).unwrap();
            }
            _ => panic!("Unexpected expression statement: {:?}", uop.0.op),
        }
    }

    fn render_expr(&mut self, uop: &UOp) -> String {
        match &uop.0.op {
            Op::Add => format!("({} + {})", self.render_expr(&uop.0.src[0]), self.render_expr(&uop.0.src[1])),
            Op::Mul => format!("({} * {})", self.render_expr(&uop.0.src[0]), self.render_expr(&uop.0.src[1])),
            Op::Recip => format!("(1.0f / {})", self.render_expr(&uop.0.src[0])),
            Op::Rem => format!("({} % {})", self.render_expr(&uop.0.src[0]), self.render_expr(&uop.0.src[1])),
            Op::Load => format!("{}[{}]", self.render_expr(&uop.0.src[0]), self.render_expr(&uop.0.src[1])),
            Op::Const(num) => format!("{}", num),
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
