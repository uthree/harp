use crate::uop::{Op, UOp};
use std::fmt::Write;

pub trait Renderer {
    fn render(&self, uop: &UOp) -> String;
}

pub struct CStyleRenderer;

impl Renderer for CStyleRenderer {
    fn render(&self, uop: &UOp) -> String {
        let mut code = String::new();
        // TODO: ヘッダーや関数のシグネチャを追加する
        writeln!(&mut code, "#include <math.h>").unwrap();
        writeln!(&mut code, "void kernel_main() {{").unwrap();
        let mut renderer_impl = CStyleRendererImpl::new();
        renderer_impl.render_op(&mut code, uop, 4);
        writeln!(&mut code, "}}").unwrap();
        code
    }
}

/// レンダリング処理の内部状態を持つヘルパー構造体
struct CStyleRendererImpl {
    // 必要に応じて、変数名や型のマッピングなどを保持できる
}

impl CStyleRendererImpl {
    fn new() -> Self {
        Self {}
    }

    fn render_op(&mut self, code: &mut String, uop: &UOp, indent: usize) {
        let indent_str = " ".repeat(indent);
        match &uop.0.op {
            Op::Loop => {
                // src: [limit, body]
                let limit = self.render_expr(&uop.0.src[0]);
                // TODO: ループ変数名をUOpから取得
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
                // src: [stmt1, stmt2, ...]
                for stmt in &uop.0.src {
                    self.render_op(code, stmt, indent);
                }
            }
            Op::Store => {
                // src: [dest_var, value_expr] or [dest_buf, idx, value_expr]
                if uop.0.src.len() == 2 {
                    // 変数への代入: `float v0 = ...;`
                    let dest = self.render_expr(&uop.0.src[0]);
                    let value = self.render_expr(&uop.0.src[1]);
                    writeln!(
                        code,
                        "{}{} {} = {};",
                        indent_str, uop.0.src[1].0.dtype, dest, value
                    )
                    .unwrap();
                } else {
                    // バッファへの書き込み: `buf[idx] = ...;`
                    let dest = self.render_expr(&uop.0.src[0]);
                    let idx = self.render_expr(&uop.0.src[1]);
                    let value = self.render_expr(&uop.0.src[2]);
                    writeln!(code, "{}{}[{}] = {};", indent_str, dest, idx, value).unwrap();
                }
            }
            Op::If => {
                // src: [condition, true_branch]
                let condition = self.render_expr(&uop.0.src[0]);
                writeln!(code, "{}if ({}) {{", indent_str, condition).unwrap();
                self.render_op(code, &uop.0.src[1], indent + 4);
                writeln!(code, "{}}}", indent_str).unwrap();
            }
            _ => {
                // 式が単独で文として現れることはないと仮定
                panic!("Unexpected expression statement: {:?}", uop.0.op);
            }
        }
    }

    /// UOpをC言語の「式」としてレンダリングする
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
