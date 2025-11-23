use crate::uop::{UOp, UOpKind};
use std::collections::HashSet;

/// OpenCLコード生成器
pub struct OpenCLCodegen {
    indent_level: usize,
}

impl OpenCLCodegen {
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }

    /// UOpからOpenCLカーネルコードを生成
    pub fn generate_kernel(&mut self, uop: &UOp, kernel_name: &str) -> String {
        let mut code = String::new();

        // カーネル関数のシグネチャ
        code.push_str(&format!("__kernel void {}(\n", kernel_name));
        code.push_str("    __global float* input,\n");
        code.push_str("    __global float* output,\n");
        code.push_str("    int size\n");
        code.push_str(") {\n");

        self.indent_level = 1;

        // カーネルの本体を生成
        let body = self.generate(uop);
        code.push_str(&body);

        code.push_str("}\n");

        code
    }

    /// UOpからコードを生成
    pub fn generate(&mut self, uop: &UOp) -> String {
        match &uop.0.op {
            UOpKind::Loop {
                var,
                start,
                end,
                parallel,
                ..
            } => self.generate_loop(var, *start, *end, *parallel, &uop.0.src[0]),

            UOpKind::Load { buffer, index } => self.generate_load(buffer, index.as_ref()),

            UOpKind::Store { buffer, index } => {
                self.generate_store(buffer, index.as_ref(), &uop.0.src[0])
            }

            UOpKind::ThreadIdx { dim } => {
                format!("get_global_id({})", dim)
            }

            UOpKind::GroupIdx { dim } => {
                format!("get_group_id({})", dim)
            }

            UOpKind::Var { name } => name.clone(),

            UOpKind::Const { value } => {
                if value.fract() == 0.0 && value.abs() < 1e10 {
                    format!("{:.1}f", value)
                } else {
                    format!("{}f", value)
                }
            }

            UOpKind::Add => {
                let lhs = self.generate(&uop.0.src[0]);
                let rhs = self.generate(&uop.0.src[1]);
                format!("({} + {})", lhs, rhs)
            }

            UOpKind::Mul => {
                let lhs = self.generate(&uop.0.src[0]);
                let rhs = self.generate(&uop.0.src[1]);
                format!("({} * {})", lhs, rhs)
            }

            UOpKind::Max => {
                let lhs = self.generate(&uop.0.src[0]);
                let rhs = self.generate(&uop.0.src[1]);
                format!("fmax({}, {})", lhs, rhs)
            }

            UOpKind::Recip => {
                let arg = self.generate(&uop.0.src[0]);
                format!("(1.0f / {})", arg)
            }

            UOpKind::Sqrt => {
                let arg = self.generate(&uop.0.src[0]);
                format!("sqrt({})", arg)
            }

            UOpKind::Sequence => {
                let mut code = String::new();
                for op in &uop.0.src {
                    code.push_str(&self.generate(op));
                }
                code
            }

            UOpKind::Barrier => {
                format!("{}barrier(CLK_LOCAL_MEM_FENCE);\n", self.indent())
            }

            _ => {
                // その他の演算は未実装
                format!("{}/* TODO: {:?} */\n", self.indent(), uop.0.op)
            }
        }
    }

    fn generate_loop(
        &mut self,
        var: &str,
        start: usize,
        end: usize,
        parallel: bool,
        body: &UOp,
    ) -> String {
        let mut code = String::new();

        if parallel {
            // 並列ループ（GPUスレッド）
            code.push_str(&format!(
                "{}int {} = get_global_id(0);\n",
                self.indent(),
                var
            ));
            code.push_str(&format!("{}if ({} < {}) {{\n", self.indent(), var, end));
            self.indent_level += 1;
            code.push_str(&self.generate(body));
            self.indent_level -= 1;
            code.push_str(&format!("{}}}\n", self.indent()));
        } else {
            // シーケンシャルループ
            code.push_str(&format!(
                "{}for (int {} = {}; {} < {}; {}++) {{\n",
                self.indent(),
                var,
                start,
                var,
                end,
                var
            ));
            self.indent_level += 1;
            code.push_str(&self.generate(body));
            self.indent_level -= 1;
            code.push_str(&format!("{}}}\n", self.indent()));
        }

        code
    }

    fn generate_load(&mut self, buffer: &str, index: Option<&Box<UOp>>) -> String {
        if let Some(idx) = index {
            let idx_code = self.generate(idx);
            format!("{}[(int){}]", buffer, idx_code)
        } else {
            buffer.to_string()
        }
    }

    fn generate_store(&mut self, buffer: &str, index: Option<&Box<UOp>>, value: &UOp) -> String {
        let value_code = self.generate(value);
        let mut code = String::new();

        if let Some(idx) = index {
            let idx_code = self.generate(idx);
            code.push_str(&format!(
                "{}{}[(int){}] = {};\n",
                self.indent(),
                buffer,
                idx_code,
                value_code
            ));
        } else {
            code.push_str(&format!("{}{} = {};\n", self.indent(), buffer, value_code));
        }

        code
    }

    fn indent(&self) -> String {
        "    ".repeat(self.indent_level)
    }
}

impl Default for OpenCLCodegen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering::Lowerer;
    use crate::uop::ElementwiseOp;
    use harp::DType;

    #[test]
    fn test_codegen_simple_add() {
        let mut codegen = OpenCLCodegen::new();

        let a = UOp::input("a", vec![10], DType::F32);
        let b = UOp::input("b", vec![10], DType::F32);
        let add = UOp::elementwise(ElementwiseOp::Add, vec![a, b], DType::F32);

        // Loweringしてからコード生成
        let lowerer = Lowerer::new(256);
        let lowered = lowerer.lower(&add);

        let code = codegen.generate_kernel(&lowered, "add_kernel");

        println!("Generated OpenCL kernel:\n{}", code);

        // 基本的なチェック
        assert!(code.contains("__kernel"));
        assert!(code.contains("add_kernel"));
        assert!(code.contains("get_global_id"));
    }

    #[test]
    fn test_codegen_manual() {
        let mut codegen = OpenCLCodegen::new();

        // 手動でUOpを構築
        let tid = UOp::thread_idx(0, DType::F32);
        let a = UOp::load("input".to_string(), Some(tid.clone()), DType::F32);
        let b = UOp::const_val(2.0, DType::F32);
        let result = UOp::mul(a, b);
        let store = UOp::store("output".to_string(), Some(tid.clone()), result);
        let kernel = UOp::loop_op("tid".to_string(), 0, 100, store, true);

        let code = codegen.generate_kernel(&kernel, "mul_kernel");

        println!("Generated OpenCL kernel (manual):\n{}", code);

        assert!(code.contains("__kernel"));
        assert!(code.contains("mul_kernel"));
        assert!(code.contains("* 2.0f"));
    }
}
