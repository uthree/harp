use crate::uop::UOp;
use std::rc::Rc;

/// OpenCLコード生成器
pub struct OpenCLCodegen {
    indent_level: usize,
}

impl OpenCLCodegen {
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }

    /// UOpからOpenCLカーネルコードを生成
    pub fn generate_kernel(&mut self, uop: &Rc<UOp>, kernel_name: &str) -> String {
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
    pub fn generate(&mut self, uop: &Rc<UOp>) -> String {
        match &**uop {
            UOp::Loop {
                var,
                start,
                end,
                parallel,
                body,
                ..
            } => self.generate_loop(var, *start, *end, *parallel, body),

            UOp::Load { buffer, index, .. } => self.generate_load(buffer, index.as_ref()),

            UOp::Store {
                buffer,
                index,
                value,
                ..
            } => self.generate_store(buffer, index.as_ref(), value),

            UOp::ThreadIdx { dim, .. } => {
                format!("get_global_id({})", dim)
            }

            UOp::GroupIdx { dim, .. } => {
                format!("get_group_id({})", dim)
            }

            UOp::Var { name, .. } => name.clone(),

            UOp::Const { value, .. } => {
                if value.fract() == 0.0 && value.abs() < 1e10 {
                    format!("{:.1}f", value)
                } else {
                    format!("{}f", value)
                }
            }

            UOp::Add { lhs, rhs, .. } => {
                let lhs_code = self.generate(lhs);
                let rhs_code = self.generate(rhs);
                format!("({} + {})", lhs_code, rhs_code)
            }

            UOp::Mul { lhs, rhs, .. } => {
                let lhs_code = self.generate(lhs);
                let rhs_code = self.generate(rhs);
                format!("({} * {})", lhs_code, rhs_code)
            }

            UOp::Max { lhs, rhs, .. } => {
                let lhs_code = self.generate(lhs);
                let rhs_code = self.generate(rhs);
                format!("fmax({}, {})", lhs_code, rhs_code)
            }

            UOp::Recip { arg, .. } => {
                let arg_code = self.generate(arg);
                format!("(1.0f / {})", arg_code)
            }

            UOp::Sqrt { arg, .. } => {
                let arg_code = self.generate(arg);
                format!("sqrt({})", arg_code)
            }

            UOp::Sequence { ops, .. } => {
                let mut code = String::new();
                for op in ops {
                    code.push_str(&self.generate(op));
                }
                code
            }

            UOp::Barrier { .. } => {
                format!("{}barrier(CLK_LOCAL_MEM_FENCE);\n", self.indent())
            }

            UOp::Rem { lhs, rhs, .. } => {
                let lhs_code = self.generate(lhs);
                let rhs_code = self.generate(rhs);
                format!("({} % {})", lhs_code, rhs_code)
            }

            UOp::Idiv { lhs, rhs, .. } => {
                let lhs_code = self.generate(lhs);
                let rhs_code = self.generate(rhs);
                format!("({} / {})", lhs_code, rhs_code)
            }

            UOp::LessThan { lhs, rhs, .. } => {
                let lhs_code = self.generate(lhs);
                let rhs_code = self.generate(rhs);
                format!("({} < {})", lhs_code, rhs_code)
            }

            UOp::Select {
                cond, then_, else_, ..
            } => {
                let cond_code = self.generate(cond);
                let then_code = self.generate(then_);
                let else_code = self.generate(else_);
                format!("({} ? {} : {})", cond_code, then_code, else_code)
            }

            _ => {
                // その他の演算は未実装
                format!("{}/* TODO: {:?} */\n", self.indent(), uop)
            }
        }
    }

    fn generate_loop(
        &mut self,
        var: &str,
        start: usize,
        end: usize,
        parallel: bool,
        body: &Rc<UOp>,
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

    fn generate_load(&mut self, buffer: &str, index: Option<&Rc<UOp>>) -> String {
        if let Some(idx) = index {
            let idx_code = self.generate(idx);
            format!("{}[(int){}]", buffer, idx_code)
        } else {
            buffer.to_string()
        }
    }

    fn generate_store(&mut self, buffer: &str, index: Option<&Rc<UOp>>, value: &Rc<UOp>) -> String {
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
    use crate::helper;
    use crate::lowering::Lowerer;
    use crate::uop::ElementwiseOp;
    use harp::DType;

    #[test]
    fn test_codegen_simple_add() {
        let mut codegen = OpenCLCodegen::new();

        let a = helper::input("a", vec![10], DType::F32);
        let b = helper::input("b", vec![10], DType::F32);
        let add = helper::elementwise(ElementwiseOp::Add, vec![a, b], DType::F32);

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
        let tid = helper::thread_idx(0, DType::F32);
        let a = helper::load("input", Some(tid.clone()), DType::F32);
        let b = helper::const_val(2.0, DType::F32);
        let result = helper::mul(a, b);
        let store = helper::store("output", Some(tid.clone()), result);
        let kernel = helper::loop_op("tid", 0, 100, store, true);

        let code = codegen.generate_kernel(&kernel, "mul_kernel");

        println!("Generated OpenCL kernel (manual):\n{}", code);

        assert!(code.contains("__kernel"));
        assert!(code.contains("mul_kernel"));
        assert!(code.contains("* 2.0f"));
    }
}
