use crate::ast::{
    AstNode, DType, Function, FunctionKind, Literal, Mutability, Program, VarDecl, VarKind,
};
use crate::backend::Renderer;
use crate::backend::metal::MetalCode;
use log::{debug, info, trace};

/// Metal Shading Language用のレンダラー
pub struct MetalRenderer {
    indent_level: usize,
    indent_size: usize,
}

impl MetalRenderer {
    pub fn new() -> Self {
        Self {
            indent_level: 0,
            indent_size: 4,
        }
    }

    fn indent(&self) -> String {
        " ".repeat(self.indent_level * self.indent_size)
    }

    fn inc_indent(&mut self) {
        self.indent_level += 1;
    }

    fn dec_indent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    /// DTypeをMetal型文字列に変換
    #[allow(clippy::only_used_in_recursion)]
    fn render_dtype(&self, dtype: &DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::Isize => "int".to_string(),
            DType::Usize => "uint".to_string(),
            DType::Ptr(inner) => format!("device {}*", self.render_dtype(inner)),
            DType::Vec(inner, size) => {
                let base = self.render_dtype(inner);
                format!("{}{}", base, size)
            }
            DType::Tuple(types) => {
                if types.is_empty() {
                    "void".to_string()
                } else {
                    // Metalはタプル型を直接サポートしないので構造体として表現
                    format!("tuple_{}", types.len())
                }
            }
            DType::Unknown => "auto".to_string(),
        }
    }

    /// AstNodeを式として描画
    fn render_expr(&self, node: &AstNode) -> String {
        match node {
            AstNode::Wildcard(name) => name.clone(),
            AstNode::Const(lit) => self.render_literal(lit),
            AstNode::Var(name) => name.clone(),
            AstNode::Add(left, right) => {
                format!("({} + {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::Mul(left, right) => {
                format!("({} * {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::Max(left, right) => {
                format!(
                    "max({}, {})",
                    self.render_expr(left),
                    self.render_expr(right)
                )
            }
            AstNode::Rem(left, right) => {
                format!("({} % {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::Idiv(left, right) => {
                format!("({} / {})", self.render_expr(left), self.render_expr(right))
            }
            AstNode::Recip(operand) => {
                format!("(1.0f / {})", self.render_expr(operand))
            }
            AstNode::Sqrt(operand) => {
                format!("sqrt({})", self.render_expr(operand))
            }
            AstNode::Log2(operand) => {
                format!("log2({})", self.render_expr(operand))
            }
            AstNode::Exp2(operand) => {
                format!("exp2({})", self.render_expr(operand))
            }
            AstNode::Sin(operand) => {
                format!("sin({})", self.render_expr(operand))
            }
            AstNode::Cast(operand, dtype) => {
                format!(
                    "{}({})",
                    self.render_dtype(dtype),
                    self.render_expr(operand)
                )
            }
            AstNode::Load { ptr, offset, count } => {
                if *count == 1 {
                    format!("{}[{}]", self.render_expr(ptr), self.render_expr(offset))
                } else {
                    // ベクトルロード
                    format!(
                        "*reinterpret_cast<device {}*>(&{}[{}])",
                        self.render_dtype(&node.infer_type()),
                        self.render_expr(ptr),
                        self.render_expr(offset)
                    )
                }
            }
            AstNode::Call { name, args } => {
                let arg_strs: Vec<String> = args.iter().map(|a| self.render_expr(a)).collect();
                format!("{}({})", name, arg_strs.join(", "))
            }
            AstNode::Return { value } => {
                format!("return {}", self.render_expr(value))
            }
            _ => format!("/* unsupported expression: {:?} */", node),
        }
    }

    /// リテラルを描画
    fn render_literal(&self, lit: &Literal) -> String {
        match lit {
            Literal::F32(v) => format!("{}f", v),
            Literal::Isize(v) => format!("{}", v),
            Literal::Usize(v) => format!("{}u", v),
        }
    }

    /// 文として描画
    fn render_statement(&mut self, node: &AstNode) -> String {
        match node {
            AstNode::Store { ptr, offset, value } => {
                format!(
                    "{}{}[{}] = {};",
                    self.indent(),
                    self.render_expr(ptr),
                    self.render_expr(offset),
                    self.render_expr(value)
                )
            }
            AstNode::Assign { var, value } => {
                format!(
                    "{}auto {} = {};",
                    self.indent(),
                    var,
                    self.render_expr(value)
                )
            }
            AstNode::Block { statements, .. } => self.render_block(statements),
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => self.render_range(var, start, step, stop, body),
            AstNode::Return { value } => {
                format!("{}return {};", self.indent(), self.render_expr(value))
            }
            AstNode::Barrier => {
                format!(
                    "{}threadgroup_barrier(mem_flags::mem_threadgroup);",
                    self.indent()
                )
            }
            AstNode::Call { name, args } => {
                let arg_strs: Vec<String> = args.iter().map(|a| self.render_expr(a)).collect();
                format!("{}{}({});", self.indent(), name, arg_strs.join(", "))
            }
            _ => {
                // 式として評価できるものは文末にセミコロンを付ける
                format!("{}{};", self.indent(), self.render_expr(node))
            }
        }
    }

    /// ブロックを描画
    fn render_block(&mut self, statements: &[AstNode]) -> String {
        let mut result = String::new();
        for stmt in statements {
            result.push_str(&self.render_statement(stmt));
            result.push('\n');
        }
        result
    }

    /// Rangeループを描画
    fn render_range(
        &mut self,
        var: &str,
        start: &AstNode,
        step: &AstNode,
        stop: &AstNode,
        body: &AstNode,
    ) -> String {
        let mut result = String::new();
        result.push_str(&format!(
            "{}for (uint {} = {}; {} < {}; {} += {}) {{",
            self.indent(),
            var,
            self.render_expr(start),
            var,
            self.render_expr(stop),
            var,
            self.render_expr(step)
        ));
        result.push('\n');
        self.inc_indent();
        result.push_str(&self.render_statement(body));
        self.dec_indent();
        result.push_str(&format!("{}}}", self.indent()));
        result
    }

    /// 関数パラメータを描画
    fn render_param(&self, param: &VarDecl, is_kernel: bool) -> String {
        let type_str = self.render_dtype(&param.dtype);
        let mut_str = match param.mutability {
            Mutability::Immutable => "const ",
            Mutability::Mutable => "",
        };

        // カーネル関数の場合、buffer_indexなどの属性を付ける
        if is_kernel {
            match &param.kind {
                VarKind::Normal => {
                    // 通常の引数
                    format!("{}{} {}", mut_str, type_str, param.name)
                }
                VarKind::ThreadId(_axis) => {
                    // スレッドIDは関数パラメータではなく組み込み変数として扱う
                    format!("uint {} [[thread_position_in_grid]]", param.name)
                }
                VarKind::GroupId(_axis) => {
                    format!("uint {} [[threadgroup_position_in_grid]]", param.name)
                }
                VarKind::GroupSize(_axis) => {
                    format!("uint {} [[threads_per_threadgroup]]", param.name)
                }
                VarKind::GridSize(_axis) => {
                    format!("uint {} [[threads_per_grid]]", param.name)
                }
            }
        } else {
            format!("{}{} {}", mut_str, type_str, param.name)
        }
    }

    /// 関数を描画
    pub fn render_function(&mut self, name: &str, func: &Function) -> String {
        let is_kernel = matches!(func.kind, FunctionKind::Kernel(_));
        debug!(
            "Rendering Metal {} function: {}",
            if is_kernel { "kernel" } else { "device" },
            name
        );
        trace!("Function params: {} parameters", func.params.len());

        let mut result = String::new();

        let func_qualifier = if is_kernel { "kernel" } else { "" };
        let return_type = self.render_dtype(&func.return_type);

        // 関数シグネチャ
        result.push_str(&format!("{} {} {}(", func_qualifier, return_type, name));

        // パラメータ
        let params: Vec<String> = func
            .params
            .iter()
            .map(|p| self.render_param(p, is_kernel))
            .collect();
        result.push_str(&params.join(", "));
        result.push_str(") {\n");

        // 関数本体
        self.inc_indent();
        result.push_str(&self.render_statement(&func.body));
        self.dec_indent();
        result.push_str("}\n");

        trace!("Function rendering completed");
        result
    }

    /// プログラム全体を描画
    pub fn render_program(&mut self, program: &Program) -> MetalCode {
        self.render_program_with_signature(program, crate::backend::KernelSignature::empty())
    }

    /// シグネチャ付きでプログラムをレンダリング
    pub fn render_program_with_signature(
        &mut self,
        program: &Program,
        signature: crate::backend::KernelSignature,
    ) -> MetalCode {
        info!(
            "Rendering Metal program: {} with {} functions",
            program.entry_point,
            program.functions.len()
        );

        let mut result = String::new();

        // ヘッダー
        result.push_str("#include <metal_stdlib>\n");
        result.push_str("using namespace metal;\n\n");

        // 全関数を描画
        for (name, func) in &program.functions {
            result.push_str(&self.render_function(name, func));
            result.push('\n');
        }

        info!("Metal program rendering completed ({} bytes)", result.len());
        trace!("Generated Metal code:\n{}", result);

        MetalCode::with_signature(result, signature)
    }
}

impl Default for MetalRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for MetalRenderer {
    type CodeRepr = MetalCode;
    type Option = ();

    fn render(&self, ast: AstNode) -> Self::CodeRepr {
        let mut renderer = Self::new();
        MetalCode::new(renderer.render_statement(&ast))
    }

    fn is_available(&self) -> bool {
        // Metalは常に利用可能として扱う（実際のコンパイルは別の問題）
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;
    use crate::ast::{AccessRegion, Scope};

    #[test]
    fn test_render_literal() {
        let renderer = MetalRenderer::new();
        assert_eq!(renderer.render_literal(&Literal::F32(3.14)), "3.14f");
        assert_eq!(renderer.render_literal(&Literal::Isize(42)), "42");
        assert_eq!(renderer.render_literal(&Literal::Usize(100)), "100u");
    }

    #[test]
    fn test_render_dtype() {
        let renderer = MetalRenderer::new();
        assert_eq!(renderer.render_dtype(&DType::F32), "float");
        assert_eq!(renderer.render_dtype(&DType::Isize), "int");
        assert_eq!(renderer.render_dtype(&DType::Usize), "uint");
        assert_eq!(renderer.render_dtype(&DType::F32.to_ptr()), "device float*");
        assert_eq!(renderer.render_dtype(&DType::F32.to_vec(4)), "float4");
    }

    #[test]
    fn test_render_binary_ops() {
        let renderer = MetalRenderer::new();
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());

        let add = a.clone() + b.clone();
        assert_eq!(renderer.render_expr(&add), "(1f + 2f)");

        let mul = a.clone() * b.clone();
        assert_eq!(renderer.render_expr(&mul), "(1f * 2f)");
    }

    #[test]
    fn test_render_math_funcs() {
        let renderer = MetalRenderer::new();
        let x = AstNode::Const(4.0f32.into());

        assert_eq!(renderer.render_expr(&sqrt(x.clone())), "sqrt(4f)");
        assert_eq!(renderer.render_expr(&sin(x.clone())), "sin(4f)");
        assert_eq!(renderer.render_expr(&log2(x.clone())), "log2(4f)");
    }

    #[test]
    fn test_render_barrier() {
        let mut renderer = MetalRenderer::new();
        let barrier_stmt = renderer.render_statement(&barrier());
        assert!(barrier_stmt.contains("threadgroup_barrier"));
    }

    #[test]
    fn test_render_simple_kernel() {
        // 簡単なカーネル: output[tid] = input[tid] * 2.0
        let params = vec![
            VarDecl {
                name: "tid".to_string(),
                dtype: DType::Usize,
                mutability: Mutability::Immutable,
                region: AccessRegion::ThreadLocal,
                kind: VarKind::ThreadId(0),
            },
            VarDecl {
                name: "input".to_string(),
                dtype: DType::F32.to_ptr(),
                mutability: Mutability::Immutable,
                region: AccessRegion::Shared,
                kind: VarKind::Normal,
            },
            VarDecl {
                name: "output".to_string(),
                dtype: DType::F32.to_ptr(),
                mutability: Mutability::Mutable,
                region: AccessRegion::ShardedBy(vec![0]),
                kind: VarKind::Normal,
            },
        ];

        let body_statements = vec![store(
            var("output"),
            var("tid"),
            load(var("input"), var("tid")) * AstNode::Const(2.0f32.into()),
        )];

        let func = Function::new(
            FunctionKind::Kernel(1),
            params,
            DType::Tuple(vec![]),
            body_statements,
        )
        .unwrap();

        let mut renderer = MetalRenderer::new();
        let code = renderer.render_function("scale_kernel", &func);

        // 基本的な構造をチェック
        assert!(code.contains("kernel"));
        assert!(code.contains("void scale_kernel"));
        assert!(code.contains("thread_position_in_grid"));
        assert!(code.contains("device float*"));
        assert!(code.contains("output[tid] = (input[tid] * 2f)"));
    }

    #[test]
    fn test_render_program() {
        let mut program = Program::new("main".to_string());

        // 簡単な関数: double(x) = x * 2
        let double_params = vec![VarDecl {
            name: "x".to_string(),
            dtype: DType::F32,
            mutability: Mutability::Immutable,
            region: AccessRegion::ThreadLocal,
            kind: VarKind::Normal,
        }];

        let double_func = Function::new(
            FunctionKind::Normal,
            double_params,
            DType::F32,
            vec![AstNode::Return {
                value: Box::new(var("x") * AstNode::Const(2.0f32.into())),
            }],
        )
        .unwrap();

        program
            .add_function("double".to_string(), double_func)
            .unwrap();

        // メイン関数
        let main_func = Function::new(
            FunctionKind::Normal,
            vec![],
            DType::F32,
            vec![AstNode::Call {
                name: "double".to_string(),
                args: vec![AstNode::Const(5.0f32.into())],
            }],
        )
        .unwrap();

        program.add_function("main".to_string(), main_func).unwrap();

        let mut renderer = MetalRenderer::new();
        let code = renderer.render_program(&program);

        // ヘッダーとインクルードをチェック
        assert!(code.contains("#include <metal_stdlib>"));
        assert!(code.contains("using namespace metal;"));

        // 関数定義をチェック
        assert!(code.contains("float double("));
        assert!(code.contains("float main("));
        assert!(code.contains("return (x * 2f)"));
        assert!(code.contains("double(5f)"));
    }

    #[test]
    fn test_render_loop_with_barrier() {
        // ループとバリアを含むカーネル
        let mut loop_scope = Scope::new();
        loop_scope
            .declare(
                "i".to_string(),
                DType::Usize,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(0usize.into())),
            step: Box::new(AstNode::Const(1usize.into())),
            stop: Box::new(AstNode::Const(10usize.into())),
            body: Box::new(AstNode::Block {
                statements: vec![
                    store(var("shared"), var("i"), load(var("input"), var("i"))),
                    barrier(),
                    store(var("output"), var("i"), load(var("shared"), var("i"))),
                ],
                scope: Box::new(loop_scope),
            }),
        };

        let mut renderer = MetalRenderer::new();
        let code = renderer.render_statement(&loop_node);

        assert!(code.contains("for (uint i = 0u; i < 10u; i += 1u)"));
        assert!(code.contains("shared[i] = input[i]"));
        assert!(code.contains("threadgroup_barrier"));
        assert!(code.contains("output[i] = shared[i]"));
    }
}
