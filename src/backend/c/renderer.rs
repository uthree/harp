use crate::{
    ast::{AstNode, ConstLiteral, DType, Function, Program, Scope, VariableDecl},
    backend::Renderer,
};
use log::debug;
use std::fmt::Write;

#[derive(Debug, Default)]
pub struct CRenderer {
    indent_level: usize,
}

impl Renderer for CRenderer {
    type CodeRepr = String;
    type Option = ();

    fn with_option(&mut self, _option: Self::Option) {}

    fn new() -> Self {
        CRenderer::default()
    }

    fn render(&mut self, program: Program) -> Self::CodeRepr {
        let code = self.render_program(&program);
        debug!("\n--- Rendered C code ---\n{code}\n-----------------------");
        code
    }
}

impl CRenderer {
    fn render_program(&mut self, program: &Program) -> String {
        let mut buffer = String::new();
        buffer.push_str("#include <math.h>\n");
        buffer.push_str("#include <stddef.h>\n");
        buffer.push_str("#include <stdint.h>\n");
        buffer.push_str("#include <stdlib.h>\n");
        buffer.push('\n');

        // Add function prototypes
        for function in program.functions.iter() {
            writeln!(buffer, "{};", self.render_function_signature(function)).unwrap();
        }
        buffer.push('\n');

        for function in program.functions.iter() {
            write!(buffer, "{}", self.render_function(function)).unwrap();
        }
        buffer
    }

    fn render_function_signature(&mut self, function: &Function) -> String {
        let (return_type, _) = Self::render_dtype_recursive(&function.return_type);
        let args = function
            .arguments
            .iter()
            .map(|(name, dtype)| {
                let (base_type, array_dims) = Self::render_dtype_recursive(dtype);
                format!("{} {}{}", base_type, name, array_dims)
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("{} {}({})", return_type, function.name, args)
    }

    fn render_function(&mut self, function: &Function) -> String {
        let mut buffer = String::new();
        writeln!(buffer, "{}", self.render_function_signature(function)).unwrap();
        let body_str = match &function.body {
            AstNode::Block { scope, statements } => self.render_scope(scope, statements),
            single_statement => {
                let empty_scope = Scope {
                    declarations: vec![],
                };
                self.render_scope(&empty_scope, std::slice::from_ref(single_statement))
            }
        };
        write!(buffer, "{}", body_str).unwrap();
        writeln!(buffer).unwrap();
        buffer
    }

    fn render_scope(&mut self, scope: &Scope, statements: &[AstNode]) -> String {
        let mut buffer = String::new();
        writeln!(buffer, "{{").unwrap();
        self.indent_level += 1;

        // Render declarations
        for decl in &scope.declarations {
            self.render_indent(&mut buffer);
            let decl_str = self.render_variable_decl(decl);
            writeln!(buffer, "{};", decl_str).unwrap();

            // For dynamic arrays (pointers with size_expr), emit malloc call
            if let Some(size_expr) = &decl.size_expr {
                if let DType::Ptr(inner) = &decl.dtype {
                    self.render_indent(&mut buffer);
                    let base_type = Self::render_scalar_dtype(inner);
                    let size_code = self.render_node(size_expr);
                    writeln!(
                        buffer,
                        "{} = ({}*)malloc(sizeof({}) * ({}));",
                        decl.name, base_type, base_type, size_code
                    )
                    .unwrap();
                }
            }
        }

        // Render statements
        if !scope.declarations.is_empty() && !statements.is_empty() {
            writeln!(buffer).unwrap();
        }
        for stmt in statements {
            self.render_indent(&mut buffer);
            writeln!(buffer, "{};", self.render_node(stmt)).unwrap();
        }

        self.indent_level -= 1;
        self.render_indent(&mut buffer);
        write!(buffer, "}}").unwrap();
        buffer
    }

    fn render_variable_decl(&mut self, decl: &VariableDecl) -> String {
        let mut buffer = String::new();
        if decl.constant {
            write!(buffer, "const ").unwrap();
        }
        let (base_type, array_dims) = Self::render_dtype_recursive(&decl.dtype);
        write!(buffer, "{} {}{}", base_type, decl.name, array_dims).unwrap();
        buffer
    }

    fn render_node(&mut self, node: &AstNode) -> String {
        let mut buffer = String::new();
        match node {
            AstNode::Const(c) => write!(buffer, "{}", self.render_const(c)).unwrap(),
            AstNode::Var(s) => write!(buffer, "{}", s).unwrap(),
            AstNode::Add(lhs, rhs) => match &**rhs {
                AstNode::Neg(negv) => write!(
                    buffer,
                    "( {} - {} )",
                    self.render_node(lhs),
                    self.render_node(negv)
                )
                .unwrap(),
                _ => write!(
                    buffer,
                    "({} + {})",
                    self.render_node(lhs),
                    self.render_node(rhs)
                )
                .unwrap(),
            },
            AstNode::Mul(lhs, rhs) => match &**rhs {
                AstNode::Recip(recipv) => write!(
                    buffer,
                    "( {} / {} )",
                    self.render_node(lhs),
                    self.render_node(recipv)
                )
                .unwrap(),
                _ => write!(
                    buffer,
                    "({} * {})",
                    self.render_node(lhs),
                    self.render_node(rhs)
                )
                .unwrap(),
            },
            AstNode::Div(lhs, rhs) => write!(
                buffer,
                "({} / {})",
                self.render_node(lhs),
                self.render_node(rhs)
            )
            .unwrap(),
            AstNode::Rem(lhs, rhs) => write!(
                buffer,
                "({} % {})",
                self.render_node(lhs),
                self.render_node(rhs)
            )
            .unwrap(),
            AstNode::Max(lhs, rhs) => write!(
                buffer,
                "fmax({}, {})",
                self.render_node(lhs),
                self.render_node(rhs)
            )
            .unwrap(),
            AstNode::Assign(var_name, rhs) => {
                write!(buffer, "{} = {}", var_name, self.render_node(rhs)).unwrap();
            }
            AstNode::Store {
                target,
                index,
                value,
            } => {
                write!(
                    buffer,
                    "*(({}) + ({})) = {}",
                    self.render_node(target),
                    self.render_node(index),
                    self.render_node(value)
                )
                .unwrap();
            }
            AstNode::Deref(expr) => {
                write!(buffer, "*({})", self.render_node(expr)).unwrap();
            }
            AstNode::Neg(v) => write!(buffer, "-{}", self.render_node(v)).unwrap(),
            AstNode::Recip(v) => write!(buffer, "(1 / {})", self.render_node(v)).unwrap(),
            AstNode::Sin(v) => write!(buffer, "sin({})", self.render_node(v)).unwrap(),
            AstNode::Sqrt(v) => write!(buffer, "sqrt({})", self.render_node(v)).unwrap(),
            AstNode::Log2(v) => write!(buffer, "log2({})", self.render_node(v)).unwrap(),
            AstNode::Exp2(v) => write!(buffer, "exp2({})", self.render_node(v)).unwrap(),
            AstNode::Range {
                counter_name,
                max,
                body,
            } => {
                let max = self.render_node(max);
                writeln!(
                    buffer,
                    "for (size_t {counter_name} = 0; {counter_name} < {max}; {counter_name}++)"
                )
                .unwrap();
                self.indent_level += 1;
                self.render_indent(&mut buffer);
                write!(buffer, "{}", self.render_node(body)).unwrap();
                self.indent_level -= 1;
            }
            AstNode::Block { scope, statements } => {
                write!(buffer, "{}", self.render_scope(scope, statements)).unwrap();
            }
            AstNode::Cast { dtype, expr } => {
                let (base_type, dims) = Self::render_dtype_recursive(dtype);
                // Cでは配列型へのキャストはポインタ型へのキャストとして書く
                let type_str = if !dims.is_empty() {
                    format!("{}*", base_type)
                } else {
                    base_type
                };
                write!(buffer, "({}){}", type_str, self.render_node(expr)).unwrap();
            }
            AstNode::CallFunction { name, args } => {
                let args_str = args
                    .iter()
                    .map(|arg| self.render_node(arg))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(buffer, "{}({})", name, args_str).unwrap();
            }

            node => todo!("render_node for {:?}", node),
        }
        buffer
    }

    fn render_indent(&self, buffer: &mut String) {
        for _ in 0..self.indent_level {
            buffer.push('\t');
        }
    }

    fn render_const(&self, c: &ConstLiteral) -> String {
        use crate::ast::ConstLiteral::*;
        match c {
            F32(v) => format!("{}", v),
            Isize(v) => format!("{}", v),
            Usize(v) => format!("{}", v),
        }
    }

    fn render_scalar_dtype(dtype: &DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::Isize => "ssize_t".to_string(),
            DType::Usize => "size_t".to_string(),
            DType::Void => "void".to_string(),
            DType::Ptr(inner) => {
                if let DType::Void = **inner {
                    // void* の特殊ケース
                    "void*".to_string()
                } else {
                    // 基本的には再帰しないが、void**のようなケースのため
                    format!("{}*", Self::render_scalar_dtype(inner))
                }
            }
            _ => unimplemented!("Unsupported scalar dtype: {:?}", dtype),
        }
    }

    // Returns (base_type, array_dims)
    fn render_dtype_recursive(dtype: &DType) -> (String, String) {
        match dtype {
            DType::Ptr(inner) => {
                let (base, dims) = Self::render_dtype_recursive(inner);
                // void** のような多重ポインタを扱う
                if base.ends_with('*') || dims.is_empty() {
                    (format!("{}*", base), dims)
                } else {
                    (base, format!("(*{}){}", "", dims))
                }
            }
            DType::Vec(inner, size) => {
                let (base, dims) = Self::render_dtype_recursive(inner);
                (base, format!("{}{}[{}]", dims, "", size))
            }
            _ => (Self::render_scalar_dtype(dtype), "".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AstNode, Scope, VariableDecl};
    use rstest::rstest;

    fn var(name: &str) -> AstNode {
        AstNode::Var(name.to_string())
    }

    #[rstest]
    // Ops
    #[case(var("a") + var("b"), "(a + b)")]
    #[case(var("a") + (-var("b")), "( a - b )")]
    #[case(var("a") * var("b"), "(a * b)")]
    #[case(var("a") * var("b").recip(), "( a / b )")]
    #[case(AstNode::Rem(Box::new(var("a")), Box::new(var("b"))), "(a % b)")]
    #[case(AstNode::Max(Box::new(var("a")), Box::new(var("b"))), "fmax(a, b)")]
    #[case(-var("a"), "-a")]
    #[case(AstNode::Sin(Box::new(var("a"))), "sin(a)")]
    #[case(AstNode::Sqrt(Box::new(var("a"))), "sqrt(a)")]
    // Accessors
    #[case(AstNode::Deref(Box::new(var("a") + var("i"))), "*((a + i))")]
    #[case(AstNode::CallFunction { name: "my_func".to_string(), args: vec![var("a"), (2 as isize).into()] }, "my_func(a, 2)")]
    // Others
    #[case(AstNode::Assign("a".to_string(), Box::new(var("b"))), "a = b")]
    #[case(AstNode::Store { target: Box::new(var("arr")), index: Box::new(var("i")), value: Box::new(var("x")) }, "*((arr) + (i)) = x")]
    #[case(AstNode::Cast { dtype: DType::F32, expr: Box::new(var("a")) }, "(float)a")]
    #[case(AstNode::Cast { dtype: DType::Ptr(Box::new(DType::F32)), expr: Box::new(var("a")) }, "(float*)a")]
    #[case(-(var("a") + var("b")) * var("c"), "(-(a + b) * c)")]
    fn test_render_node(#[case] input: AstNode, #[case] expected: &str) {
        let mut renderer = CRenderer::new();
        assert_eq!(renderer.render_node(&input), expected);
    }

    #[test]
    fn test_render_function() {
        let _ = env_logger::try_init();
        let function = Function::new(
            "my_func".to_string(),
            vec![("a".to_string(), DType::Vec(Box::new(DType::Isize), 10))],
            DType::Void,
            AstNode::Block {
                scope: Scope {
                    declarations: vec![VariableDecl {
                        name: "b".to_string(),
                        dtype: DType::F32,
                        constant: false,
                        size_expr: None,
                    }],
                },
                statements: vec![AstNode::Assign(
                    "b".to_string(),
                    Box::new(AstNode::from(1.0f32)),
                )],
            },
        );
        let program = Program {
            functions: vec![function],
            entry_point: "my_func".to_string(),
        };
        let expected = r###"#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

void my_func(ssize_t a[10]);

void my_func(ssize_t a[10])
{
	float b;

	b = 1;
}
"###;
        let mut renderer = CRenderer::new();
        assert_eq!(renderer.render(program), expected);
    }

    #[test]
    fn test_render_function_single_statement() {
        let function = Function {
            name: "my_func".to_string(),
            arguments: vec![("a".to_string(), DType::Isize)],
            return_type: DType::Void,
            body: AstNode::Var("a".to_string()), // Directly pass a single node
        };
        let expected = r###"void my_func(ssize_t a)
{
	a;
}
"###;
        let mut renderer = CRenderer::new();
        assert_eq!(renderer.render_function(&function), expected);
    }

    #[test]
    fn test_render_dynamic_array() {
        let _ = env_logger::try_init();
        let function = Function::new(
            "dynamic_alloc".to_string(),
            vec![("n".to_string(), DType::Usize)],
            DType::Void,
            AstNode::Block {
                scope: Scope {
                    declarations: vec![VariableDecl {
                        name: "arr".to_string(),
                        dtype: DType::Ptr(Box::new(DType::F32)),
                        constant: false,
                        size_expr: Some(Box::new(AstNode::Var("n".to_string()))),
                    }],
                },
                statements: vec![AstNode::Store {
                    target: Box::new(var("arr")),
                    index: Box::new(AstNode::from(0usize)),
                    value: Box::new(AstNode::from(1.0f32)),
                }],
            },
        );
        let program = Program {
            functions: vec![function],
            entry_point: "dynamic_alloc".to_string(),
        };
        let expected = r###"#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

void dynamic_alloc(size_t n);

void dynamic_alloc(size_t n)
{
	float* arr;
	arr = (float*)malloc(sizeof(float) * (n));

	*((arr) + (0)) = 1;
}
"###;
        let mut renderer = CRenderer::new();
        assert_eq!(renderer.render(program), expected);
    }
}
