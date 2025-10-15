use crate::{
    ast::{AstNode, ConstLiteral, DType},
    backend::{c_like::CLikeRenderer, c_like::MemoryConfig, Renderer},
};
use log::debug;
use std::fmt::Write;

#[derive(Debug)]
pub struct CRenderer {
    indent_level: usize,
    memory_config: MemoryConfig,
}

impl Default for CRenderer {
    fn default() -> Self {
        Self {
            indent_level: 0,
            memory_config: MemoryConfig::default(),
        }
    }
}

impl Renderer for CRenderer {
    type CodeRepr = String;
    type Option = ();

    fn with_option(&mut self, _option: Self::Option) {}

    fn new() -> Self {
        CRenderer::default()
    }

    fn render(&mut self, program: AstNode) -> Self::CodeRepr {
        let code = if let AstNode::Program { .. } = &program {
            self.render_program(&program)
        } else {
            panic!("Expected Program node, got {:?}", program);
        };
        debug!("\n--- Rendered C code ---\n{code}\n-----------------------");
        code
    }
}

impl CLikeRenderer for CRenderer {
    fn indent_level(&self) -> usize {
        self.indent_level
    }

    fn set_indent_level(&mut self, level: usize) {
        self.indent_level = level;
    }

    fn memory_config(&self) -> &MemoryConfig {
        &self.memory_config
    }

    fn render_includes(&self) -> String {
        let mut buffer = String::new();
        buffer.push_str("#include <math.h>\n");
        buffer.push_str("#include <stddef.h>\n");
        buffer.push_str("#include <stdint.h>\n");
        buffer.push_str("#include <stdlib.h>\n");
        buffer.push_str("#include <sys/types.h>\n"); // for ssize_t
        buffer.push('\n');
        buffer
    }

    fn render_scalar_dtype(&self, dtype: &DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::Isize => "ssize_t".to_string(),
            DType::Usize => "size_t".to_string(),
            DType::Bool => "int".to_string(),
            DType::Void => "void".to_string(),
            DType::Ptr(inner) => {
                if let DType::Void = **inner {
                    "void*".to_string()
                } else {
                    format!("{}*", self.render_scalar_dtype(inner))
                }
            }
            _ => unimplemented!("Unsupported scalar dtype: {:?}", dtype),
        }
    }

    fn render_const(&self, c: &ConstLiteral) -> String {
        use crate::ast::ConstLiteral::*;
        match c {
            F32(v) => {
                if v.is_infinite() {
                    if v.is_sign_negative() {
                        "(-INFINITY)".to_string()
                    } else {
                        "INFINITY".to_string()
                    }
                } else if v.is_nan() {
                    "NAN".to_string()
                } else {
                    format!("{}", v)
                }
            }
            Isize(v) => format!("{}", v),
            Usize(v) => format!("{}", v),
            Bool(v) => format!("{}", if *v { 1 } else { 0 }),
        }
    }
}

impl CRenderer {
    fn render_program(&mut self, program: &AstNode) -> String {
        let mut buffer = String::new();

        let functions = if let AstNode::Program { functions, .. } = program {
            functions
        } else {
            panic!("Expected Program node, got {:?}", program);
        };

        // Render includes
        buffer.push_str(&self.render_includes());

        // Add function prototypes
        for function_node in functions.iter() {
            if let AstNode::Function {
                name,
                arguments,
                return_type,
                ..
            } = function_node
            {
                let (ret_type, _) = self.render_dtype_recursive(return_type);
                let args = arguments
                    .iter()
                    .map(|(arg_name, dtype)| {
                        let (base_type, array_dims) = self.render_dtype_recursive(dtype);
                        format!("{} {}{}", base_type, arg_name, array_dims)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(buffer, "{} {}({});", ret_type, name, args).unwrap();
            }
        }
        buffer.push('\n');

        // Render function definitions
        for function_node in functions.iter() {
            write!(buffer, "{}", self.render_node(function_node)).unwrap();
            buffer.push('\n');
        }
        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::function;
    use crate::ast::{AstNode, Scope, VariableDecl};
    use rstest::rstest;

    fn var(name: &str) -> AstNode {
        AstNode::Var(name.to_string())
    }

    #[rstest]
    // Ops
    #[case(var("a") + var("b"), "a + b")]
    #[case(var("a") + (-var("b")), "a - b")]
    #[case(var("a") * var("b"), "a * b")]
    #[case(var("a") * var("b").recip(), "a / b")]
    #[case(AstNode::Rem(Box::new(var("a")), Box::new(var("b"))), "a % b")]
    #[case(AstNode::BitAnd(Box::new(var("a")), Box::new(var("b"))), "a & b")]
    #[case(AstNode::BitOr(Box::new(var("a")), Box::new(var("b"))), "a | b")]
    #[case(AstNode::BitXor(Box::new(var("a")), Box::new(var("b"))), "a ^ b")]
    #[case(AstNode::Shl(Box::new(var("a")), Box::new(var("b"))), "a << b")]
    #[case(AstNode::Shr(Box::new(var("a")), Box::new(var("b"))), "a >> b")]
    #[case(AstNode::BitNot(Box::new(var("a"))), "~a")]
    #[case(AstNode::Max(Box::new(var("a")), Box::new(var("b"))), "fmax(a, b)")]
    #[case(-var("a"), "-a")]
    #[case(AstNode::Sin(Box::new(var("a"))), "sin(a)")]
    #[case(AstNode::Sqrt(Box::new(var("a"))), "sqrt(a)")]
    // Accessors
    #[case(AstNode::Load { target: Box::new(var("a")), index: Box::new(var("i")), vector_width: 1 }, "*(a + i)")]
    #[case(AstNode::CallFunction { name: "my_func".to_string(), args: vec![var("a"), 2_isize.into()] }, "my_func(a, 2)")]
    // Comparisons
    #[case(AstNode::LessThan(Box::new(var("a")), Box::new(var("b"))), "a < b")]
    #[case(AstNode::Eq(Box::new(var("a")), Box::new(var("b"))), "a == b")]
    // Others
    #[case(AstNode::Assign("a".to_string(), Box::new(var("b"))), "a = b")]
    #[case(AstNode::Store { target: Box::new(var("arr")), index: Box::new(var("i")), value: Box::new(var("x")), vector_width: 1 }, "*(arr + i) = x")]
    #[case(AstNode::Cast { dtype: DType::F32, expr: Box::new(var("a")) }, "(float)a")]
    #[case(AstNode::Cast { dtype: DType::Ptr(Box::new(DType::F32)), expr: Box::new(var("a")) }, "(float*)a")]
    #[case(-(var("a") + var("b")) * var("c"), "-(a + b) * c")]
    fn test_render_node(#[case] input: AstNode, #[case] expected: &str) {
        let mut renderer = CRenderer::new();
        assert_eq!(renderer.render_node(&input), expected);
    }

    #[test]
    fn test_render_function() {
        let _ = env_logger::try_init();
        let func = function(
            "my_func".to_string(),
            vec![("a".to_string(), DType::Vec(Box::new(DType::Isize), 10))],
            DType::Void,
            Scope {
                declarations: vec![VariableDecl {
                    name: "b".to_string(),
                    dtype: DType::F32,
                    constant: false,
                    size_expr: None,
                }],
            },
            vec![AstNode::Assign(
                "b".to_string(),
                Box::new(AstNode::from(1.0f32)),
            )],
        );
        let program = AstNode::program(vec![func], "my_func");
        let expected = r###"#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>

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
        let func = function(
            "my_func".to_string(),
            vec![("a".to_string(), DType::Isize)],
            DType::Void,
            Scope {
                declarations: vec![],
            },
            vec![AstNode::Var("a".to_string())], // Single statement
        );
        let expected = r###"void my_func(ssize_t a)
{
	a;
}"###;
        let mut renderer = CRenderer::new();
        let buf = renderer.render_node(&func);
        assert_eq!(buf, expected);
    }

    #[test]
    fn test_render_dynamic_array() {
        let _ = env_logger::try_init();
        let func = function(
            "dynamic_alloc".to_string(),
            vec![("n".to_string(), DType::Usize)],
            DType::Void,
            Scope {
                declarations: vec![VariableDecl {
                    name: "arr".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32)),
                    constant: false,
                    size_expr: Some(Box::new(AstNode::Var("n".to_string()))),
                }],
            },
            vec![AstNode::Store {
                target: Box::new(var("arr")),
                index: Box::new(AstNode::from(0usize)),
                value: Box::new(AstNode::from(1.0f32)),
                vector_width: 1,
            }],
        );
        let program = AstNode::program(vec![func], "dynamic_alloc");
        let expected = r###"#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>

void dynamic_alloc(size_t n);

void dynamic_alloc(size_t n)
{
	float* arr;
	arr = (float*)malloc(sizeof(float) * (n));
	*(arr + 0) = 1;
	free(arr);
}
"###;
        let mut renderer = CRenderer::new();
        assert_eq!(renderer.render(program), expected);
    }

    #[test]
    fn test_render_if_statement() {
        use crate::ast::helper::*;

        // Test simple if without else
        let if_stmt = if_then(
            AstNode::LessThan(Box::new(var("i")), Box::new(var("n"))),
            AstNode::Assign("x".to_string(), Box::new(AstNode::from(1_isize))),
        );
        let expected = r###"if (i < n)
	x = 1;"###;
        let mut renderer = CRenderer::new();
        assert_eq!(renderer.render_node(&if_stmt), expected);
    }

    #[test]
    fn test_render_if_else_statement() {
        use crate::ast::helper::*;

        // Test if-else
        let if_stmt = if_then_else(
            AstNode::LessThan(Box::new(var("i")), Box::new(var("n"))),
            AstNode::Assign("x".to_string(), Box::new(AstNode::from(1_isize))),
            AstNode::Assign("x".to_string(), Box::new(AstNode::from(0_isize))),
        );
        let expected = r###"if (i < n)
	x = 1;
else
	x = 0;"###;
        let mut renderer = CRenderer::new();
        assert_eq!(renderer.render_node(&if_stmt), expected);
    }

    #[test]
    fn test_render_if_with_block() {
        use crate::ast::helper::*;

        // Test if with block
        let if_stmt = if_then(
            AstNode::LessThan(Box::new(var("i")), Box::new(var("n"))),
            block_with_statements(vec![
                AstNode::Assign("x".to_string(), Box::new(AstNode::from(1_isize))),
                AstNode::Assign("y".to_string(), Box::new(AstNode::from(2_isize))),
            ]),
        );
        let expected = r###"if (i < n) {
	x = 1;
	y = 2;
}"###;
        let mut renderer = CRenderer::new();
        assert_eq!(renderer.render_node(&if_stmt), expected);
    }

    #[test]
    fn test_render_if_else_if_chain() {
        use crate::ast::helper::*;

        // Test if-else if-else chain
        let if_stmt = if_then_else(
            AstNode::Eq(Box::new(var("x")), Box::new(AstNode::from(1_isize))),
            AstNode::Assign("result".to_string(), Box::new(AstNode::from(10_isize))),
            if_then_else(
                AstNode::Eq(Box::new(var("x")), Box::new(AstNode::from(2_isize))),
                AstNode::Assign("result".to_string(), Box::new(AstNode::from(20_isize))),
                AstNode::Assign("result".to_string(), Box::new(AstNode::from(0_isize))),
            ),
        );
        let expected = r###"if (x == 1)
	result = 10;
else if (x == 2)
	result = 20;
else
	result = 0;"###;
        let mut renderer = CRenderer::new();
        assert_eq!(renderer.render_node(&if_stmt), expected);
    }
}
