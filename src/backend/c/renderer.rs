use crate::{
    ast::{AstNode, ConstLiteral, DType, Scope, VariableDecl},
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

impl CRenderer {
    /// Get the precedence level of an operator.
    /// Higher number = higher precedence (binds tighter).
    fn precedence(node: &AstNode) -> u8 {
        match node {
            // Highest precedence: atoms and function calls
            AstNode::Const(_) | AstNode::Var(_) | AstNode::CallFunction { .. } => 100,

            // Unary operators
            AstNode::Neg(_)
            | AstNode::Recip(_)
            | AstNode::Sin(_)
            | AstNode::Sqrt(_)
            | AstNode::Log2(_)
            | AstNode::Exp2(_)
            | AstNode::Deref(_)
            | AstNode::Cast { .. } => 90,

            // Multiplicative operators
            AstNode::Mul(_, _) | AstNode::Rem(_, _) => 80,

            // Additive operators
            AstNode::Add(_, _) => 70,

            // Shift operators
            AstNode::Shl(_, _) | AstNode::Shr(_, _) => 60,

            // Comparison operators
            AstNode::LessThan(_, _) | AstNode::Eq(_, _) => 55,

            // Bitwise AND
            AstNode::BitAnd(_, _) => 50,

            // Bitwise XOR
            AstNode::BitXor(_, _) => 40,

            // Bitwise OR
            AstNode::BitOr(_, _) => 30,

            // Ternary/Select operator
            AstNode::Select { .. } => 20,

            // Everything else (statements, etc.)
            _ => 0,
        }
    }

    /// Render a node with parentheses if needed based on precedence.
    fn render_with_parens(
        &mut self,
        node: &AstNode,
        parent_precedence: u8,
        is_rhs: bool,
    ) -> String {
        let node_precedence = Self::precedence(node);
        let needs_parens = if is_rhs {
            // Right side needs parens if precedence is lower or equal (for left-associative ops)
            node_precedence <= parent_precedence && parent_precedence > 0
        } else {
            // Left side needs parens only if precedence is strictly lower
            node_precedence < parent_precedence && parent_precedence > 0
        };

        let rendered = self.render_node(node);
        if needs_parens && !matches!(node, AstNode::Const(_) | AstNode::Var(_)) {
            format!("({})", rendered)
        } else {
            rendered
        }
    }

    fn render_program(&mut self, program: &AstNode) -> String {
        let mut buffer = String::new();

        let functions = if let AstNode::Program { functions, .. } = program {
            functions
        } else {
            panic!("Expected Program node, got {:?}", program);
        };

        buffer.push_str("#include <math.h>\n");
        buffer.push_str("#include <stddef.h>\n");
        buffer.push_str("#include <stdint.h>\n");
        buffer.push_str("#include <stdlib.h>\n");
        buffer.push_str("#include <sys/types.h>\n"); // for ssize_t
        buffer.push('\n');

        // Add function prototypes
        for function_node in functions.iter() {
            if let AstNode::Function {
                name,
                arguments,
                return_type,
                ..
            } = function_node
            {
                let (ret_type, _) = Self::render_dtype_recursive(return_type);
                let args = arguments
                    .iter()
                    .map(|(arg_name, dtype)| {
                        let (base_type, array_dims) = Self::render_dtype_recursive(dtype);
                        format!("{} {}{}", base_type, arg_name, array_dims)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(buffer, "{} {}({});", ret_type, name, args).unwrap();
            }
        }
        buffer.push('\n');

        for function_node in functions.iter() {
            write!(buffer, "{}", self.render_node(function_node)).unwrap();
            buffer.push('\n');
        }
        buffer
    }

    fn render_scope(&mut self, scope: &Scope, statements: &[AstNode]) -> String {
        let mut buffer = String::new();
        //self.render_indent(&mut buffer);
        self.indent_level += 1;
        writeln!(buffer, "{{").unwrap();

        // Track variables that need to be freed
        let mut malloc_vars = Vec::new();

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
                    // Track this variable for cleanup
                    malloc_vars.push(decl.name.clone());
                }
            }
        }

        // Render statements
        for stmt in statements {
            self.render_indent(&mut buffer);
            writeln!(buffer, "{};", self.render_node(stmt)).unwrap();
        }

        // Free dynamically allocated memory before closing scope
        if !malloc_vars.is_empty() {
            for var_name in malloc_vars {
                self.render_indent(&mut buffer);
                writeln!(buffer, "free({});", var_name).unwrap();
            }
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
            AstNode::Add(lhs, rhs) => {
                let prec = Self::precedence(node);
                match &**rhs {
                    AstNode::Neg(negv) => {
                        // a + (-b) => a - b
                        write!(
                            buffer,
                            "{} - {}",
                            self.render_with_parens(lhs, prec, false),
                            self.render_with_parens(negv, prec, true)
                        )
                        .unwrap()
                    }
                    _ => write!(
                        buffer,
                        "{} + {}",
                        self.render_with_parens(lhs, prec, false),
                        self.render_with_parens(rhs, prec, true)
                    )
                    .unwrap(),
                }
            }
            AstNode::Mul(lhs, rhs) => {
                let prec = Self::precedence(node);
                match &**rhs {
                    AstNode::Recip(recipv) => {
                        // a * recip(b) => a / b
                        write!(
                            buffer,
                            "{} / {}",
                            self.render_with_parens(lhs, prec, false),
                            self.render_with_parens(recipv, prec, true)
                        )
                        .unwrap()
                    }
                    _ => write!(
                        buffer,
                        "{} * {}",
                        self.render_with_parens(lhs, prec, false),
                        self.render_with_parens(rhs, prec, true)
                    )
                    .unwrap(),
                }
            }
            AstNode::Rem(lhs, rhs) => {
                let prec = Self::precedence(node);
                write!(
                    buffer,
                    "{} % {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap()
            }
            AstNode::BitAnd(lhs, rhs) => {
                let prec = Self::precedence(node);
                write!(
                    buffer,
                    "{} & {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap()
            }
            AstNode::BitOr(lhs, rhs) => {
                let prec = Self::precedence(node);
                write!(
                    buffer,
                    "{} | {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap()
            }
            AstNode::BitXor(lhs, rhs) => {
                let prec = Self::precedence(node);
                write!(
                    buffer,
                    "{} ^ {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap()
            }
            AstNode::Shl(lhs, rhs) => {
                let prec = Self::precedence(node);
                write!(
                    buffer,
                    "{} << {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap()
            }
            AstNode::Shr(lhs, rhs) => {
                let prec = Self::precedence(node);
                write!(
                    buffer,
                    "{} >> {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap()
            }
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
            AstNode::Neg(v) => {
                let prec = Self::precedence(node);
                write!(buffer, "-{}", self.render_with_parens(v, prec, false)).unwrap()
            }
            AstNode::BitNot(v) => {
                let prec = Self::precedence(node);
                write!(buffer, "~{}", self.render_with_parens(v, prec, false)).unwrap()
            }
            AstNode::Recip(v) => write!(buffer, "(1 / {})", self.render_node(v)).unwrap(),
            AstNode::Sin(v) => write!(buffer, "sin({})", self.render_node(v)).unwrap(),
            AstNode::Sqrt(v) => write!(buffer, "sqrt({})", self.render_node(v)).unwrap(),
            AstNode::Log2(v) => write!(buffer, "log2({})", self.render_node(v)).unwrap(),
            AstNode::Exp2(v) => write!(buffer, "exp2({})", self.render_node(v)).unwrap(),
            AstNode::Range {
                counter_name,
                start,
                max,
                step,
                body,
                unroll,
            } => {
                let start_str = self.render_node(start);
                let max_str = self.render_node(max);
                let step_str = self.render_node(step);

                // Generate #pragma unroll if requested
                if let Some(factor) = unroll {
                    if *factor == 0 {
                        // Full unroll
                        writeln!(buffer, "#pragma unroll").unwrap();
                    } else {
                        // Partial unroll with factor
                        writeln!(buffer, "#pragma unroll {}", factor).unwrap();
                    }
                    self.render_indent(&mut buffer);
                }

                // Generate: for (size_t i = start; i < max; i += step)
                writeln!(
                    buffer,
                    "for (size_t {counter_name} = {start_str}; {counter_name} < {max_str}; {counter_name} += {step_str})"
                )
                .unwrap();

                match *body.as_ref() {
                    AstNode::Block { .. } => {
                        self.render_indent(&mut buffer);
                        write!(buffer, "{}", self.render_node(body.as_ref())).unwrap();
                    }
                    _ => {
                        self.indent_level += 1;
                        self.render_indent(&mut buffer);
                        write!(buffer, "{}", self.render_node(body)).unwrap();
                        self.indent_level -= 1;
                    }
                }
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
            AstNode::LessThan(lhs, rhs) => {
                let prec = Self::precedence(node);
                write!(
                    buffer,
                    "{} < {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap();
            }
            AstNode::Eq(lhs, rhs) => {
                let prec = Self::precedence(node);
                write!(
                    buffer,
                    "{} == {}",
                    self.render_with_parens(lhs, prec, false),
                    self.render_with_parens(rhs, prec, true)
                )
                .unwrap();
            }
            AstNode::Select {
                cond,
                true_val,
                false_val,
            } => {
                // Render as C ternary operator: cond ? true_val : false_val
                write!(
                    buffer,
                    "{} ? {} : {}",
                    self.render_node(cond),
                    self.render_node(true_val),
                    self.render_node(false_val)
                )
                .unwrap();
            }
            AstNode::CallFunction { name, args } => {
                let args_str = args
                    .iter()
                    .map(|arg| self.render_node(arg))
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(buffer, "{}({})", name, args_str).unwrap();
            }

            AstNode::Barrier => {
                // Synchronization barrier for parallel execution
                // Currently rendered as a comment for future parallel backend support
                write!(buffer, "/* BARRIER */").unwrap();
            }

            AstNode::Function {
                name,
                scope,
                statements,
                arguments,
                return_type,
            } => {
                // Render function signature
                let (ret_type, _) = Self::render_dtype_recursive(return_type);
                let args_str = arguments
                    .iter()
                    .map(|(arg_name, dtype)| {
                        let (base_type, array_dims) = Self::render_dtype_recursive(dtype);
                        format!("{} {}{}", base_type, arg_name, array_dims)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(buffer, "{} {}({})", ret_type, name, args_str).unwrap();

                // Render function body
                write!(buffer, "{}", self.render_scope(scope, statements)).unwrap();
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

    fn render_scalar_dtype(dtype: &DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::Isize => "ssize_t".to_string(),
            DType::Usize => "size_t".to_string(),
            DType::Bool => "int".to_string(),
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
    #[case(AstNode::Deref(Box::new(var("a") + var("i"))), "*(a + i)")]
    #[case(AstNode::CallFunction { name: "my_func".to_string(), args: vec![var("a"), 2_isize.into()] }, "my_func(a, 2)")]
    // Others
    #[case(AstNode::Assign("a".to_string(), Box::new(var("b"))), "a = b")]
    #[case(AstNode::Store { target: Box::new(var("arr")), index: Box::new(var("i")), value: Box::new(var("x")) }, "*((arr) + (i)) = x")]
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
	*((arr) + (0)) = 1;
	free(arr);
}
"###;
        let mut renderer = CRenderer::new();
        assert_eq!(renderer.render(program), expected);
    }
}
