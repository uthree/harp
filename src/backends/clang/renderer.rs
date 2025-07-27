use crate::backends::Renderer;
use crate::uop::{Op, UOp};
use log::debug;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt::Write;

/// A `Renderer` that converts a `UOp` graph into a C-style source code string.
///
/// This renderer produces a C function `kernel_main` that takes an array of buffer
/// pointers and an array of integer shape arguments. It handles basic C constructs
/// like loops and arithmetic operations.
#[derive(Debug)]
pub struct CStyleRenderer;

impl Renderer for CStyleRenderer {
    fn render(&self, uops: &[UOp]) -> String {
        let mut code = String::new();
        let mut context = CStyleRenderContext::new();
        context.collect_vars(uops);

        // Write C headers and the function signature.
        writeln!(&mut code, "#include <math.h>").unwrap();
        writeln!(&mut code, "#include <stddef.h>").unwrap();
        writeln!(
            &mut code,
            "void kernel_main(void** bufs, size_t* shape_args) {{"
        )
        .unwrap();

        // --- Argument Declaration ---
        // Buffer and integer arguments are cast to local variables for clarity and type safety.
        let mut sorted_args: Vec<_> = context.args.iter().collect();
        // Sort arguments to ensure a consistent order (data buffers first, then others).
        sorted_args.sort_by(|(a, _), (b, _)| {
            let a_is_data = a.starts_with("data");
            let b_is_data = b.starts_with("data");
            if a_is_data && b_is_data {
                let a_num: usize = a[4..].parse().unwrap_or(0);
                let b_num: usize = b[4..].parse().unwrap_or(0);
                a_num.cmp(&b_num)
            } else {
                a.cmp(b)
            }
        });

        let mut buf_idx = 0;
        let mut int_idx = 0;
        for (name, dtype) in sorted_args {
            if context.defined_vars.contains(name.as_str()) {
                continue; // Skip loop variables, etc.
            }
            if context.buffers.contains(name.as_str()) {
                writeln!(&mut code, "    {dtype}* {name} = bufs[{buf_idx}];").unwrap();
                buf_idx += 1;
            } else {
                writeln!(&mut code, "    size_t {name} = shape_args[{int_idx}];").unwrap();
                int_idx += 1;
            }
        }

        // --- Kernel Body ---
        context.render_ops(&mut code, uops, 4);
        writeln!(&mut code, "}}").unwrap();
        debug!("Rendered C code:\n{code}");
        code
    }
}

/// Holds the context for rendering a C-style kernel.
///
/// This includes tracking defined variables, arguments, and buffers to correctly
/// generate declarations and expressions.
struct CStyleRenderContext {
    /// A map of all unique variable names (arguments) to their data types.
    args: FxHashMap<String, crate::dtype::DType>,
    /// A set of variable names that are identified as buffers (pointers).
    buffers: FxHashSet<String>,
    /// A set of variable names that are defined within the kernel (e.g., loop indices).
    defined_vars: FxHashSet<String>,
}

impl CStyleRenderContext {
    /// Creates a new, empty rendering context.
    fn new() -> Self {
        Self {
            args: FxHashMap::default(),
            buffers: FxHashSet::default(),
            defined_vars: FxHashSet::default(),
        }
    }

    /// Traverses the `UOp` graph to collect all required arguments and buffers.
    ///
    /// This populates the `args`, `buffers`, and `defined_vars` fields.
    fn collect_vars(&mut self, uops: &[UOp]) {
        // First, find all loop variables and mark them as "defined" so they aren't treated as args.
        for uop in uops {
            if let Op::LoopStart = &uop.0.op {
                if let Some(loop_var) = uop.0.src.first() {
                    if let Op::Var(name) = &loop_var.0.op {
                        self.defined_vars.insert(name.clone());
                    }
                }
            }
        }

        for uop in uops {
            // Collect variables from the operation itself
            if let Op::Var(name) = &uop.0.op {
                if !self.args.contains_key(name) && !self.defined_vars.contains(name) {
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
                            if !self.args.contains_key(name) && !self.defined_vars.contains(name) {
                                self.args.insert(name.clone(), dest.0.dtype.clone());
                            }
                            // If it's a variable declaration (Store with 2 sources), mark it as defined.
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

    /// Recursively traverses a `UOp` expression to find all variable dependencies.
    fn collect_vars_recursive(&mut self, uop: &UOp) {
        // If the uop is a variable, add it to the arguments list, unless it's a defined var.
        if let Op::Var(name) = &uop.0.op {
            if !self.args.contains_key(name) && !self.defined_vars.contains(name) {
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

        // Do not treat constants as arguments, but recurse into their sources (if any).
        if let Op::Const(_) = &uop.0.op {
            return;
        }

        // Recursively traverse the sources of the expression.
        for src in &uop.0.src {
            self.collect_vars_recursive(src);
        }
    }

    /// Renders the sequence of operations (statements) into the C code.
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
                Op::Declare(name, dtype) => {
                    let value = self.render_expr(&uop.0.src[0]);
                    writeln!(code, "{indent_str}{dtype} {name} = {value};").unwrap();
                }
                Op::Store => {
                    if uop.0.src.len() == 2 {
                        // Variable re-assignment
                        let dest = self.render_expr(&uop.0.src[0]);
                        let value = self.render_expr(&uop.0.src[1]);
                        writeln!(code, "{indent_str}{dest} = {value};").unwrap();
                    } else {
                        // Array store
                        let dest = self.render_expr(&uop.0.src[0]);
                        let idx = self.render_expr(&uop.0.src[1]);
                        let value = self.render_expr(&uop.0.src[2]);
                        writeln!(code, "{indent_str}{dest}[{idx}] = {value};").unwrap();
                    }
                }
                Op::If => {
                    let condition = self.render_expr(&uop.0.src[0]);
                    writeln!(code, "{indent_str}if ({condition}) {{").unwrap();
                }
                Op::Block => {
                    self.render_ops(code, &uop.0.src, indent);
                }
                _ => panic!("Unexpected statement in kernel: {:?}", uop.0.op),
            }
        }
    }

    /// Renders a `UOp` node and its children as a C expression string.
    fn render_expr(&mut self, uop: &UOp) -> String {
        match &uop.0.op {
            Op::Add => format!(
                "({} + {})",
                self.render_expr(&uop.0.src[0]),
                self.render_expr(&uop.0.src[1])
            ),
            Op::Mul => {
                let a = &uop.0.src[0];
                let b = &uop.0.src[1];
                if let Op::Recip = b.0.op {
                    // a * (1/b) -> a / b
                    format!(
                        "({} / {})",
                        self.render_expr(a),
                        self.render_expr(&b.0.src[0])
                    )
                } else if let Op::Recip = a.0.op {
                    // (1/a) * b -> b / a
                    format!(
                        "({} / {})",
                        self.render_expr(b),
                        self.render_expr(&a.0.src[0])
                    )
                } else {
                    format!("({} * {})", self.render_expr(a), self.render_expr(b))
                }
            }
            Op::Neg => format!("(-{})", self.render_expr(&uop.0.src[0])),
            Op::Recip => format!("(1.0f / {})", self.render_expr(&uop.0.src[0])),
            Op::Rem => format!(
                "({} % {})",
                self.render_expr(&uop.0.src[0]),
                self.render_expr(&uop.0.src[1])
            ),
            Op::Load => format!(
                "{}[{}]",
                self.render_expr(&uop.0.src[0]),
                self.render_expr(&uop.0.src[1])
            ),
            Op::Const(num) => match num {
                crate::uop::Number::F32(f) => format!("{f:.1}f"),
                crate::uop::Number::F64(d) => format!("{d:.1}"),
                _ => format!("{num}"),
            },
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
