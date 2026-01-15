//! CUDA code renderer
//!
//! Converts AST to CUDA kernel source code.

use eclat::ast::{AstNode, DType};
use eclat::backend::renderer::CLikeRenderer;
use eclat::backend::Renderer;

/// CUDA kernel code wrapper type
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaCode(String);

impl CudaCode {
    /// Create new CudaCode
    pub fn new(code: String) -> Self {
        Self(code)
    }

    /// Get reference to inner string
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Take ownership of inner string
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Get code length in bytes
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if code is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Check if code contains a pattern
    pub fn contains(&self, pat: &str) -> bool {
        self.0.contains(pat)
    }
}

impl From<String> for CudaCode {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<CudaCode> for String {
    fn from(code: CudaCode) -> Self {
        code.into_inner()
    }
}

impl AsRef<str> for CudaCode {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for CudaCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// CUDA code renderer
///
/// Converts AST to CUDA kernel source code.
#[derive(Debug, Clone)]
pub struct CudaRenderer {
    indent_level: usize,
}

impl CudaRenderer {
    /// Create a new CUDA renderer
    pub fn new() -> Self {
        Self { indent_level: 0 }
    }

    /// Render a program to CUDA code
    pub fn render_program(&mut self, program: &AstNode) -> CudaCode {
        if let AstNode::Program { functions, .. } = program {
            let mut code = String::new();

            // Header
            code.push_str(&CLikeRenderer::render_header(self));
            code.push('\n');

            // Render kernel functions
            for func in functions {
                match func {
                    AstNode::Kernel { name: Some(_), .. } => {
                        code.push_str(&self.render_kernel_function(func));
                        code.push_str("\n\n");
                    }
                    AstNode::Kernel { name: None, .. } => {
                        log::warn!("CudaRenderer: Kernel with no name found");
                    }
                    AstNode::Function { name: Some(_), .. } => {
                        code.push_str(&self.render_sequential_function_as_kernel(func));
                        code.push_str("\n\n");
                    }
                    AstNode::Function { name: None, .. } => {
                        log::warn!("CudaRenderer: Function with no name found");
                    }
                    _ => {}
                }
            }

            CudaCode::new(code)
        } else {
            CudaCode::new(String::new())
        }
    }

    /// Render a kernel function
    fn render_kernel_function(&mut self, func: &AstNode) -> String {
        self.render_function_node(func)
    }

    /// Render a sequential function as a CUDA kernel
    fn render_sequential_function_as_kernel(&mut self, func: &AstNode) -> String {
        if let AstNode::Function {
            name,
            params,
            return_type,
            body,
        } = func
        {
            let mut code = String::new();

            // __global__ qualifier
            code.push_str("__global__ ");

            // Return type
            code.push_str(&self.render_dtype_backend(return_type));
            code.push(' ');

            // Function name
            if let Some(n) = name {
                code.push_str(n);
            }

            // Parameters
            code.push('(');

            if params.is_empty() {
                use eclat::backend::renderer::extract_buffer_placeholders;
                let (inputs, has_output) = extract_buffer_placeholders(body);
                let mut buffer_params: Vec<String> = Vec::new();

                // Input buffer parameters
                for input_name in &inputs {
                    buffer_params.push(format!("const float* {}", input_name));
                }

                // Output buffer parameter
                if has_output {
                    buffer_params.push("float* output".to_string());
                }

                code.push_str(&buffer_params.join(", "));
            } else {
                let rendered_params: Vec<String> = params
                    .iter()
                    .map(|p| self.render_param_attribute(p, true))
                    .filter(|s| !s.is_empty())
                    .collect();
                code.push_str(&rendered_params.join(", "));
            }
            code.push_str(") {\n");

            // Function body
            *self.indent_level_mut() += 1;
            code.push_str(&self.render_statement(body));
            *self.indent_level_mut() -= 1;

            code.push_str("}\n");

            code
        } else {
            String::new()
        }
    }
}

impl Default for CudaRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for CudaRenderer {
    type CodeRepr = CudaCode;
    type Option = ();

    fn render(&self, program: &AstNode) -> Self::CodeRepr {
        let mut renderer = self.clone();
        renderer.render_program(program)
    }

    fn is_available(&self) -> bool {
        true
    }
}

impl CLikeRenderer for CudaRenderer {
    fn indent_level(&self) -> usize {
        self.indent_level
    }

    fn indent_level_mut(&mut self) -> &mut usize {
        &mut self.indent_level
    }

    fn render_dtype_backend(&self, dtype: &DType) -> String {
        match dtype {
            DType::Void => "void".to_string(),
            DType::Bool => "bool".to_string(),
            DType::I8 => "char".to_string(),
            DType::I16 => "short".to_string(),
            DType::I32 => "int".to_string(),
            DType::I64 => "long long".to_string(),
            DType::U8 => "unsigned char".to_string(),
            DType::U16 => "unsigned short".to_string(),
            DType::U32 => "unsigned int".to_string(),
            DType::U64 => "unsigned long long".to_string(),
            DType::F32 => "float".to_string(),
            DType::F64 => "double".to_string(),
            DType::Int => "int".to_string(), // Index type: 32-bit for GPU efficiency
            DType::Ptr(inner) => {
                // CUDA doesn't need __global qualifier for pointers
                let base = self.render_dtype_backend(inner);
                format!("{}*", base)
            }
            DType::Vec(inner, size) => {
                // CUDA vector types (float2, float4, etc.)
                let base = self.render_dtype_backend(inner);
                format!("{}{}", base, size)
            }
            DType::Tuple(types) => {
                if types.is_empty() {
                    "void".to_string()
                } else {
                    format!("tuple_{}", types.len())
                }
            }
            DType::Unknown => {
                panic!(
                    "Type inference failed: DType::Unknown should not appear in code generation."
                )
            }
            DType::Complex32 => {
                // CUDA has cuComplex, but we use float2 for simplicity
                "float2".to_string()
            }
            DType::Complex64 => {
                // CUDA has cuDoubleComplex, but we use double2 for simplicity
                "double2".to_string()
            }
        }
    }

    fn render_barrier_backend(&self) -> String {
        // CUDA thread synchronization
        "__syncthreads();".to_string()
    }

    fn render_header(&self) -> String {
        let mut header = String::new();

        header.push_str("// CUDA Kernel Code\n");
        header.push_str("// Generated by Eclat\n\n");

        // CUDA headers (for math functions)
        header.push_str("#include <cuda_runtime.h>\n");
        header.push_str("#include <math.h>\n\n");

        // Helper for atomic operations on float
        header.push_str("// Atomic add for float using CAS\n");
        header.push_str("__device__ inline float atomicAddFloat(float* address, float val) {\n");
        header.push_str("    int* address_as_int = (int*)address;\n");
        header.push_str("    int old = *address_as_int, assumed;\n");
        header.push_str("    do {\n");
        header.push_str("        assumed = old;\n");
        header.push_str("        old = atomicCAS(address_as_int, assumed,\n");
        header.push_str("            __float_as_int(val + __int_as_float(assumed)));\n");
        header.push_str("    } while (assumed != old);\n");
        header.push_str("    return __int_as_float(old);\n");
        header.push_str("}\n\n");

        header
    }

    fn render_function_qualifier(&self, is_kernel: bool) -> String {
        if is_kernel {
            "__global__ ".to_string()
        } else {
            "__device__ ".to_string()
        }
    }

    fn render_param_attribute(&self, param: &eclat::ast::VarDecl, _is_kernel: bool) -> String {
        use eclat::ast::{Mutability, VarKind};

        let type_str = self.render_dtype_backend(&param.dtype);
        let mut_str = match param.mutability {
            Mutability::Immutable => "const ",
            Mutability::Mutable => "",
        };

        // GroupId, LocalId etc. are not function parameters in CUDA
        // They are accessed via built-in variables (threadIdx, blockIdx)
        match &param.kind {
            VarKind::Normal => {
                format!("{}{} {}", mut_str, type_str, param.name)
            }
            VarKind::GroupId(_)
            | VarKind::LocalId(_)
            | VarKind::GroupSize(_)
            | VarKind::GridSize(_) => {
                // These are declared inside the function body
                String::new()
            }
        }
    }

    fn render_thread_var_declarations(
        &self,
        params: &[eclat::ast::VarDecl],
        indent: &str,
    ) -> String {
        use eclat::ast::VarKind;

        let mut declarations = String::new();

        for param in params {
            let axis_char = |axis: &usize| match axis {
                0 => "x",
                1 => "y",
                2 => "z",
                _ => "x",
            };

            match &param.kind {
                VarKind::GroupId(axis) => {
                    // blockIdx.x/y/z
                    declarations.push_str(&format!(
                        "{}int {} = blockIdx.{};\n",
                        indent,
                        param.name,
                        axis_char(axis)
                    ));
                }
                VarKind::LocalId(axis) => {
                    // threadIdx.x/y/z
                    declarations.push_str(&format!(
                        "{}int {} = threadIdx.{};\n",
                        indent,
                        param.name,
                        axis_char(axis)
                    ));
                }
                VarKind::GroupSize(axis) => {
                    // blockDim.x/y/z
                    declarations.push_str(&format!(
                        "{}int {} = blockDim.{};\n",
                        indent,
                        param.name,
                        axis_char(axis)
                    ));
                }
                VarKind::GridSize(axis) => {
                    // gridDim.x/y/z * blockDim.x/y/z (total global size)
                    let a = axis_char(axis);
                    declarations.push_str(&format!(
                        "{}int {} = gridDim.{} * blockDim.{};\n",
                        indent, param.name, a, a
                    ));
                }
                VarKind::Normal => {}
            }
        }

        declarations
    }

    fn render_math_func(&self, name: &str, args: &[String]) -> String {
        // CUDA provides standard C math functions
        // For float, use f-suffixed versions for better performance
        match name {
            "max" => {
                if args.len() == 2 {
                    format!("fmaxf({}, {})", args[0], args[1])
                } else {
                    format!("max({})", args.join(", "))
                }
            }
            "min" => {
                if args.len() == 2 {
                    format!("fminf({}, {})", args[0], args[1])
                } else {
                    format!("min({})", args.join(", "))
                }
            }
            "sqrt" => format!("sqrtf({})", args[0]),
            "exp" => format!("expf({})", args[0]),
            "exp2" => format!("exp2f({})", args[0]),
            "log" => format!("logf({})", args[0]),
            "log2" => format!("log2f({})", args[0]),
            "sin" => format!("sinf({})", args[0]),
            "cos" => format!("cosf({})", args[0]),
            "tan" => format!("tanf({})", args[0]),
            "tanh" => format!("tanhf({})", args[0]),
            "abs" => format!("fabsf({})", args[0]),
            "floor" => format!("floorf({})", args[0]),
            "ceil" => format!("ceilf({})", args[0]),
            "rsqrt" => format!("rsqrtf({})", args[0]), // CUDA has rsqrtf
            "pow" => {
                if args.len() == 2 {
                    format!("powf({}, {})", args[0], args[1])
                } else {
                    format!("pow({})", args.join(", "))
                }
            }
            _ => format!("{}({})", name, args.join(", ")),
        }
    }

    fn render_vector_load(&self, ptr_expr: &str, offset_expr: &str, dtype: &str) -> String {
        // CUDA doesn't have vload like OpenCL, use pointer cast
        let vec_size = dtype.chars().last().and_then(|c| c.to_digit(10));
        match vec_size {
            Some(_n) => format!(
                "*(({} *)(&{}[{}]))",
                dtype, ptr_expr, offset_expr
            ),
            None => format!("{}[{}]", ptr_expr, offset_expr),
        }
    }

    fn render_vector_store(
        &self,
        ptr_expr: &str,
        offset_expr: &str,
        value_expr: &str,
        dtype: &str,
    ) -> String {
        // CUDA doesn't have vstore like OpenCL, use pointer cast
        let vec_size = dtype.chars().last().and_then(|c| c.to_digit(10));
        match vec_size {
            Some(_) => format!(
                "*(({} *)(&{}[{}])) = {}",
                dtype, ptr_expr, offset_expr, value_expr
            ),
            None => format!("{}[{}] = {}", ptr_expr, offset_expr, value_expr),
        }
    }

    fn render_atomic_add(&self, ptr: &str, offset: &str, value: &str, dtype: &DType) -> String {
        match dtype {
            DType::I32 | DType::U32 | DType::I64 | DType::U64 => {
                // CUDA has native atomicAdd for integers
                format!("atomicAdd(&{}[{}], {})", ptr, offset, value)
            }
            DType::F32 => {
                // CUDA 2.0+ has atomicAdd for float
                // But we provide a fallback for older architectures
                format!("atomicAdd(&{}[{}], {})", ptr, offset, value)
            }
            DType::F64 => {
                // CUDA 6.0+ has atomicAdd for double
                format!("atomicAdd(&{}[{}], {})", ptr, offset, value)
            }
            _ => format!(
                "/* unsupported atomic_add for {:?} */ {}[{}] += {}",
                dtype, ptr, offset, value
            ),
        }
    }

    fn render_atomic_max(&self, ptr: &str, offset: &str, value: &str, dtype: &DType) -> String {
        match dtype {
            DType::I32 | DType::U32 | DType::I64 | DType::U64 => {
                // CUDA has native atomicMax for integers
                format!("atomicMax(&{}[{}], {})", ptr, offset, value)
            }
            DType::F32 => {
                // CUDA doesn't have atomicMax for float, need CAS loop
                // For simplicity, use a compare-and-swap pattern
                format!(
                    "/* atomic_max_float */ {{ \
                    float old = {}[{}]; \
                    while (old < {} && atomicCAS((int*)&{}[{}], __float_as_int(old), __float_as_int({})) != __float_as_int(old)) \
                        old = {}[{}]; \
                    }}",
                    ptr, offset, value, ptr, offset, value, ptr, offset
                )
            }
            _ => format!(
                "/* unsupported atomic_max for {:?} */ {}[{}] = max({}[{}], {})",
                dtype, ptr, offset, ptr, offset, value
            ),
        }
    }
}

/// Implementation of KernelSourceRenderer for CUDA backend
impl eclat::backend::pipeline::KernelSourceRenderer for CudaRenderer {
    fn render_kernel_source(&mut self, program: &AstNode) -> String {
        if let AstNode::Program { functions, .. } = program {
            let mut code = String::new();

            // CUDA kernel header
            code.push_str(&CLikeRenderer::render_header(self));
            code.push('\n');

            // Render all kernel/function nodes
            for func in functions {
                match func {
                    AstNode::Kernel { name: Some(_), .. } => {
                        code.push_str(&self.render_kernel_function(func));
                        code.push_str("\n\n");
                    }
                    AstNode::Function { name: Some(_), .. } => {
                        code.push_str(&self.render_sequential_function_as_kernel(func));
                        code.push_str("\n\n");
                    }
                    _ => {}
                }
            }

            code
        } else {
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eclat::ast::{Mutability, VarDecl, VarKind};
    use eclat::backend::renderer::CLikeRenderer;

    #[test]
    fn test_render_header() {
        let renderer = CudaRenderer::new();
        let header = CLikeRenderer::render_header(&renderer);

        assert!(header.contains("CUDA Kernel Code"));
        assert!(header.contains("cuda_runtime.h"));
    }

    #[test]
    fn test_render_dtype_backend() {
        let renderer = CudaRenderer::new();

        // Basic types
        assert_eq!(renderer.render_dtype_backend(&DType::Bool), "bool");
        assert_eq!(renderer.render_dtype_backend(&DType::F32), "float");
        assert_eq!(renderer.render_dtype_backend(&DType::I64), "long long");

        // Pointer type (no __global in CUDA)
        assert_eq!(
            renderer.render_dtype_backend(&DType::Ptr(Box::new(DType::F32))),
            "float*"
        );

        // Vector type
        assert_eq!(
            renderer.render_dtype_backend(&DType::Vec(Box::new(DType::F32), 4)),
            "float4"
        );
    }

    #[test]
    fn test_render_function_qualifier() {
        let renderer = CudaRenderer::new();

        assert_eq!(renderer.render_function_qualifier(true), "__global__ ");
        assert_eq!(renderer.render_function_qualifier(false), "__device__ ");
    }

    #[test]
    fn test_render_barrier() {
        let renderer = CudaRenderer::new();
        let barrier = renderer.render_barrier_backend();
        assert_eq!(barrier, "__syncthreads();");
    }

    #[test]
    fn test_render_thread_var_declarations() {
        let renderer = CudaRenderer::new();

        let params = vec![
            VarDecl {
                name: "gidx".to_string(),
                dtype: DType::I64,
                kind: VarKind::GroupId(0),
                mutability: Mutability::Immutable,
            },
            VarDecl {
                name: "lidx".to_string(),
                dtype: DType::I64,
                kind: VarKind::LocalId(0),
                mutability: Mutability::Immutable,
            },
            VarDecl {
                name: "block_size".to_string(),
                dtype: DType::I64,
                kind: VarKind::GroupSize(0),
                mutability: Mutability::Immutable,
            },
        ];

        let decls = renderer.render_thread_var_declarations(&params, "    ");
        assert!(decls.contains("int gidx = blockIdx.x"));
        assert!(decls.contains("int lidx = threadIdx.x"));
        assert!(decls.contains("int block_size = blockDim.x"));
    }

    #[test]
    fn test_render_math_functions() {
        let renderer = CudaRenderer::new();

        // float versions
        assert_eq!(
            renderer.render_math_func("sqrt", &["x".to_string()]),
            "sqrtf(x)"
        );
        assert_eq!(
            renderer.render_math_func("sin", &["x".to_string()]),
            "sinf(x)"
        );
        assert_eq!(
            renderer.render_math_func("max", &["a".to_string(), "b".to_string()]),
            "fmaxf(a, b)"
        );
    }
}
