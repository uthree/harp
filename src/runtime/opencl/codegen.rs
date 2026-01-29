//! Code generation for fused kernels.

use crate::dtype::ScalarValue;
use crate::ops::Ops;
use crate::runtime::opencl::kernel::dtype_to_cl_type;
use crate::schedule::{FusedKernel, FusedOp, FusedSource};

/// Generates OpenCL kernel source code for a fused kernel.
pub struct FusedKernelCodeGen<'a> {
    kernel: &'a FusedKernel,
}

impl<'a> FusedKernelCodeGen<'a> {
    /// Create a new code generator for the given fused kernel.
    pub fn new(kernel: &'a FusedKernel) -> Self {
        Self { kernel }
    }

    /// Generate the OpenCL kernel source code and kernel name.
    pub fn generate(&self) -> (String, String) {
        let kernel_name = &self.kernel.name;
        let output_type = dtype_to_cl_type(self.kernel.output_dtype);

        // Generate input parameters
        let input_params = self.generate_input_params();

        // Generate operation chain
        let ops_code = self.generate_ops_chain();

        let source = format!(
            r#"
__kernel void {kernel_name}(
    {input_params}
    __global {output_type}* output,
    const uint n
) {{
    uint gid = get_global_id(0);
    if (gid < n) {{
{ops_code}
        output[gid] = v{last_idx};
    }}
}}
"#,
            kernel_name = kernel_name,
            input_params = input_params,
            output_type = output_type,
            ops_code = ops_code,
            last_idx = self.kernel.ops_chain.len() - 1,
        );

        (source, kernel_name.clone())
    }

    fn generate_input_params(&self) -> String {
        let params: Vec<String> = self
            .kernel
            .inputs
            .iter()
            .enumerate()
            .map(|(i, input)| {
                let cl_type = dtype_to_cl_type(input.dtype);
                format!("    __global const {}* in{},", cl_type, i)
            })
            .collect();

        params.join("\n")
    }

    fn generate_ops_chain(&self) -> String {
        let lines: Vec<String> = self
            .kernel
            .ops_chain
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let expr = self.generate_op_expr(op, i);
                let cl_type = dtype_to_cl_type(op.dtype);
                format!("        {} v{} = {};", cl_type, i, expr)
            })
            .collect();

        lines.join("\n")
    }

    fn generate_op_expr(&self, op: &FusedOp, _op_idx: usize) -> String {
        match op.op {
            // Unary operations
            Ops::Neg => {
                let src = self.source_expr(&op.sources[0]);
                format!("-({})", src)
            }
            Ops::Exp => {
                let src = self.source_expr(&op.sources[0]);
                format!("exp({})", src)
            }
            Ops::Log => {
                let src = self.source_expr(&op.sources[0]);
                format!("log({})", src)
            }
            Ops::Sqrt => {
                let src = self.source_expr(&op.sources[0]);
                format!("sqrt({})", src)
            }
            Ops::Sin => {
                let src = self.source_expr(&op.sources[0]);
                format!("sin({})", src)
            }
            Ops::Cos => {
                let src = self.source_expr(&op.sources[0]);
                format!("cos({})", src)
            }
            Ops::Recip => {
                let src = self.source_expr(&op.sources[0]);
                let cl_type = dtype_to_cl_type(op.dtype);
                format!("({})1.0 / ({})", cl_type, src)
            }

            // Binary operations
            Ops::Add => {
                let a = self.source_expr(&op.sources[0]);
                let b = self.source_expr(&op.sources[1]);
                format!("({}) + ({})", a, b)
            }
            Ops::Sub => {
                let a = self.source_expr(&op.sources[0]);
                let b = self.source_expr(&op.sources[1]);
                format!("({}) - ({})", a, b)
            }
            Ops::Mul => {
                let a = self.source_expr(&op.sources[0]);
                let b = self.source_expr(&op.sources[1]);
                format!("({}) * ({})", a, b)
            }
            Ops::Div => {
                let a = self.source_expr(&op.sources[0]);
                let b = self.source_expr(&op.sources[1]);
                format!("({}) / ({})", a, b)
            }
            Ops::Max => {
                let a = self.source_expr(&op.sources[0]);
                let b = self.source_expr(&op.sources[1]);
                format!("fmax({}, {})", a, b)
            }

            // Comparison operations
            Ops::CmpLt => {
                let a = self.source_expr(&op.sources[0]);
                let b = self.source_expr(&op.sources[1]);
                format!("({}) < ({}) ? 1 : 0", a, b)
            }
            Ops::CmpEq => {
                let a = self.source_expr(&op.sources[0]);
                let b = self.source_expr(&op.sources[1]);
                format!("({}) == ({}) ? 1 : 0", a, b)
            }

            // Ternary operations
            Ops::Where => {
                let cond = self.source_expr(&op.sources[0]);
                let then_val = self.source_expr(&op.sources[1]);
                let else_val = self.source_expr(&op.sources[2]);
                format!("({}) ? ({}) : ({})", cond, then_val, else_val)
            }

            _ => panic!("Unsupported operation in fused kernel: {:?}", op.op),
        }
    }

    fn source_expr(&self, source: &FusedSource) -> String {
        match source {
            FusedSource::Input(idx) => format!("in{}[gid]", idx),
            FusedSource::PrevOp(idx) => format!("v{}", idx),
            FusedSource::Constant(val) => self.scalar_literal(val),
        }
    }

    fn scalar_literal(&self, val: &ScalarValue) -> String {
        match val {
            ScalarValue::Bool(b) => if *b { "1" } else { "0" }.to_string(),
            ScalarValue::Int32(v) => format!("{}", v),
            ScalarValue::Int64(v) => format!("{}L", v),
            ScalarValue::Float32(v) => {
                if v.is_nan() {
                    "NAN".to_string()
                } else if v.is_infinite() {
                    if *v > 0.0 {
                        "INFINITY".to_string()
                    } else {
                        "-INFINITY".to_string()
                    }
                } else {
                    format!("{:.8}f", v)
                }
            }
            ScalarValue::Float64(v) => {
                if v.is_nan() {
                    "(double)NAN".to_string()
                } else if v.is_infinite() {
                    if *v > 0.0 {
                        "(double)INFINITY".to_string()
                    } else {
                        "-(double)INFINITY".to_string()
                    }
                } else {
                    format!("{:.16}", v)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::schedule::{FusedOp, FusedSource, KernelInput};
    use crate::shape::Shape;

    #[test]
    fn test_simple_fused_kernel() {
        // Test: add(a, b) -> neg
        let kernel = FusedKernel::new(
            "fused_add_neg".to_string(),
            vec![
                FusedOp::new(
                    Ops::Add,
                    vec![FusedSource::Input(0), FusedSource::Input(1)],
                    DType::Float32,
                ),
                FusedOp::new(
                    Ops::Neg,
                    vec![FusedSource::PrevOp(0)],
                    DType::Float32,
                ),
            ],
            vec![
                KernelInput::new(0, DType::Float32, Shape::from(vec![4])),
                KernelInput::new(1, DType::Float32, Shape::from(vec![4])),
            ],
            Shape::from(vec![4]),
            DType::Float32,
        );

        let codegen = FusedKernelCodeGen::new(&kernel);
        let (source, name) = codegen.generate();

        assert_eq!(name, "fused_add_neg");
        assert!(source.contains("__kernel void fused_add_neg"));
        assert!(source.contains("__global const float* in0"));
        assert!(source.contains("__global const float* in1"));
        assert!(source.contains("float v0 = (in0[gid]) + (in1[gid])"));
        assert!(source.contains("float v1 = -(v0)"));
        assert!(source.contains("output[gid] = v1"));
    }

    #[test]
    fn test_kernel_with_constant() {
        // Test: mul(a, 2.0)
        let kernel = FusedKernel::new(
            "mul_const".to_string(),
            vec![FusedOp::new(
                Ops::Mul,
                vec![
                    FusedSource::Input(0),
                    FusedSource::Constant(ScalarValue::Float32(2.0)),
                ],
                DType::Float32,
            )],
            vec![KernelInput::new(0, DType::Float32, Shape::from(vec![4]))],
            Shape::from(vec![4]),
            DType::Float32,
        );

        let codegen = FusedKernelCodeGen::new(&kernel);
        let (source, _name) = codegen.generate();

        assert!(source.contains("2.0"));
    }
}
