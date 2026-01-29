//! Elementwise operations (unary and binary) for OpenCL.

use crate::dtype::DType;
use crate::runtime::opencl::kernel::dtype_to_cl_type;

/// Generates an OpenCL kernel source for a unary operation.
pub fn gen_unary_kernel(op: &str, dtype: DType) -> (String, String) {
    let cl_type = dtype_to_cl_type(dtype);
    let kernel_name = format!("{}_{}", op, cl_type);

    let expr = match op {
        "neg" => "-x".to_string(),
        "exp" => "exp(x)".to_string(),
        "log" => "log(x)".to_string(),
        "sqrt" => "sqrt(x)".to_string(),
        "sin" => "sin(x)".to_string(),
        "cos" => "cos(x)".to_string(),
        "recip" => format!("({})1.0 / x", cl_type),
        _ => panic!("Unknown unary operation: {}", op),
    };

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const {cl_type}* input,
    __global {cl_type}* output,
    const uint n
) {{
    uint gid = get_global_id(0);
    if (gid < n) {{
        {cl_type} x = input[gid];
        output[gid] = {expr};
    }}
}}
"#,
        kernel_name = kernel_name,
        cl_type = cl_type,
        expr = expr,
    );

    (source, kernel_name)
}

/// Generates an OpenCL kernel source for a binary operation with broadcast support.
pub fn gen_binary_kernel(op: &str, dtype: DType, rank: usize) -> (String, String) {
    let cl_type = dtype_to_cl_type(dtype);
    let kernel_name = format!("{}_{}_{}", op, cl_type, rank);

    let expr = match op {
        "add" => "a + b".to_string(),
        "sub" => "a - b".to_string(),
        "mul" => "a * b".to_string(),
        "div" => "a / b".to_string(),
        "max" => "fmax(a, b)".to_string(),
        _ => panic!("Unknown binary operation: {}", op),
    };

    // Generate index calculation for broadcast
    let index_calc = gen_broadcast_index_calc(rank);

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const {cl_type}* a_data,
    __global const {cl_type}* b_data,
    __global {cl_type}* output,
    const uint n,
    __global const uint* out_shape,
    __global const uint* a_shape,
    __global const uint* b_shape,
    const uint rank
) {{
    uint gid = get_global_id(0);
    if (gid < n) {{
        {index_calc}

        {cl_type} a = a_data[a_idx];
        {cl_type} b = b_data[b_idx];
        output[gid] = {expr};
    }}
}}
"#,
        kernel_name = kernel_name,
        cl_type = cl_type,
        index_calc = index_calc,
        expr = expr,
    );

    (source, kernel_name)
}

/// Generates an OpenCL kernel for simple binary operations without broadcast.
pub fn gen_binary_simple_kernel(op: &str, dtype: DType) -> (String, String) {
    let cl_type = dtype_to_cl_type(dtype);
    let kernel_name = format!("{}_simple_{}", op, cl_type);

    let expr = match op {
        "add" => "a + b".to_string(),
        "sub" => "a - b".to_string(),
        "mul" => "a * b".to_string(),
        "div" => "a / b".to_string(),
        "max" => "fmax(a, b)".to_string(),
        _ => panic!("Unknown binary operation: {}", op),
    };

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const {cl_type}* a_data,
    __global const {cl_type}* b_data,
    __global {cl_type}* output,
    const uint n
) {{
    uint gid = get_global_id(0);
    if (gid < n) {{
        {cl_type} a = a_data[gid];
        {cl_type} b = b_data[gid];
        output[gid] = {expr};
    }}
}}
"#,
        kernel_name = kernel_name,
        cl_type = cl_type,
        expr = expr,
    );

    (source, kernel_name)
}

/// Generates broadcast index calculation code.
fn gen_broadcast_index_calc(max_rank: usize) -> String {
    // Generate code that computes indices for both inputs based on output index
    format!(
        r#"
        // Compute multi-dimensional index from flat index
        uint indices[{max_rank}];
        uint remaining = gid;
        for (int i = rank - 1; i >= 0; i--) {{
            indices[i] = remaining % out_shape[i];
            remaining /= out_shape[i];
        }}

        // Compute a_idx with broadcast
        uint a_idx = 0;
        uint a_stride = 1;
        for (int i = rank - 1; i >= 0; i--) {{
            uint idx = (a_shape[i] == 1) ? 0 : indices[i];
            a_idx += idx * a_stride;
            a_stride *= a_shape[i];
        }}

        // Compute b_idx with broadcast
        uint b_idx = 0;
        uint b_stride = 1;
        for (int i = rank - 1; i >= 0; i--) {{
            uint idx = (b_shape[i] == 1) ? 0 : indices[i];
            b_idx += idx * b_stride;
            b_stride *= b_shape[i];
        }}"#,
        max_rank = max_rank.max(8) // Ensure at least 8 for common cases
    )
}

/// Generates a const fill kernel.
pub fn gen_const_kernel(dtype: DType) -> (String, String) {
    let cl_type = dtype_to_cl_type(dtype);
    let kernel_name = format!("const_{}", cl_type);

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global {cl_type}* output,
    const {cl_type} value,
    const uint n
) {{
    uint gid = get_global_id(0);
    if (gid < n) {{
        output[gid] = value;
    }}
}}
"#,
        kernel_name = kernel_name,
        cl_type = cl_type,
    );

    (source, kernel_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_unary_kernel() {
        let (source, name) = gen_unary_kernel("neg", DType::Float32);
        assert!(source.contains("__kernel void neg_float"));
        assert!(source.contains("-x"));
        assert_eq!(name, "neg_float");
    }

    #[test]
    fn test_gen_binary_simple_kernel() {
        let (source, name) = gen_binary_simple_kernel("add", DType::Float32);
        assert!(source.contains("__kernel void add_simple_float"));
        assert!(source.contains("a + b"));
        assert_eq!(name, "add_simple_float");
    }
}
