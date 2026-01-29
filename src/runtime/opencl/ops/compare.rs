//! Comparison and conditional operations for OpenCL.

use crate::dtype::DType;
use crate::runtime::opencl::kernel::dtype_to_cl_type;

/// Generates a comparison kernel (CmpLt, CmpEq).
pub fn gen_cmp_kernel(op: &str, input_dtype: DType, rank: usize) -> (String, String) {
    let cl_type = dtype_to_cl_type(input_dtype);
    let kernel_name = format!("{}_{}_{}", op, cl_type, rank);

    let expr = match op {
        "cmplt" => "a < b",
        "cmpeq" => "a == b",
        _ => panic!("Unknown comparison operation: {}", op),
    };

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const {cl_type}* a_data,
    __global const {cl_type}* b_data,
    __global uchar* output,
    const uint n,
    __global const uint* out_shape,
    __global const uint* a_shape,
    __global const uint* b_shape,
    const uint rank
) {{
    uint gid = get_global_id(0);
    if (gid < n) {{
        // Compute multi-dimensional index from flat index
        uint indices[8];
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
        }}

        {cl_type} a = a_data[a_idx];
        {cl_type} b = b_data[b_idx];
        output[gid] = ({expr}) ? 1 : 0;
    }}
}}
"#,
        kernel_name = kernel_name,
        cl_type = cl_type,
        expr = expr,
    );

    (source, kernel_name)
}

/// Generates a simple comparison kernel without broadcast.
pub fn gen_cmp_simple_kernel(op: &str, input_dtype: DType) -> (String, String) {
    let cl_type = dtype_to_cl_type(input_dtype);
    let kernel_name = format!("{}_simple_{}", op, cl_type);

    let expr = match op {
        "cmplt" => "a < b",
        "cmpeq" => "a == b",
        _ => panic!("Unknown comparison operation: {}", op),
    };

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const {cl_type}* a_data,
    __global const {cl_type}* b_data,
    __global uchar* output,
    const uint n
) {{
    uint gid = get_global_id(0);
    if (gid < n) {{
        {cl_type} a = a_data[gid];
        {cl_type} b = b_data[gid];
        output[gid] = ({expr}) ? 1 : 0;
    }}
}}
"#,
        kernel_name = kernel_name,
        cl_type = cl_type,
        expr = expr,
    );

    (source, kernel_name)
}

/// Generates a Where (ternary select) kernel with broadcast support.
pub fn gen_where_kernel(dtype: DType, rank: usize) -> (String, String) {
    let cl_type = dtype_to_cl_type(dtype);
    let kernel_name = format!("where_{}_{}", cl_type, rank);

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const uchar* cond_data,
    __global const {cl_type}* x_data,
    __global const {cl_type}* y_data,
    __global {cl_type}* output,
    const uint n,
    __global const uint* out_shape,
    __global const uint* cond_shape,
    __global const uint* x_shape,
    __global const uint* y_shape,
    const uint rank
) {{
    uint gid = get_global_id(0);
    if (gid < n) {{
        // Compute multi-dimensional index from flat index
        uint indices[8];
        uint remaining = gid;
        for (int i = rank - 1; i >= 0; i--) {{
            indices[i] = remaining % out_shape[i];
            remaining /= out_shape[i];
        }}

        // Compute cond_idx with broadcast
        uint cond_idx = 0;
        uint cond_stride = 1;
        for (int i = rank - 1; i >= 0; i--) {{
            uint idx = (cond_shape[i] == 1) ? 0 : indices[i];
            cond_idx += idx * cond_stride;
            cond_stride *= cond_shape[i];
        }}

        // Compute x_idx with broadcast
        uint x_idx = 0;
        uint x_stride = 1;
        for (int i = rank - 1; i >= 0; i--) {{
            uint idx = (x_shape[i] == 1) ? 0 : indices[i];
            x_idx += idx * x_stride;
            x_stride *= x_shape[i];
        }}

        // Compute y_idx with broadcast
        uint y_idx = 0;
        uint y_stride = 1;
        for (int i = rank - 1; i >= 0; i--) {{
            uint idx = (y_shape[i] == 1) ? 0 : indices[i];
            y_idx += idx * y_stride;
            y_stride *= y_shape[i];
        }}

        uchar c = cond_data[cond_idx];
        {cl_type} x = x_data[x_idx];
        {cl_type} y = y_data[y_idx];
        output[gid] = c ? x : y;
    }}
}}
"#,
        kernel_name = kernel_name,
        cl_type = cl_type,
    );

    (source, kernel_name)
}

/// Generates a simple Where kernel without broadcast.
pub fn gen_where_simple_kernel(dtype: DType) -> (String, String) {
    let cl_type = dtype_to_cl_type(dtype);
    let kernel_name = format!("where_simple_{}", cl_type);

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const uchar* cond_data,
    __global const {cl_type}* x_data,
    __global const {cl_type}* y_data,
    __global {cl_type}* output,
    const uint n
) {{
    uint gid = get_global_id(0);
    if (gid < n) {{
        uchar c = cond_data[gid];
        {cl_type} x = x_data[gid];
        {cl_type} y = y_data[gid];
        output[gid] = c ? x : y;
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
    fn test_gen_cmp_kernel() {
        let (source, name) = gen_cmp_simple_kernel("cmplt", DType::Float32);
        assert!(source.contains("a < b"));
        assert_eq!(name, "cmplt_simple_float");
    }

    #[test]
    fn test_gen_where_kernel() {
        let (source, name) = gen_where_simple_kernel(DType::Float32);
        assert!(source.contains("c ? x : y"));
        assert_eq!(name, "where_simple_float");
    }
}
