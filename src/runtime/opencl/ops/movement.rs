//! Movement operations (Expand, Permute, Cast) for OpenCL.

use crate::dtype::DType;
use crate::runtime::opencl::kernel::dtype_to_cl_type;

/// Generates an Expand kernel with broadcast.
pub fn gen_expand_kernel(dtype: DType, rank: usize) -> (String, String) {
    let cl_type = dtype_to_cl_type(dtype);
    let kernel_name = format!("expand_{}_{}", cl_type, rank);

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const {cl_type}* input,
    __global {cl_type}* output,
    const uint n,
    __global const uint* out_shape,
    __global const uint* src_shape,
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

        // Compute source index with broadcast
        uint src_idx = 0;
        uint src_stride = 1;
        for (int i = rank - 1; i >= 0; i--) {{
            uint idx = (src_shape[i] == 1) ? 0 : indices[i];
            src_idx += idx * src_stride;
            src_stride *= src_shape[i];
        }}

        output[gid] = input[src_idx];
    }}
}}
"#,
        kernel_name = kernel_name,
        cl_type = cl_type,
    );

    (source, kernel_name)
}

/// Generates a Permute kernel.
pub fn gen_permute_kernel(dtype: DType, rank: usize) -> (String, String) {
    let cl_type = dtype_to_cl_type(dtype);
    let kernel_name = format!("permute_{}_{}", cl_type, rank);

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const {cl_type}* input,
    __global {cl_type}* output,
    const uint n,
    __global const uint* out_shape,
    __global const uint* src_shape,
    __global const uint* axes,
    const uint rank
) {{
    uint gid = get_global_id(0);
    if (gid < n) {{
        // Compute output multi-dimensional index from flat index
        uint out_indices[8];
        uint remaining = gid;
        for (int i = rank - 1; i >= 0; i--) {{
            out_indices[i] = remaining % out_shape[i];
            remaining /= out_shape[i];
        }}

        // Compute source indices by reversing the permutation
        uint src_indices[8];
        for (uint i = 0; i < rank; i++) {{
            src_indices[axes[i]] = out_indices[i];
        }}

        // Compute flat source index
        uint src_idx = 0;
        uint src_stride = 1;
        for (int i = rank - 1; i >= 0; i--) {{
            src_idx += src_indices[i] * src_stride;
            src_stride *= src_shape[i];
        }}

        output[gid] = input[src_idx];
    }}
}}
"#,
        kernel_name = kernel_name,
        cl_type = cl_type,
    );

    (source, kernel_name)
}

/// Generates a Cast kernel.
pub fn gen_cast_kernel(src_dtype: DType, dst_dtype: DType) -> (String, String) {
    let src_type = dtype_to_cl_type(src_dtype);
    let dst_type = dtype_to_cl_type(dst_dtype);
    let kernel_name = format!("cast_{}_{}", src_type, dst_type);

    // Special handling for bool conversions
    let conversion = if src_dtype == DType::Bool {
        format!("({})(input[gid] != 0)", dst_type)
    } else if dst_dtype == DType::Bool {
        "input[gid] != 0 ? 1 : 0".to_string()
    } else {
        format!("({})(input[gid])", dst_type)
    };

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const {src_type}* input,
    __global {dst_type}* output,
    const uint n
) {{
    uint gid = get_global_id(0);
    if (gid < n) {{
        output[gid] = {conversion};
    }}
}}
"#,
        kernel_name = kernel_name,
        src_type = src_type,
        dst_type = dst_type,
        conversion = conversion,
    );

    (source, kernel_name)
}

/// Generates a copy kernel (identity movement).
pub fn gen_copy_kernel(dtype: DType) -> (String, String) {
    let cl_type = dtype_to_cl_type(dtype);
    let kernel_name = format!("copy_{}", cl_type);

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const {cl_type}* input,
    __global {cl_type}* output,
    const uint n
) {{
    uint gid = get_global_id(0);
    if (gid < n) {{
        output[gid] = input[gid];
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
    fn test_gen_expand_kernel() {
        let (source, name) = gen_expand_kernel(DType::Float32, 2);
        assert!(source.contains("__kernel void expand_float_2"));
        assert_eq!(name, "expand_float_2");
    }

    #[test]
    fn test_gen_permute_kernel() {
        let (source, name) = gen_permute_kernel(DType::Float32, 3);
        assert!(source.contains("__kernel void permute_float_3"));
        assert_eq!(name, "permute_float_3");
    }

    #[test]
    fn test_gen_cast_kernel() {
        let (source, name) = gen_cast_kernel(DType::Float32, DType::Int32);
        assert!(source.contains("__kernel void cast_float_int"));
        assert_eq!(name, "cast_float_int");
    }
}
