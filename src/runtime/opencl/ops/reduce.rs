//! Reduction operations for OpenCL.

use crate::dtype::DType;
use crate::runtime::opencl::kernel::dtype_to_cl_type;

/// Generates a reduction kernel (Sum, ReduceMax).
/// Uses a two-pass approach: local reduction within work-groups, then global reduction.
pub fn gen_reduce_kernel(
    op: &str,
    dtype: DType,
    axes: &[usize],
    src_rank: usize,
) -> (String, String) {
    let cl_type = dtype_to_cl_type(dtype);
    let kernel_name = format!("reduce_{}_{}_r{}", op, cl_type, src_rank);

    let (init_val, reduce_op) = match op {
        "sum" => ("0", "acc + val"),
        "max" => {
            let init = match dtype {
                DType::Float32 | DType::Float64 => "-INFINITY",
                DType::Int32 => "INT_MIN",
                DType::Int64 => "LONG_MIN",
                DType::Bool => "0",
            };
            (init, "fmax(acc, val)")
        }
        _ => panic!("Unknown reduce operation: {}", op),
    };

    // Build axis mask
    let mut axis_mask = vec![false; src_rank];
    for &axis in axes {
        if axis < src_rank {
            axis_mask[axis] = true;
        }
    }

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const {cl_type}* input,
    __global {cl_type}* output,
    const uint src_numel,
    const uint out_numel,
    __global const uint* src_shape,
    __global const uint* out_shape,
    __global const uint* axis_mask,
    const uint src_rank,
    const uint out_rank
) {{
    uint out_idx = get_global_id(0);
    if (out_idx >= out_numel) return;

    // Compute output multi-dimensional index
    uint out_indices[8];
    uint remaining = out_idx;
    for (int i = out_rank - 1; i >= 0; i--) {{
        out_indices[i] = remaining % out_shape[i];
        remaining /= out_shape[i];
    }}

    // Iterate over source elements that map to this output index
    {cl_type} acc = ({cl_type}){init_val};

    for (uint src_i = 0; src_i < src_numel; src_i++) {{
        // Compute source multi-dimensional index
        uint src_indices[8];
        uint rem = src_i;
        for (int i = src_rank - 1; i >= 0; i--) {{
            src_indices[i] = rem % src_shape[i];
            rem /= src_shape[i];
        }}

        // Check if this source element maps to our output index
        bool matches = true;
        uint out_dim = 0;
        for (uint i = 0; i < src_rank; i++) {{
            if (axis_mask[i] == 0) {{
                // Not a reduce axis - must match output index
                if (src_indices[i] != out_indices[out_dim]) {{
                    matches = false;
                    break;
                }}
                out_dim++;
            }}
        }}

        if (matches) {{
            {cl_type} val = input[src_i];
            acc = {reduce_op};
        }}
    }}

    output[out_idx] = acc;
}}
"#,
        kernel_name = kernel_name,
        cl_type = cl_type,
        init_val = init_val,
        reduce_op = reduce_op,
    );

    (source, kernel_name)
}

/// Generates a simple full reduction kernel (reduces all elements to a single value).
pub fn gen_full_reduce_kernel(op: &str, dtype: DType) -> (String, String) {
    let cl_type = dtype_to_cl_type(dtype);
    let kernel_name = format!("reduce_full_{}_{}", op, cl_type);

    let (init_val, reduce_op, local_reduce_op) = match op {
        "sum" => ("0", "acc + val", "scratch[lid] += scratch[lid + s];"),
        "max" => {
            let init = match dtype {
                DType::Float32 | DType::Float64 => "-INFINITY",
                DType::Int32 => "INT_MIN",
                DType::Int64 => "LONG_MIN",
                DType::Bool => "0",
            };
            (
                init,
                "fmax(acc, val)",
                "scratch[lid] = fmax(scratch[lid], scratch[lid + s]);",
            )
        }
        _ => panic!("Unknown reduce operation: {}", op),
    };

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const {cl_type}* input,
    __global {cl_type}* output,
    __local {cl_type}* scratch,
    const uint n
) {{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group_size = get_local_size(0);

    // Load and reduce within work-item
    {cl_type} acc = ({cl_type}){init_val};
    for (uint i = gid; i < n; i += get_global_size(0)) {{
        {cl_type} val = input[i];
        acc = {reduce_op};
    }}

    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Local reduction
    for (uint s = group_size / 2; s > 0; s >>= 1) {{
        if (lid < s) {{
            {local_reduce_op}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    // Write result
    if (lid == 0) {{
        output[get_group_id(0)] = scratch[0];
    }}
}}
"#,
        kernel_name = kernel_name,
        cl_type = cl_type,
        init_val = init_val,
        reduce_op = reduce_op,
        local_reduce_op = local_reduce_op,
    );

    (source, kernel_name)
}

/// Generates a kernel that reduces along the last axis (common case, more efficient).
pub fn gen_reduce_last_axis_kernel(op: &str, dtype: DType) -> (String, String) {
    let cl_type = dtype_to_cl_type(dtype);
    let kernel_name = format!("reduce_last_{}_{}", op, cl_type);

    let (init_val, reduce_op) = match op {
        "sum" => ("0", "acc + val"),
        "max" => {
            let init = match dtype {
                DType::Float32 | DType::Float64 => "-INFINITY",
                DType::Int32 => "INT_MIN",
                DType::Int64 => "LONG_MIN",
                DType::Bool => "0",
            };
            (init, "fmax(acc, val)")
        }
        _ => panic!("Unknown reduce operation: {}", op),
    };

    let source = format!(
        r#"
__kernel void {kernel_name}(
    __global const {cl_type}* input,
    __global {cl_type}* output,
    const uint out_numel,
    const uint reduce_size
) {{
    uint out_idx = get_global_id(0);
    if (out_idx >= out_numel) return;

    {cl_type} acc = ({cl_type}){init_val};
    uint base = out_idx * reduce_size;

    for (uint i = 0; i < reduce_size; i++) {{
        {cl_type} val = input[base + i];
        acc = {reduce_op};
    }}

    output[out_idx] = acc;
}}
"#,
        kernel_name = kernel_name,
        cl_type = cl_type,
        init_val = init_val,
        reduce_op = reduce_op,
    );

    (source, kernel_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_reduce_kernel() {
        let (source, name) = gen_reduce_last_axis_kernel("sum", DType::Float32);
        assert!(source.contains("acc + val"));
        assert_eq!(name, "reduce_last_sum_float");
    }

    #[test]
    fn test_gen_full_reduce_kernel() {
        let (source, name) = gen_full_reduce_kernel("max", DType::Float32);
        assert!(source.contains("fmax"));
        assert_eq!(name, "reduce_full_max_float");
    }
}
