//! Vector-Jacobian Product (VJP) implementations for each operation.
//!
//! This module defines how gradients are propagated backward through each
//! operation in the computation graph.

use crate::Tensor;
use crate::ops::Ops;
use crate::shape::Shape;
use crate::uop::{UOp, UOpArg};

/// Computes the VJP (backward pass) for a given operation.
///
/// # Arguments
/// * `output_uop` - The UOp node for which we're computing gradients
/// * `grad_output` - The gradient of the loss with respect to the output of this operation
///
/// # Returns
/// A vector of (input_ptr_id, gradient) pairs for inputs that require gradients.
pub fn compute_vjp(output_uop: &UOp, grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    let op = output_uop.op();
    let srcs = output_uop.src();

    match op {
        // Unary operations
        Ops::Neg => vjp_neg(srcs, grad_output),
        Ops::Exp => vjp_exp(srcs, output_uop, grad_output),
        Ops::Log => vjp_log(srcs, grad_output),
        Ops::Sqrt => vjp_sqrt(output_uop, grad_output),
        Ops::Recip => vjp_recip(output_uop, grad_output),
        Ops::Sin => vjp_sin(srcs, grad_output),
        Ops::Cos => vjp_cos(srcs, grad_output),

        // Binary operations
        Ops::Add => vjp_add(srcs, grad_output),
        Ops::Sub => vjp_sub(srcs, grad_output),
        Ops::Mul => vjp_mul(srcs, grad_output),
        Ops::Div => vjp_div(srcs, grad_output),
        Ops::Max => vjp_max(srcs, output_uop, grad_output),

        // Reduce operations
        Ops::Sum => vjp_sum(srcs, output_uop, grad_output),
        Ops::ReduceMax => vjp_reduce_max(srcs, output_uop, grad_output),

        // Movement operations
        Ops::Reshape => vjp_reshape(srcs, grad_output),
        Ops::Expand => vjp_expand(srcs, grad_output),
        Ops::Permute => vjp_permute(srcs, output_uop, grad_output),

        // Cast (pass gradient through for same-category types)
        Ops::Cast => vjp_cast(srcs, grad_output),

        // Comparison ops don't propagate gradients
        Ops::CmpLt | Ops::CmpEq => vec![],

        // Where propagates gradients to both branches
        Ops::Where => vjp_where(srcs, grad_output),

        // Load and Const are leaf nodes
        Ops::Load | Ops::Const => vec![],

        // Pad operation
        Ops::Pad => vjp_pad(srcs, output_uop, grad_output),

        // Store is not used in forward computation graphs
        Ops::Store => vec![],

        // Shrink and Stride - similar to slice operations
        Ops::Shrink => vjp_shrink(srcs, output_uop, grad_output),
        Ops::Stride => vjp_stride(srcs, output_uop, grad_output),
    }
}

// ============ Unary VJPs ============

fn vjp_neg(srcs: &[UOp], grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // y = -x => dx = -dy
    vec![(srcs[0].ptr_id(), grad_output.neg())]
}

fn vjp_exp(srcs: &[UOp], output_uop: &UOp, grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // y = exp(x) => dx = dy * y
    let output = Tensor::from_uop(output_uop.clone(), "CPU");
    vec![(srcs[0].ptr_id(), grad_output.mul(&output))]
}

fn vjp_log(srcs: &[UOp], grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // y = log(x) => dx = dy / x
    let x = Tensor::from_uop(srcs[0].clone(), "CPU");
    vec![(srcs[0].ptr_id(), grad_output.div(&x))]
}

fn vjp_sqrt(output_uop: &UOp, grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // y = sqrt(x) => dx = dy / (2 * y)
    let y = Tensor::from_uop(output_uop.clone(), "CPU");
    let two = Tensor::full(Shape::scalar(), 2.0f32);
    let grad = grad_output.div(&y.mul(&two));
    vec![(output_uop.src()[0].ptr_id(), grad)]
}

fn vjp_recip(output_uop: &UOp, grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // y = 1/x => dx = -dy * y^2 = -dy / x^2
    let y = Tensor::from_uop(output_uop.clone(), "CPU");
    let grad = grad_output.neg().mul(&y).mul(&y);
    vec![(output_uop.src()[0].ptr_id(), grad)]
}

fn vjp_sin(srcs: &[UOp], grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // y = sin(x) => dx = dy * cos(x)
    let x = Tensor::from_uop(srcs[0].clone(), "CPU");
    vec![(srcs[0].ptr_id(), grad_output.mul(&x.cos()))]
}

fn vjp_cos(srcs: &[UOp], grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // y = cos(x) => dx = -dy * sin(x)
    let x = Tensor::from_uop(srcs[0].clone(), "CPU");
    vec![(srcs[0].ptr_id(), grad_output.neg().mul(&x.sin()))]
}

// ============ Binary VJPs ============

fn vjp_add(srcs: &[UOp], grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // z = x + y => dx = dz, dy = dz
    let grad_x = reduce_broadcast(grad_output, srcs[0].shape());
    let grad_y = reduce_broadcast(grad_output, srcs[1].shape());
    vec![(srcs[0].ptr_id(), grad_x), (srcs[1].ptr_id(), grad_y)]
}

fn vjp_sub(srcs: &[UOp], grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // z = x - y => dx = dz, dy = -dz
    let grad_x = reduce_broadcast(grad_output, srcs[0].shape());
    let grad_y = reduce_broadcast(&grad_output.neg(), srcs[1].shape());
    vec![(srcs[0].ptr_id(), grad_x), (srcs[1].ptr_id(), grad_y)]
}

fn vjp_mul(srcs: &[UOp], grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // z = x * y => dx = dz * y, dy = dz * x
    let x = Tensor::from_uop(srcs[0].clone(), "CPU");
    let y = Tensor::from_uop(srcs[1].clone(), "CPU");
    let grad_x = reduce_broadcast(&grad_output.mul(&y), srcs[0].shape());
    let grad_y = reduce_broadcast(&grad_output.mul(&x), srcs[1].shape());
    vec![(srcs[0].ptr_id(), grad_x), (srcs[1].ptr_id(), grad_y)]
}

fn vjp_div(srcs: &[UOp], grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // z = x / y => dx = dz / y, dy = -dz * x / y^2
    let x = Tensor::from_uop(srcs[0].clone(), "CPU");
    let y = Tensor::from_uop(srcs[1].clone(), "CPU");
    let grad_x = reduce_broadcast(&grad_output.div(&y), srcs[0].shape());
    let grad_y = reduce_broadcast(&grad_output.neg().mul(&x).div(&y).div(&y), srcs[1].shape());
    vec![(srcs[0].ptr_id(), grad_x), (srcs[1].ptr_id(), grad_y)]
}

fn vjp_max(srcs: &[UOp], output_uop: &UOp, grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // z = max(x, y) => dx = dz * (x >= y), dy = dz * (y > x)
    // Use soft version: dx = dz * (x >= y), dy = dz * (1 - (x >= y)) for tied case
    let x = Tensor::from_uop(srcs[0].clone(), "CPU");
    let y = Tensor::from_uop(srcs[1].clone(), "CPU");
    let z = Tensor::from_uop(output_uop.clone(), "CPU");

    // mask_x = (x == z), cast to float
    let mask_x = x.eq(&z).cast(grad_output.dtype());
    let mask_y = y.eq(&z).cast(grad_output.dtype());

    // Normalize masks for tied case (both x and y equal z)
    let mask_sum = mask_x.add(&mask_y);
    let mask_x_norm = mask_x.div(&mask_sum);
    let mask_y_norm = mask_y.div(&mask_sum);

    let grad_x = reduce_broadcast(&grad_output.mul(&mask_x_norm), srcs[0].shape());
    let grad_y = reduce_broadcast(&grad_output.mul(&mask_y_norm), srcs[1].shape());
    vec![(srcs[0].ptr_id(), grad_x), (srcs[1].ptr_id(), grad_y)]
}

// ============ Reduce VJPs ============

fn vjp_sum(srcs: &[UOp], output_uop: &UOp, grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // sum reduces input to output shape, expand gradient back
    let input_shape = srcs[0].shape();

    // Determine the axes that were reduced
    let axes = match output_uop.arg() {
        Some(UOpArg::Axes(ax)) => ax.clone(),
        _ => (0..input_shape.rank()).collect(),
    };

    // Expand gradient back to input shape
    let grad = expand_for_reduce(grad_output, input_shape, &axes);
    vec![(srcs[0].ptr_id(), grad)]
}

fn vjp_reduce_max(srcs: &[UOp], output_uop: &UOp, grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // dx = dz * (input == output_expanded)
    let input = Tensor::from_uop(srcs[0].clone(), "CPU");
    let input_shape = srcs[0].shape();

    // Determine the axes that were reduced
    let axes = match output_uop.arg() {
        Some(UOpArg::Axes(ax)) => ax.clone(),
        _ => (0..input_shape.rank()).collect(),
    };

    // Expand output to input shape for comparison
    let output_expanded = expand_for_reduce(
        &Tensor::from_uop(output_uop.clone(), "CPU"),
        input_shape,
        &axes,
    );

    // Mask where input equals max
    let mask = input.eq(&output_expanded).cast(grad_output.dtype());

    // Count matches along reduced axes for normalization
    let mask_sum = expand_for_reduce(&mask.sum(Some(axes.clone()), false), input_shape, &axes);
    let mask_norm = mask.div(&mask_sum);

    // Expand gradient and multiply by normalized mask
    let grad_expanded = expand_for_reduce(grad_output, input_shape, &axes);
    let grad = grad_expanded.mul(&mask_norm);

    vec![(srcs[0].ptr_id(), grad)]
}

// ============ Movement VJPs ============

fn vjp_reshape(srcs: &[UOp], grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // Reshape gradient back to input shape
    let input_shape = srcs[0].shape().clone();
    vec![(srcs[0].ptr_id(), grad_output.reshape(input_shape))]
}

fn vjp_expand(srcs: &[UOp], grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // Reduce gradient to input shape
    let grad = reduce_broadcast(grad_output, srcs[0].shape());
    vec![(srcs[0].ptr_id(), grad)]
}

fn vjp_permute(srcs: &[UOp], output_uop: &UOp, grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // Apply inverse permutation
    let axes = match output_uop.arg() {
        Some(UOpArg::Axes(ax)) => ax.clone(),
        _ => panic!("Permute must have axes argument"),
    };

    let inverse_axes = invert_permutation(&axes);
    vec![(srcs[0].ptr_id(), grad_output.permute(inverse_axes))]
}

fn vjp_cast(srcs: &[UOp], grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // Cast gradient back to input dtype
    let input_dtype = srcs[0].dtype();
    vec![(srcs[0].ptr_id(), grad_output.cast(input_dtype))]
}

fn vjp_where(srcs: &[UOp], grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // where(cond, x, y) => dx = dz * cond, dy = dz * (1 - cond)
    // Note: condition itself doesn't get gradients (boolean)
    let cond = Tensor::from_uop(srcs[0].clone(), "CPU");
    let cond_float = cond.cast(grad_output.dtype());
    let one = Tensor::full(Shape::scalar(), 1.0f32);
    let not_cond = one.sub(&cond_float);

    let grad_x = reduce_broadcast(&grad_output.mul(&cond_float), srcs[1].shape());
    let grad_y = reduce_broadcast(&grad_output.mul(&not_cond), srcs[2].shape());

    vec![(srcs[1].ptr_id(), grad_x), (srcs[2].ptr_id(), grad_y)]
}

fn vjp_pad(srcs: &[UOp], output_uop: &UOp, grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // Pad adds zeros around the tensor, so gradient is just slicing back
    // For now, we'll implement a simplified version using reshaping if possible
    // TODO: Implement proper slice operation for more general pad gradients
    let _ = output_uop;
    let input_shape = srcs[0].shape();

    // If shapes match, pass through
    if grad_output.shape() == input_shape {
        return vec![(srcs[0].ptr_id(), grad_output.clone())];
    }

    // For now, sum over the padded dimensions to get back to input shape
    // This is a simplification - proper implementation would need a slice operation
    let grad = reduce_broadcast(grad_output, input_shape);
    vec![(srcs[0].ptr_id(), grad)]
}

fn vjp_shrink(srcs: &[UOp], output_uop: &UOp, grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // Shrink is like slicing - gradient needs to be expanded back with zeros
    // TODO: Implement proper pad operation for Shrink gradient
    let _ = output_uop;
    let input_shape = srcs[0].shape();

    // For now, if shapes match, pass through
    if grad_output.shape() == input_shape {
        return vec![(srcs[0].ptr_id(), grad_output.clone())];
    }

    // Simplified: expand gradient back to input shape
    // This is not fully correct for general shrink, needs padding
    let grad = grad_output.expand(input_shape.clone());
    vec![(srcs[0].ptr_id(), grad)]
}

fn vjp_stride(srcs: &[UOp], output_uop: &UOp, grad_output: &Tensor) -> Vec<(usize, Tensor)> {
    // Stride operation samples every Nth element
    // Gradient needs to place values back at strided positions
    // TODO: Implement proper scatter for Stride gradient
    let _ = output_uop;
    let input_shape = srcs[0].shape();

    // For now, if shapes match, pass through
    if grad_output.shape() == input_shape {
        return vec![(srcs[0].ptr_id(), grad_output.clone())];
    }

    // Simplified: expand gradient back to input shape
    // This is not fully correct for general stride
    let grad = grad_output.expand(input_shape.clone());
    vec![(srcs[0].ptr_id(), grad)]
}

// ============ Helper Functions ============

/// Reduces gradient from broadcast shape back to target shape.
///
/// When a tensor is broadcast during forward pass, the gradient needs to be
/// summed along the broadcast dimensions during backward pass.
fn reduce_broadcast(grad: &Tensor, target_shape: &Shape) -> Tensor {
    let grad_shape = grad.shape();

    // If shapes already match, return as-is
    if grad_shape == target_shape {
        return grad.clone();
    }

    let grad_dims = grad_shape.dims();
    let target_dims = target_shape.dims();

    // Handle rank difference: sum over leading dimensions
    let rank_diff = grad_dims.len().saturating_sub(target_dims.len());
    let mut axes_to_reduce: Vec<usize> = (0..rank_diff).collect();

    // Find dimensions that were broadcast (size 1 in target, larger in grad)
    for (i, (&grad_dim, &target_dim)) in grad_dims[rank_diff..]
        .iter()
        .zip(target_dims.iter())
        .enumerate()
    {
        if target_dim == 1 && grad_dim > 1 {
            axes_to_reduce.push(rank_diff + i);
        }
    }

    if axes_to_reduce.is_empty() {
        return grad.clone();
    }

    // Sum along broadcast dimensions with keepdims to maintain shape compatibility
    let reduced = grad.sum(Some(axes_to_reduce), true);

    // Reshape to target shape
    reduced.reshape(target_shape.clone())
}

/// Expands gradient for reduce operations back to input shape.
fn expand_for_reduce(grad: &Tensor, input_shape: &Shape, axes: &[usize]) -> Tensor {
    // Build the shape after keepdims=true would produce
    let mut expanded_dims = input_shape.dims().to_vec();
    for &ax in axes {
        expanded_dims[ax] = 1;
    }

    // Reshape gradient to this intermediate shape (with 1s in reduced dimensions)
    let reshaped = if grad.shape().dims() == expanded_dims.as_slice() {
        grad.clone()
    } else if grad.shape().is_scalar() {
        grad.reshape(Shape::new(expanded_dims.clone()))
    } else {
        // Need to insert dimensions for reduced axes
        let mut new_shape = Vec::new();
        let mut grad_idx = 0;
        let grad_dims = grad.shape().dims();
        for (i, &dim) in input_shape.dims().iter().enumerate() {
            if axes.contains(&i) {
                new_shape.push(1);
            } else if grad_idx < grad_dims.len() {
                new_shape.push(grad_dims[grad_idx]);
                grad_idx += 1;
            } else {
                new_shape.push(dim);
            }
        }
        grad.reshape(Shape::new(new_shape))
    };

    // Expand to full input shape
    reshaped.expand(input_shape.clone())
}

/// Computes the inverse of a permutation.
fn invert_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inverse[p] = i;
    }
    inverse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invert_permutation() {
        assert_eq!(invert_permutation(&[0, 1, 2]), vec![0, 1, 2]);
        assert_eq!(invert_permutation(&[2, 0, 1]), vec![1, 2, 0]);
        assert_eq!(invert_permutation(&[1, 0]), vec![1, 0]);
    }
}
