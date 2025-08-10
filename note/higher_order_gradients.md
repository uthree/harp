# Higher-Order Gradients and Functional `grad()` API

This document outlines a potential future improvement for the automatic differentiation engine in `harp` to cleanly support higher-order gradients (grad of a grad).

## Current `backward()` Implementation

The current `backward()` method is implemented with side effects. It computes the gradients of all tensors in the graph with respect to the tensor it's called on, and stores these gradients directly in the `.grad` field of each respective tensor.

```rust
// Example
let x = Tensor::ones(..., true);
let y = x.clone() * x.clone(); // y = x^2
y.backward();
// Now, x.grad contains the gradient of y with respect to x.
```

This is a common and intuitive API for many use cases, especially for training neural networks where the primary goal is to get the gradients of the model parameters with respect to a loss function.

## The Challenge with Higher-Order Gradients

While the current architecture (where gradients are also computation graphs) can technically compute higher-order derivatives, the `backward()` API makes it awkward. Calling `x.grad.unwrap().backward()` would attempt to compute the gradient of the gradient, but it would also overwrite the `.grad` fields of the tensors involved, which is not always the desired or clear behavior.

## Proposed Solution: A Functional `grad()` API

To provide a cleaner, more robust, and functionally pure way to handle gradients, we can introduce a `grad()` function.

### Signature

```rust
pub fn grad(output: &Tensor, inputs: &[Tensor]) -> Vec<Tensor>
```

### Behavior

- **No Side Effects**: This function would **not** modify the `.grad` field of any tensor.
- **Returns New Tensors**: It returns a `Vec<Tensor>`, where each tensor in the vector represents the gradient of the `output` with respect to the corresponding tensor in the `inputs` slice.
- **Lazy Evaluation**: The returned gradient tensors are, like all other tensors, just computation graphs. Their concrete values are only computed when their `.forward()` method is called.

### Implementation Sketch

The internal logic of `grad()` would be very similar to the current `backward()` implementation:
1. Perform a topological sort of the computation graph starting from `output`.
2. Use a `HashMap` to compute and accumulate gradients for each node in the graph, starting with the gradient of `output` with respect to itself being `1.0`.
3. Instead of writing the final gradients to the `.grad` fields, it would collect the gradients corresponding to the `inputs` tensors and return them.

### Benefits

With this API, calculating higher-order gradients becomes natural and unambiguous:

```rust
// y = a^3
let a = Tensor::full(..., 2.0, true);
let y = a.clone() * a.clone() * a.clone();

// First-order gradient (dy/da = 3a^2)
let dy_da = grad(&y, &[a.clone()])[0].clone();

// Second-order gradient (d/da(dy/da) = 6a)
let d2y_da2 = grad(&dy_da, &[a.clone()])[0].clone();

// Third-order gradient (d/da(d2y_da2) = 6)
let d3y_da3 = grad(&d2y_da2, &[a.clone()])[0].clone();

// We can now execute the forward pass on these gradient tensors to get concrete values
d3y_da3.forward();
// The buffer of d3y_da3 should now contain the value 6.0.
```

### Conclusion

Introducing a functional `grad()` API would be a powerful addition to `harp`. It would make the automatic differentiation system more robust, flexible, and philosophically aligned with the principles of functional programming, while the existing `backward()` method can be kept as a convenient wrapper for common use cases. This is a valuable direction for future development.
