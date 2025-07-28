# 2025-07-28: Feature Enhancements and Refactoring

This document summarizes the series of feature additions, API improvements, and refactoring work done on this date.

## 1. Generic `Scan` (Cumulative) Operation

To support cumulative operations like `cumsum` and `cumprod` in a generalized way, a `Scan` operation was introduced, mirroring the existing `Reduce` operation.

- **`TensorOp::Scan`**: A new `TensorOp` variant, `Scan { axis: usize, op: Op }`, was added. It accepts an axis and any binary `Op` (e.g., `Add`, `Mul`).
- **`Tensor::scan()`**: A corresponding `scan()` method was added to `Tensor`, along with convenient wrappers `cumsum()` and `cumprod()`.
- **Lowering**: The `Lowerizer` was updated to handle `TensorOp::Scan`. The logic is similar to `Reduce`, but it stores the intermediate accumulator value to the output buffer inside the loop.
- **Identity Element**: An `identity_element()` method was added to `Op` to provide the correct initial value for the accumulator (e.g., 0 for `Add`, 1 for `Mul`).

## 2. `Tensor::full` for Arbitrary Value Initialization

Inspired by `torch.full`, a new constructor `Tensor::full` was implemented to create a tensor filled with a specified value.

- **Generic API**: The method signature is `full<T: Into<Number>>(shape: Vec<usize>, fill_value: T)`, allowing for intuitive calls like `Tensor::full(shape, 7.0f32)`.
- **Implementation**: It leverages `TensorOp::Constant` to represent the operation lazily without immediate memory allocation.

## 3. Tensor-Level Optimization Framework

A major feature was the introduction of a pattern matching optimization framework that operates directly on the `Tensor` graph, before lowering to `UOp`s. This allows for more powerful, context-aware optimizations.

- **`TPat` Enum**: A `TPat` (Tensor Pattern) enum was created to define structural patterns in the `Tensor` graph, analogous to `UPat` for `UOp`s.
- **`TensorPatternMatcher`**: This struct manages and applies a set of optimization rules. It traverses the graph in post-order, applying rules from the leaves up to the root.
- **Conditional Rules**: Rules (`TPatRule`) were designed to include a `condition` closure. This allows matching not just on the graph structure (`TPat`) but also on arbitrary properties of the captured tensors, such as a constant's value or a tensor's shape. This elegantly solves the problem of matching "View" operations like `Reshape` that only modify the `ShapeTracker`.

## 4. `tpat!` Macro

To simplify the creation of `Tensor` optimization rules, a declarative macro `tpat!` was implemented.

- **Syntax**: It provides a concise syntax for defining a rule's name, capture variables, pattern, optional condition, and replacer.

  ```rust
  tpat!({
      "rule_name": (x, c) | TPat::Binary(Op::Mul, x, c), if {
          // condition on captured tensors `x` and `c`
      } => /* replacer expression using `x` and `c` */,
  })
  ```

## 5. Code Refactoring: `optimization` Module

To improve code organization, several modules related to performance optimization were consolidated into a new `src/optimization` directory.

- **Moved Modules**: `autotuner.rs`, `optimizer.rs`, `pattern.rs`, and `linearizer.rs`.
- **Re-exports**: The new `optimization` module re-exports the most commonly used items from its submodules, simplifying `use` statements throughout the codebase (e.g., `use harp::optimization::Optimizer`).

## 6. Known Issue: `log` and `exp` Implementation

An attempt was made to implement `Tensor::log` (natural logarithm) and `Tensor::exp` (natural exponential) by using the existing `log2` and `exp2` operations combined with the change of base formula.

- **Problem**: The implementation currently fails its tests. The root cause is suspected to be a bug in the `Lowerizer` when handling `TensorOp::Load`. It appears to incorrectly generate access to `shape_args` even for simple element-wise operations on loaded tensors, leading to incorrect calculations.
- **Status**: The feature has been temporarily shelved, and the implementation was reverted. Fixing the underlying `Lowerizer` bug is required before this can be completed.
