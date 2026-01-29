//! Automatic differentiation module.
//!
//! This module provides reverse-mode automatic differentiation (backpropagation)
//! for computing gradients of tensor computations.

mod backward;
mod context;
mod vjp;

pub use backward::backward;
pub use context::GradientContext;

use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Returns whether gradient computation is currently enabled.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// Sets whether gradient computation is enabled.
fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED.with(|g| g.set(enabled));
}

/// RAII guard that temporarily disables gradient computation.
///
/// When dropped, restores the previous gradient computation state.
///
/// # Example
///
/// ```ignore
/// use eclat::autograd::NoGradGuard;
///
/// // Gradients are enabled by default
/// assert!(eclat::autograd::is_grad_enabled());
///
/// {
///     let _guard = NoGradGuard::new();
///     // Gradients are disabled in this scope
///     assert!(!eclat::autograd::is_grad_enabled());
/// }
///
/// // Gradients are re-enabled
/// assert!(eclat::autograd::is_grad_enabled());
/// ```
pub struct NoGradGuard {
    prev_enabled: bool,
}

impl NoGradGuard {
    /// Creates a new NoGradGuard, disabling gradient computation.
    pub fn new() -> Self {
        let prev_enabled = is_grad_enabled();
        set_grad_enabled(false);
        NoGradGuard { prev_enabled }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev_enabled);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_grad_guard() {
        assert!(is_grad_enabled());

        {
            let _guard = NoGradGuard::new();
            assert!(!is_grad_enabled());

            {
                let _guard2 = NoGradGuard::new();
                assert!(!is_grad_enabled());
            }

            // Still disabled because outer guard is active
            assert!(!is_grad_enabled());
        }

        assert!(is_grad_enabled());
    }
}
