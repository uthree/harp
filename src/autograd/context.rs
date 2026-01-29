//! Gradient context for managing gradients during backpropagation.

use std::collections::HashMap;

use crate::Tensor;

/// Context for storing and managing gradients during backpropagation.
///
/// This struct holds the accumulated gradients for each tensor that
/// requires gradients.
pub struct GradientContext {
    /// Mapping from tensor pointer ID to accumulated gradient.
    gradients: HashMap<usize, Tensor>,
}

impl GradientContext {
    /// Creates a new empty gradient context.
    pub fn new() -> Self {
        GradientContext {
            gradients: HashMap::new(),
        }
    }

    /// Gets the gradient for a tensor by its pointer ID.
    pub fn get_by_id(&self, ptr_id: usize) -> Option<&Tensor> {
        self.gradients.get(&ptr_id)
    }

    /// Gets the gradient for a tensor.
    pub fn get(&self, tensor: &Tensor) -> Option<&Tensor> {
        self.get_by_id(tensor.uop().ptr_id())
    }

    /// Accumulates gradient for a tensor (adds to existing gradient if present).
    pub fn accumulate(&mut self, ptr_id: usize, grad: Tensor) {
        if let Some(existing) = self.gradients.get(&ptr_id) {
            self.gradients.insert(ptr_id, existing.add(&grad));
        } else {
            self.gradients.insert(ptr_id, grad);
        }
    }

    /// Sets the gradient for a tensor (replaces any existing gradient).
    pub fn set(&mut self, ptr_id: usize, grad: Tensor) {
        self.gradients.insert(ptr_id, grad);
    }

    /// Returns the number of gradients stored.
    pub fn len(&self) -> usize {
        self.gradients.len()
    }

    /// Returns true if no gradients are stored.
    pub fn is_empty(&self) -> bool {
        self.gradients.is_empty()
    }

    /// Returns an iterator over (ptr_id, gradient) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &Tensor)> {
        self.gradients.iter()
    }
}

impl Default for GradientContext {
    fn default() -> Self {
        Self::new()
    }
}
