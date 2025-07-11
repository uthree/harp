use crate::op::Operator;

/// A trait for rendering a specific operator `O` into a string representation.
///
/// This trait is implemented by a `Renderer` for each primitive operator it supports.
pub trait Render<O: Operator> {
    /// Renders the given operator.
    ///
    /// # Arguments
    ///
    /// * `op` - The operator to render.
    /// * `operands` - A slice of strings, where each string is the already-rendered
    ///   code for an operand of this operator.
    ///
    /// # Returns
    ///
    /// A string representing the rendered code for this operation.
    fn render(&self, op: &O, operands: &[String]) -> String;
}

/// A trait for a code generation backend.
///
/// A `Renderer` is responsible for dispatching to the correct `Render<O>`
/// implementation for a given dynamic `Operator` trait object. It also handles
/// the fallback logic for `FusedOp` operators.
pub trait Renderer {
    /// Renders a dynamic operator trait object.
    ///
    /// This method should handle downcasting the `&dyn Operator` to a concrete
    /// type and calling the appropriate `Render<O>` implementation. If the
    /// operator is not supported, it should return `None`.
    fn render_op(&self, op: &dyn Operator, operands: &[String]) -> Option<String>;
}
