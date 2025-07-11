use crate::backend::codegen::Instruction;
use crate::op::Operator;

/// A trait for rendering a specific operator `O` into a string representation (an expression).
pub trait Render<O: Operator> {
    fn render(&self, op: &O, operands: &[String]) -> String;
}

/// A trait for a code generation backend.
///
/// A `Renderer` is responsible for two things:
/// 1. Rendering individual operators into expression strings (`render_op`).
/// 2. Rendering a complete list of abstract `Instruction`s into a final function string (`render_function`).
pub trait Renderer {
    /// Renders a dynamic operator trait object into an expression string.
    ///
    /// If the operator is not supported for expression-level rendering (e.g., it's a
    /// statement-level operator like `Store`), it should return `None`.
    fn render_op(&self, op: &dyn Operator, operands: &[String]) -> Option<String>;

    /// Renders a complete function from a list of abstract `Instruction`s.
    fn render_function(
        &self,
        fn_name: &str,
        args: &[(&str, &str)],
        body: &[Instruction],
        return_type: &str,
    ) -> String;
}
