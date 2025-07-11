use crate::op::*;
use crate::renderer::{Render, Renderer};
use std::any::Any;

/// A renderer for generating C code from a computation graph.
pub struct CRenderer;

impl Renderer for CRenderer {
    fn render_op(&self, op: &dyn Operator, operands: &[String]) -> Option<String> {
        let op_any = op.as_any();
        match op_any {
            _ if op_any.is::<OpAdd>() => Some(self.render(op_any.downcast_ref::<OpAdd>().unwrap(), operands)),
            _ if op_any.is::<OpMul>() => Some(self.render(op_any.downcast_ref::<OpMul>().unwrap(), operands)),
            _ if op_any.is::<Sin>() => Some(self.render(op_any.downcast_ref::<Sin>().unwrap(), operands)),
            _ if op_any.is::<Const>() => Some(self.render(op_any.downcast_ref::<Const>().unwrap(), operands)),
            _ if op_any.is::<Variable>() => Some(self.render(op_any.downcast_ref::<Variable>().unwrap(), operands)),
            _ => None, // Operator not supported by this renderer
        }
    }
}

impl Render<OpAdd> for CRenderer {
    fn render(&self, _op: &OpAdd, operands: &[String]) -> String {
        format!("({} + {})", operands[0], operands[1])
    }
}

impl Render<OpMul> for CRenderer {
    fn render(&self, _op: &OpMul, operands: &[String]) -> String {
        format!("({} * {})", operands[0], operands[1])
    }
}

impl Render<Sin> for CRenderer {
    fn render(&self, _op: &Sin, operands: &[String]) -> String {
        format!("sin({})", operands[0])
    }
}

impl Render<Const> for CRenderer {
    fn render(&self, op: &Const, _operands: &[String]) -> String {
        format!("{:?}", op.0)
    }
}

impl Render<Variable> for CRenderer {
    fn render(&self, op: &Variable, _operands: &[String]) -> String {
        op.0.clone()
    }
}
