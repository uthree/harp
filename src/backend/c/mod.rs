use crate::op::*;
use crate::backend::renderer::{Render, Renderer};

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
            _ if op_any.is::<LoopVariable>() => Some(self.render(op_any.downcast_ref::<LoopVariable>().unwrap(), operands)),
            // Loop operator is handled by the CodeGenerator itself.
            _ if op_any.is::<Loop>() => None,
            _ => None, // Operator not supported by this renderer
        }
    }
}

impl Render<LoopVariable> for CRenderer {
    fn render(&self, _op: &LoopVariable, _operands: &[String]) -> String {
        "i".to_string() // A default loop variable name. This could be made more robust.
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
