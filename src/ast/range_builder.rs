use super::{AstNode, ConstLiteral};

/// Builder for Range nodes with default values
pub struct RangeBuilder {
    counter_name: String,
    start: Box<AstNode>,
    max: Box<AstNode>,
    step: Box<AstNode>,
    body: Box<AstNode>,
    unroll: Option<usize>,
}

impl RangeBuilder {
    /// Create a new RangeBuilder with required fields and default start=0, step=1, unroll=None
    pub fn new(
        counter_name: impl Into<String>,
        max: impl Into<AstNode>,
        body: impl Into<AstNode>,
    ) -> Self {
        Self {
            counter_name: counter_name.into(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(max.into()),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(body.into()),
            unroll: None,
        }
    }

    pub fn start(mut self, start: impl Into<AstNode>) -> Self {
        self.start = Box::new(start.into());
        self
    }

    pub fn step(mut self, step: impl Into<AstNode>) -> Self {
        self.step = Box::new(step.into());
        self
    }

    /// Enable full unrolling (#pragma unroll)
    pub fn unroll(mut self) -> Self {
        self.unroll = Some(0);
        self
    }

    /// Enable unrolling with a specific factor (#pragma unroll N)
    pub fn unroll_by(mut self, factor: usize) -> Self {
        self.unroll = Some(factor);
        self
    }

    pub fn build(self) -> AstNode {
        AstNode::Range {
            counter_name: self.counter_name,
            start: self.start,
            max: self.max,
            step: self.step,
            body: self.body,
            unroll: self.unroll,
        }
    }
}
