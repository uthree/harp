use crate::graph::GraphNode;

impl std::ops::Sub for GraphNode {
    type Output = GraphNode;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Div for GraphNode {
    type Output = GraphNode;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.recip()
    }
}

impl GraphNode {
    /// Natural logarithm: log(x) = log2(x) / log2(e)
    pub fn log(self) -> Self {
        // log(x) = log2(x) / log2(e)
        // 1 / log2(e) = ln(2) ≈ 0.6931471805599453
        let ln_2 = std::f32::consts::LN_2;
        self.log2() * GraphNode::f32(ln_2)
    }

    /// Natural exponential: exp(x) = exp2(x * log2(e))
    pub fn exp(self) -> Self {
        // exp(x) = exp2(x * log2(e))
        // log2(e) = 1 / ln(2) ≈ 1.4426950408889634
        let log2_e = 1.0f32 / std::f32::consts::LN_2;
        (self * GraphNode::f32(log2_e)).exp2()
    }

    /// Cosine: cos(x) = sin(x + π/2)
    pub fn cos(self) -> Self {
        // cos(x) = sin(x + π/2)
        let pi_over_2 = std::f32::consts::FRAC_PI_2;
        (self + GraphNode::f32(pi_over_2)).sin()
    }
}
