use crate::graph::shape::Expr as ShapeExpr;
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

    /// Tangent: tan(x) = sin(x) / cos(x)
    pub fn tan(self) -> Self {
        // tan(x) = sin(x) / cos(x)
        self.clone().sin() / self.cos()
    }

    /// 1D Convolution operation
    ///
    /// # Arguments
    /// * `kernel` - Kernel weights with shape [out_channels, in_channels, kernel_size]
    /// * `stride` - Stride for the convolution (default: 1)
    ///
    /// # Input shape
    /// [batch, in_channels, length]
    ///
    /// # Output shape
    /// [batch, out_channels, output_length] where output_length = (length - kernel_size) / stride + 1
    ///
    /// # Example
    /// ```ignore
    /// let input = graph.input(DType::F32, vec![1.into(), 3.into(), 32.into()]);
    /// let kernel = graph.input(DType::F32, vec![16.into(), 3.into(), 5.into()]);
    /// let output = input.conv1d(kernel, 1); // [1, 16, 28]
    /// ```
    pub fn conv1d<E: Into<ShapeExpr> + Clone>(self, kernel: GraphNode, stride: E) -> GraphNode {
        use super::ReduceOps;

        // Get shapes first (clone to avoid borrowing issues)
        let input_shape = self.view.shape().to_vec();
        let kernel_shape = kernel.view.shape().to_vec();

        let batch = input_shape[0].clone();
        let in_channels = input_shape[1].clone();
        let out_channels = kernel_shape[0].clone();
        let kernel_size = kernel_shape[2].clone();

        // Apply sliding window: [B, C_in, L] -> [B, C_in, L', K]
        let windowed = self.sliding_window(2, kernel_size.clone(), stride.into());
        let windowed_shape = windowed.view.shape().to_vec();
        let output_len = windowed_shape[2].clone();

        // Reshape for broadcasting
        // windowed: [B, C_in, L', K] -> [B, 1, C_in, L', K]
        let windowed_expanded = windowed.unsqueeze(1);

        // kernel: [C_out, C_in, K] -> [1, C_out, C_in, 1, K]
        let kernel_expanded = kernel.unsqueeze(0).unsqueeze(3);

        // Expand to match shapes
        let windowed_broadcast = windowed_expanded.expand(vec![
            batch.clone(),
            out_channels.clone(),
            in_channels.clone(),
            output_len.clone(),
            kernel_size.clone(),
        ]);
        let kernel_broadcast = kernel_expanded.expand(vec![
            batch,
            out_channels,
            in_channels,
            output_len,
            kernel_size,
        ]);

        // Element-wise multiplication and sum reduction
        let product = windowed_broadcast * kernel_broadcast;
        // Sum over [in_channels, kernel_size] at axes [2, 4]
        product.sum(2).sum(3)
    }

    /// 2D Convolution operation
    ///
    /// # Arguments
    /// * `kernel` - Kernel weights with shape [out_channels, in_channels, kernel_h, kernel_w]
    /// * `stride` - Stride for the convolution (default: 1)
    ///
    /// # Input shape
    /// [batch, in_channels, height, width]
    ///
    /// # Output shape
    /// [batch, out_channels, out_h, out_w] where:
    /// - out_h = (height - kernel_h) / stride + 1
    /// - out_w = (width - kernel_w) / stride + 1
    ///
    /// # Example
    /// ```ignore
    /// let input = graph.input(DType::F32, vec![1.into(), 3.into(), 224.into(), 224.into()]);
    /// let kernel = graph.input(DType::F32, vec![64.into(), 3.into(), 7.into(), 7.into()]);
    /// let output = input.conv2d(kernel, 2); // [1, 64, 109, 109]
    /// ```
    pub fn conv2d<E: Into<ShapeExpr> + Clone>(self, kernel: GraphNode, stride: E) -> GraphNode {
        use super::ReduceOps;

        // Get shapes first (clone to avoid borrowing issues)
        let input_shape = self.view.shape().to_vec();
        let kernel_shape = kernel.view.shape().to_vec();

        let batch = input_shape[0].clone();
        let in_channels = input_shape[1].clone();
        let out_channels = kernel_shape[0].clone();
        let kernel_h = kernel_shape[2].clone();
        let kernel_w = kernel_shape[3].clone();

        // Apply sliding window on height and width
        // [B, C_in, H, W] -> [B, C_in, H', W, Kh]
        let stride_expr = stride.into();
        let windowed_h = self.sliding_window(2, kernel_h.clone(), stride_expr.clone());
        // [B, C_in, H', W, Kh] -> [B, C_in, H', W', Kh, Kw]
        let windowed_hw = windowed_h.sliding_window(3, kernel_w.clone(), stride_expr);
        let windowed_shape = windowed_hw.view.shape().to_vec();
        let out_h = windowed_shape[2].clone();
        let out_w = windowed_shape[3].clone();

        // Reshape for broadcasting
        // windowed: [B, C_in, H', W', Kh, Kw] -> [B, 1, C_in, H', W', Kh, Kw]
        let windowed_expanded = windowed_hw.unsqueeze(1);

        // kernel: [C_out, C_in, Kh, Kw] -> [1, C_out, C_in, 1, 1, Kh, Kw]
        let kernel_expanded = kernel.unsqueeze(0).unsqueeze(3).unsqueeze(4);

        // Expand to match shapes
        let windowed_broadcast = windowed_expanded.expand(vec![
            batch.clone(),
            out_channels.clone(),
            in_channels.clone(),
            out_h.clone(),
            out_w.clone(),
            kernel_h.clone(),
            kernel_w.clone(),
        ]);
        let kernel_broadcast = kernel_expanded.expand(vec![
            batch,
            out_channels,
            in_channels,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
        ]);

        // Element-wise multiplication and sum reduction
        let product = windowed_broadcast * kernel_broadcast;
        // Sum over [in_channels, Kh, Kw] at axes [2, 5, 6]
        product.sum(2).sum(4).sum(4)
    }

    /// 3D Convolution operation
    ///
    /// # Arguments
    /// * `kernel` - Kernel weights with shape [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
    /// * `stride` - Stride for the convolution (default: 1)
    ///
    /// # Input shape
    /// [batch, in_channels, depth, height, width]
    ///
    /// # Output shape
    /// [batch, out_channels, out_d, out_h, out_w] where:
    /// - out_d = (depth - kernel_d) / stride + 1
    /// - out_h = (height - kernel_h) / stride + 1
    /// - out_w = (width - kernel_w) / stride + 1
    ///
    /// # Example
    /// ```ignore
    /// let input = graph.input(DType::F32, vec![1.into(), 3.into(), 16.into(), 112.into(), 112.into()]);
    /// let kernel = graph.input(DType::F32, vec![64.into(), 3.into(), 3.into(), 7.into(), 7.into()]);
    /// let output = input.conv3d(kernel, 2); // [1, 64, 7, 53, 53]
    /// ```
    pub fn conv3d<E: Into<ShapeExpr> + Clone>(self, kernel: GraphNode, stride: E) -> GraphNode {
        use super::ReduceOps;

        // Get shapes first (clone to avoid borrowing issues)
        let input_shape = self.view.shape().to_vec();
        let kernel_shape = kernel.view.shape().to_vec();

        let batch = input_shape[0].clone();
        let in_channels = input_shape[1].clone();
        let out_channels = kernel_shape[0].clone();
        let kernel_d = kernel_shape[2].clone();
        let kernel_h = kernel_shape[3].clone();
        let kernel_w = kernel_shape[4].clone();

        // Apply sliding window on depth, height, and width
        // [B, C_in, D, H, W] -> [B, C_in, D', H, W, Kd]
        let stride_expr = stride.into();
        let windowed_d = self.sliding_window(2, kernel_d.clone(), stride_expr.clone());
        // [B, C_in, D', H, W, Kd] -> [B, C_in, D', H', W, Kd, Kh]
        let windowed_dh = windowed_d.sliding_window(3, kernel_h.clone(), stride_expr.clone());
        // [B, C_in, D', H', W, Kd, Kh] -> [B, C_in, D', H', W', Kd, Kh, Kw]
        let windowed_dhw = windowed_dh.sliding_window(4, kernel_w.clone(), stride_expr);
        let windowed_shape = windowed_dhw.view.shape().to_vec();
        let out_d = windowed_shape[2].clone();
        let out_h = windowed_shape[3].clone();
        let out_w = windowed_shape[4].clone();

        // Reshape for broadcasting
        // windowed: [B, C_in, D', H', W', Kd, Kh, Kw] -> [B, 1, C_in, D', H', W', Kd, Kh, Kw]
        let windowed_expanded = windowed_dhw.unsqueeze(1);

        // kernel: [C_out, C_in, Kd, Kh, Kw] -> [1, C_out, C_in, 1, 1, 1, Kd, Kh, Kw]
        let kernel_expanded = kernel.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5);

        // Expand to match shapes
        let windowed_broadcast = windowed_expanded.expand(vec![
            batch.clone(),
            out_channels.clone(),
            in_channels.clone(),
            out_d.clone(),
            out_h.clone(),
            out_w.clone(),
            kernel_d.clone(),
            kernel_h.clone(),
            kernel_w.clone(),
        ]);
        let kernel_broadcast = kernel_expanded.expand(vec![
            batch,
            out_channels,
            in_channels,
            out_d,
            out_h,
            out_w,
            kernel_d,
            kernel_h,
            kernel_w,
        ]);

        // Element-wise multiplication and sum reduction
        let product = windowed_broadcast * kernel_broadcast;
        // Sum over [in_channels, Kd, Kh, Kw] at axes [2, 6, 7, 8]
        product.sum(2).sum(5).sum(5).sum(5)
    }
}
