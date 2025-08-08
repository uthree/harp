use crate::graph::{
    context::Graph,
    node::NodeId,
    op::GraphOp,
    shape::tracker::ShapeTracker,
    ops::{ElementwiseOps, ReduceOps, ShapeOps},
};

pub trait ConvolutionOps {
    fn unfold1d(&self, src: NodeId, dim: usize, kernel_size: usize, stride: usize) -> NodeId;
    fn unfold2d(
        &self,
        src: NodeId,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> NodeId;
    fn conv1d(
        &self,
        input: NodeId,
        weight: NodeId,
        kernel_size: usize,
        stride: usize,
        groups: usize,
    ) -> NodeId;
    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &self,
        input: NodeId,
        weight: NodeId,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        groups: usize,
    ) -> NodeId;
}

impl ConvolutionOps for Graph {
    fn unfold1d(&self, src: NodeId, dim: usize, kernel_size: usize, stride: usize) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = ShapeTracker::new(shape);
        let new_shape = tracker.unfold1d(dim, kernel_size, stride).shape().to_vec();
        self.add_node(
            GraphOp::Unfold1d {
                dim,
                kernel_size,
                stride,
            },
            vec![src],
            dtype,
            new_shape,
        )
    }

    fn unfold2d(
        &self,
        src: NodeId,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = ShapeTracker::new(shape);
        // For conv2d, unfold is typically on the last two dimensions (H and W)
        let h_dim = tracker.ndim() - 2;
        let w_dim = tracker.ndim() - 1;
        let unfolded_tracker = tracker.unfold2d(h_dim, w_dim, kernel_size, stride);

        // After unfolding, the shape is [..., H_out, KH, W_out, KW, ...]
        // We want [..., H_out, W_out, KH, KW, ...]
        // So we need to swap the KH and W_out axes.
        let mut axes: Vec<_> = (0..unfolded_tracker.ndim()).collect();
        axes.swap(h_dim + 1, h_dim + 2);
        let permuted_shape = unfolded_tracker
            .clone()
            .permute(axes.clone())
            .shape()
            .to_vec();

        let unfolded_node = self.add_node(
            GraphOp::Unfold2d {
                kernel_size,
                stride,
            },
            vec![src],
            dtype.clone(),
            unfolded_tracker.shape().to_vec(),
        );

        self.add_node(
            GraphOp::Permute(axes),
            vec![unfolded_node],
            dtype,
            permuted_shape,
        )
    }

    fn conv1d(
        &self,
        input: NodeId,
        weight: NodeId,
        kernel_size: usize,
        stride: usize,
        groups: usize,
    ) -> NodeId {
        // Get input and weight shapes
        let (n, c_in, l, c_out, k) = {
            let nodes = self.nodes.borrow();
            let in_node = &nodes[input.0];
            let w_node = &nodes[weight.0];
            // input: [N, C_in, L]
            // weight: [C_out, C_in/G, K]
            let n = in_node.shape[0].clone();
            let c_in = in_node.shape[1].clone();
            let l = in_node.shape[2].clone();
            let c_out = w_node.shape[0].clone();
            let k = w_node.shape[2].clone();
            (n, c_in, l, c_out, k)
        };
        let c_in_per_group = (c_in.clone() / groups).simplify();
        let c_out_per_group = (c_out.clone() / groups).simplify();
        let l_out = ((l.clone() - k.clone()) / stride + 1).simplify();

        // 1. Reshape input and weight
        // input: [N, C_in, L] -> [N, G, C_in/G, L]
        let x_reshaped = self.reshape(
            input,
            vec![n.clone(), groups.into(), c_in_per_group.clone(), l.clone()],
        );
        // weight: [C_out, C_in/G, K] -> [G, C_out/G, C_in/G, K]
        let w_reshaped = self.reshape(
            weight,
            vec![
                groups.into(),
                c_out_per_group.clone(),
                c_in_per_group.clone(),
                k.clone(),
            ],
        );

        // 2. Unfold input
        // x_reshaped: [N, G, C_in/G, L] -> [N, G, C_in/G, L_out, K]
        let x_unfolded = self.unfold1d(x_reshaped, 3, kernel_size, stride);

        // 3. Reshape for broadcast
        // x_unfolded: [N, G, C_in/G, L_out, K] -> [N, G, 1, C_in/G, L_out, K]
        let x_broadcastable = self.unsqueeze(x_unfolded, 2);
        // w_reshaped: [G, C_out/G, C_in/G, K] -> [1, G, C_out/G, C_in/G, 1, K]
        let w_broadcastable_1 = self.unsqueeze(w_reshaped, 0);
        let w_broadcastable = self.unsqueeze(w_broadcastable_1, 4);

        // 4. Expand
        let broadcast_shape = vec![
            n.clone(),
            groups.into(),
            c_out_per_group.clone(),
            c_in_per_group.clone(),
            l_out.clone(),
            k.clone(),
        ];
        let x_expanded = self.expand(x_broadcastable, broadcast_shape.clone());
        let w_expanded = self.expand(w_broadcastable, broadcast_shape);

        // 5. Multiply
        let mul_result = self.mul(x_expanded, w_expanded);

        // 6. Sum reduction
        // sum over K (axis=5) -> [N, G, C_out/G, C_in/G, L_out]
        let sum_k = self.sum(mul_result, 5);
        // sum over C_in/G (axis=3) -> [N, G, C_out/G, L_out]
        let sum_c_in = self.sum(sum_k, 3);

        // 7. Reshape output
        // [N, G, C_out/G, L_out] -> [N, C_out, L_out]
        self.reshape(sum_c_in, vec![n, c_out, l_out])
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &self,
        input: NodeId,
        weight: NodeId,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        groups: usize,
    ) -> NodeId {
        // Get input and weight shapes
        let (n, c_in, h, w, c_out, kh, kw) = {
            let nodes = self.nodes.borrow();
            let in_node = &nodes[input.0];
            let w_node = &nodes[weight.0];
            // input: [N, C_in, H, W]
            // weight: [C_out, C_in/G, KH, KW]
            let n = in_node.shape[0].clone();
            let c_in = in_node.shape[1].clone();
            let h = in_node.shape[2].clone();
            let w = in_node.shape[3].clone();
            let c_out = w_node.shape[0].clone();
            let kh = w_node.shape[2].clone();
            let kw = w_node.shape[3].clone();
            (n, c_in, h, w, c_out, kh, kw)
        };
        let c_in_per_group = (c_in.clone() / groups).simplify();
        let c_out_per_group = (c_out.clone() / groups).simplify();
        let h_out = ((h.clone() - kh.clone()) / stride.0 + 1).simplify();
        let w_out = ((w.clone() - kw.clone()) / stride.1 + 1).simplify();

        // 1. Reshape input and weight
        // input: [N, C_in, H, W] -> [N, G, C_in/G, H, W]
        let x_reshaped = self.reshape(
            input,
            vec![
                n.clone(),
                groups.into(),
                c_in_per_group.clone(),
                h.clone(),
                w.clone(),
            ],
        );
        // weight: [C_out, C_in/G, KH, KW] -> [G, C_out/G, C_in/G, KH, KW]
        let w_reshaped = self.reshape(
            weight,
            vec![
                groups.into(),
                c_out_per_group.clone(),
                c_in_per_group.clone(),
                kh.clone(),
                kw.clone(),
            ],
        );

        // 2. Unfold input
        // x_reshaped: [N, G, C_in/G, H, W] -> [N, G, C_in/G, H_out, W_out, KH, KW]
        let x_unfolded = self.unfold2d(x_reshaped, kernel_size, stride);

        // 3. Reshape for broadcast
        // x_unfolded: [N, G, C_in/G, H_out, W_out, KH, KW] -> [N, G, 1, C_in/G, H_out, W_out, KH, KW]
        let x_broadcastable = self.unsqueeze(x_unfolded, 2);
        // w_reshaped: [G, C_out/G, C_in/G, KH, KW] -> [1, G, C_out/G, C_in/G, 1, 1, KH, KW]
        let w_broadcastable_1 = self.unsqueeze(w_reshaped, 0);
        let w_broadcastable_2 = self.unsqueeze(w_broadcastable_1, 4);
        let w_broadcastable = self.unsqueeze(w_broadcastable_2, 5);

        // 4. Expand
        let broadcast_shape = vec![
            n.clone(),
            groups.into(),
            c_out_per_group.clone(),
            c_in_per_group.clone(),
            h_out.clone(),
            w_out.clone(),
            kh.clone(),
            kw.clone(),
        ];
        let x_expanded = self.expand(x_broadcastable, broadcast_shape.clone());
        let w_expanded = self.expand(w_broadcastable, broadcast_shape);

        // 5. Multiply
        let mul_result = self.mul(x_expanded, w_expanded);

        // 6. Sum reduction
        // sum over KW (axis=7)
        let sum_kw = self.sum(mul_result, 7);
        // sum over KH (axis=6)
        let sum_kh = self.sum(sum_kw, 6);
        // sum over C_in/G (axis=3)
        let sum_c_in = self.sum(sum_kh, 3);

        // 7. Reshape output
        // [N, G, C_out/G, H_out, W_out] -> [N, C_out, H_out, W_out]
        self.reshape(sum_c_in, vec![n, c_out, h_out, w_out])
    }
}
