//! 畳み込み演算の勾配関数

use super::{GradFn, Tensor};

// === 畳み込み演算の勾配関数 ===

/// Conv1d演算の勾配
///
/// TODO: 完全な勾配実装にはfold演算（col2im）が必要
/// 現在は簡易実装として panic! を返す
#[derive(Debug)]
#[allow(dead_code)]
pub struct Conv1dBackward {
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl GradFn for Conv1dBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 2, "Conv1d requires 2 inputs (input, kernel)");
        let input = &inputs[0];
        let kernel = &inputs[1];

        // 入力とカーネルのshape取得
        let input_shape = input.data.view.shape();
        let kernel_shape = kernel.data.view.shape();
        let grad_output_shape = grad_output.data.view.shape();

        // 定数のみサポート（動的shapeは未対応）
        let c_in = input_shape[0].expect_usize("Conv1d backward requires constant input shape");
        let l_in = input_shape[1].expect_usize("Conv1d backward requires constant input shape");
        let kernel_size =
            kernel_shape[2].expect_usize("Conv1d backward requires constant kernel shape");

        let (grad_input_data, grad_kernel_data) = if self.groups == 1 {
            // === groups=1: 既存の実装 ===

            // === 入力に対する勾配: 転置畳み込み（stride対応） ===
            // transposed convolutionは、fold演算を使って実装します
            //
            // アルゴリズム：
            // 1. grad_outputをunfold: (C_out, k*C_in, L_out) → (C_out, k, L_out)
            // 2. kernel_flippedと乗算
            // 3. foldで元の入力形状に戻す

            // kernelを反転してpermute: (C_out, C_in, k) -> (C_in, C_out, k)
            let kernel_flipped_view = kernel.data.view.clone().flip(2);
            let kernel_transposed = kernel
                .data
                .view(kernel_flipped_view)
                .view(kernel.data.view.clone().permute(vec![1, 0, 2])); // (C_in, C_out, k)

            // grad_outputをunfoldして、kernel_transposedで畳み込む代わりに、
            // より直接的な方法：grad_outputとkernel_transposedから、foldを使ってgrad_inputを構築
            //
            // stride>1の場合、grad_outputの各要素が入力の複数の位置に影響を与えます
            // これはfoldのstride parameterで自然に表現できます

            // まず、grad_outputをkernel_transposedでunfoldに相当する形に変換
            // grad_output: (C_out, L_out)
            // kernel_transposed: (C_in, C_out, k)

            // grad_outputをexpandして、kernelとの積を計算
            // (C_out, L_out) -> (C_in, C_out, k, L_out)
            let grad_out_unsqueezed = grad_output.data.view.clone().unsqueeze(0).unsqueeze(2);
            let grad_out_expanded_for_input = grad_output.data.view(grad_out_unsqueezed);

            let kernel_t_unsqueezed = kernel_transposed.view.clone().unsqueeze(3);
            let kernel_t_expanded_for_input = kernel_transposed.view(kernel_t_unsqueezed);

            let c_out_expr = &kernel_shape[0];
            let l_out_expr = &grad_output_shape[1];
            let shape_for_mult = vec![
                crate::graph::shape::Expr::from(c_in as isize),
                c_out_expr.clone(),
                crate::graph::shape::Expr::from(kernel_size as isize),
                l_out_expr.clone(),
            ];

            let grad_out_bc = grad_out_expanded_for_input.expand(shape_for_mult.clone());
            let kernel_t_bc = kernel_t_expanded_for_input.expand(shape_for_mult);

            // 乗算: (C_in, C_out, k, L_out)
            let multiplied = &grad_out_bc * &kernel_t_bc;

            // C_out次元でreduce: (C_in, k, L_out)
            let reduced_for_fold = multiplied.reduce_sum(1);

            // reshapeして(C_out, C_in*k, L_out)の形にしてからfold
            // ただし、foldは(C_out, C_in*k, L_out)の形式を想定
            // 現在は(C_in, k, L_out)なので、これを(1, C_in*k, L_out)にreshape
            let reduced_shape = reduced_for_fold.view.shape();
            let c_in_val = reduced_shape[0].as_usize().unwrap_or(c_in);
            let k_val = reduced_shape[1].as_usize().unwrap_or(kernel_size);

            let reshaped_for_fold_view = reduced_for_fold.view.clone().reshape(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from((c_in_val * k_val) as isize),
                l_out_expr.clone(),
            ]);
            let reshaped_for_fold = reduced_for_fold.view(reshaped_for_fold_view);

            // foldで元の入力形状に戻す
            let grad_input_folded = reshaped_for_fold.fold1d(
                vec![c_in, l_in],
                kernel_size,
                self.stride,
                self.dilation,
                1,
            );
            let grad_input_data = grad_input_folded;

            // === カーネルに対する勾配: 相関計算（stride対応） ===
            // grad_kernel = correlate(input, grad_output)
            // input: (C_in, L_in), grad_output: (C_out, L_out)
            // unfold input with stride -> (C_in, k, L_out)
            let input_unfolded = input
                .data
                .unfold1d(kernel_size, self.stride, self.dilation, 1);

            // unsqueeze grad_output: (C_out, L') -> (C_out, 1, 1, L')
            let grad_out_tmp1 = grad_output.data.view.clone().unsqueeze(1);
            let grad_out_tmp2 = grad_out_tmp1.unsqueeze(2);
            let grad_output_expanded_view = grad_output.data.view(grad_out_tmp2);

            // unsqueeze input_unfolded: (C_in, k, L') -> (1, C_in, k, L')
            let input_unf_tmp = input_unfolded.view.clone().unsqueeze(0);
            let input_unfolded_expanded = input_unfolded.view(input_unf_tmp);

            // expand to common shape: (C_out, C_in, k, L')
            let c_out =
                kernel_shape[0].expect_const("Conv1d backward requires constant kernel shape");
            let common_shape = vec![
                crate::graph::shape::Expr::from(c_out),
                crate::graph::shape::Expr::from(c_in as isize),
                crate::graph::shape::Expr::from(kernel_size as isize),
                grad_output_shape[1].clone(),
            ];

            let grad_out_broadcasted = grad_output_expanded_view.expand(common_shape.clone());
            let input_unf_broadcasted = input_unfolded_expanded.expand(common_shape);

            // multiply and reduce over L'
            let grad_kernel_data = (&grad_out_broadcasted * &input_unf_broadcasted).reduce_sum(3); // reduce over L'

            (grad_input_data, grad_kernel_data)
        } else {
            // === groups>1: group convolution backward ===

            let c_in_per_group = c_in / self.groups;
            let c_out =
                kernel_shape[0].expect_usize("Conv1d backward requires constant kernel shape");
            let c_out_per_group = c_out / self.groups;

            // === grad_input: transposed conv with groups ===
            // kernel: (C_out, C_in/groups, k) -> reshape -> (groups, C_out/groups, C_in/groups, k)
            // -> flip and permute -> (groups, C_in/groups, C_out/groups, k)

            // まずcontiguous化
            let kernel_contiguous_view =
                crate::graph::shape::View::contiguous(kernel.data.view.shape().to_vec());
            let kernel_contiguous = crate::graph::GraphNode::new(
                kernel.data.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![kernel.data.clone()],
                kernel_contiguous_view,
            );

            let kernel_reshaped_view = kernel_contiguous.view.clone().reshape(vec![
                crate::graph::shape::Expr::from(self.groups as isize),
                crate::graph::shape::Expr::from(c_out_per_group as isize),
                crate::graph::shape::Expr::from(c_in_per_group as isize),
                crate::graph::shape::Expr::from(kernel_size as isize),
            ]);
            let kernel_reshaped = kernel_contiguous.view(kernel_reshaped_view);

            // flip and permute: (groups, C_out/groups, C_in/groups, k) -> (groups, C_in/groups, C_out/groups, k)
            let kernel_flipped = kernel_reshaped.view(kernel_reshaped.view.clone().flip(3));
            let kernel_transposed =
                kernel_flipped.view(kernel_flipped.view.clone().permute(vec![0, 2, 1, 3]));

            // grad_output: (C_out, L') -> reshape -> (groups, C_out/groups, L')
            let grad_out_contiguous_view =
                crate::graph::shape::View::contiguous(grad_output.data.view.shape().to_vec());
            let grad_out_contiguous = crate::graph::GraphNode::new(
                grad_output.data.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![grad_output.data.clone()],
                grad_out_contiguous_view,
            );

            let grad_out_reshaped_view = grad_out_contiguous.view.clone().reshape(vec![
                crate::graph::shape::Expr::from(self.groups as isize),
                crate::graph::shape::Expr::from(c_out_per_group as isize),
                grad_output_shape[1].clone(),
            ]);
            let grad_out_reshaped = grad_out_contiguous.view(grad_out_reshaped_view);

            // expand and multiply: (groups, C_in/groups, C_out/groups, k, L')
            let grad_out_tmp = grad_out_reshaped.view.clone().unsqueeze(1).unsqueeze(3);
            let grad_out_expanded = grad_out_reshaped.view(grad_out_tmp);

            let kernel_t_tmp = kernel_transposed.view.clone().unsqueeze(4);
            let kernel_t_expanded = kernel_transposed.view(kernel_t_tmp);

            let l_out_expr = &grad_output_shape[1];
            let shape_for_mult = vec![
                crate::graph::shape::Expr::from(self.groups as isize),
                crate::graph::shape::Expr::from(c_in_per_group as isize),
                crate::graph::shape::Expr::from(c_out_per_group as isize),
                crate::graph::shape::Expr::from(kernel_size as isize),
                l_out_expr.clone(),
            ];

            let grad_out_bc = grad_out_expanded.expand(shape_for_mult.clone());
            let kernel_t_bc = kernel_t_expanded.expand(shape_for_mult);

            let multiplied = &grad_out_bc * &kernel_t_bc;
            let reduced_for_fold = multiplied.reduce_sum(2); // reduce C_out/groups

            // fold with groups: (groups, C_in/groups, k, L') -> (C_in, L)
            let grad_input_folded = reduced_for_fold.fold1d(
                vec![c_in, l_in],
                kernel_size,
                self.stride,
                self.dilation,
                self.groups,
            );
            let grad_input_data = grad_input_folded;

            // === grad_kernel: correlation with groups ===
            // unfold input with groups: (C_in, L) -> (groups, C_in/groups, k, L')
            let input_unfolded =
                input
                    .data
                    .unfold1d(kernel_size, self.stride, self.dilation, self.groups);

            // grad_output: (groups, C_out/groups, L') (already reshaped above)
            // expand: (groups, C_out/groups, C_in/groups, k, L')
            let grad_out_tmp1 = grad_out_reshaped.view.clone().unsqueeze(2).unsqueeze(3);
            let grad_out_for_kernel = grad_out_reshaped.view(grad_out_tmp1);

            let input_unf_tmp = input_unfolded.view.clone().unsqueeze(1);
            let input_unf_for_kernel = input_unfolded.view(input_unf_tmp);

            let common_shape_kernel = vec![
                crate::graph::shape::Expr::from(self.groups as isize),
                crate::graph::shape::Expr::from(c_out_per_group as isize),
                crate::graph::shape::Expr::from(c_in_per_group as isize),
                crate::graph::shape::Expr::from(kernel_size as isize),
                l_out_expr.clone(),
            ];

            let grad_out_bc_kernel = grad_out_for_kernel.expand(common_shape_kernel.clone());
            let input_unf_bc_kernel = input_unf_for_kernel.expand(common_shape_kernel);

            let grad_kernel_grouped = (&grad_out_bc_kernel * &input_unf_bc_kernel).reduce_sum(4);

            // reshape back: (groups, C_out/groups, C_in/groups, k) -> (C_out, C_in/groups, k)
            // contiguous化してからreshape
            let grad_kernel_grouped_cont_view =
                crate::graph::shape::View::contiguous(grad_kernel_grouped.view.shape().to_vec());
            let grad_kernel_grouped_cont = crate::graph::GraphNode::new(
                grad_kernel_grouped.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![grad_kernel_grouped.clone()],
                grad_kernel_grouped_cont_view,
            );

            let grad_kernel_data_view = grad_kernel_grouped_cont.view.clone().reshape(vec![
                crate::graph::shape::Expr::from(c_out as isize),
                crate::graph::shape::Expr::from(c_in_per_group as isize),
                crate::graph::shape::Expr::from(kernel_size as isize),
            ]);
            let grad_kernel_data = grad_kernel_grouped_cont.view(grad_kernel_data_view);

            (grad_input_data, grad_kernel_data)
        };

        vec![
            Some(Tensor::from_graph_node(grad_input_data, false)),
            Some(Tensor::from_graph_node(grad_kernel_data, false)),
        ]
    }
}

/// Conv2d演算の勾配
///
/// fold演算（col2im）を使用してstride対応の勾配計算を実装
#[derive(Debug)]
#[allow(dead_code)]
pub struct Conv2dBackward {
    pub stride: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl GradFn for Conv2dBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 2, "Conv2d requires 2 inputs (input, kernel)");
        let input = &inputs[0];
        let kernel = &inputs[1];

        // 入力とカーネルのshape取得
        let input_shape = input.data.view.shape();
        let kernel_shape = kernel.data.view.shape();
        let grad_output_shape = grad_output.data.view.shape();

        log::debug!(
            "Conv2dBackward: grad_output_shape = {:?} (ndim={})",
            grad_output_shape,
            grad_output_shape.len()
        );

        // 定数のみサポート
        let c_in = input_shape[0].expect_usize("Conv2d backward requires constant input shape");
        let h_in = input_shape[1].expect_usize("Conv2d backward requires constant input shape");
        let w_in = input_shape[2].expect_usize("Conv2d backward requires constant input shape");

        let kernel_h =
            kernel_shape[2].expect_usize("Conv2d backward requires constant kernel shape");
        let kernel_w =
            kernel_shape[3].expect_usize("Conv2d backward requires constant kernel shape");

        let (grad_input_data, grad_kernel_data) = if self.groups == 1 {
            // === groups=1: 既存の実装 ===

            // === grad_input: transposed convolution using fold ===
            let kernel_transposed_view = kernel
                .data
                .view
                .clone()
                .flip(2)
                .flip(3)
                .permute(vec![1, 0, 2, 3]);
            let kernel_transposed = kernel.data.view(kernel_transposed_view);
            log::debug!(
                "kernel_transposed.view.shape() = {:?}",
                kernel_transposed.view.shape()
            );

            // grad_outputをexpandしてkernelとの積を計算
            let grad_out_step1 = grad_output.data.view.clone().unsqueeze(0);
            log::debug!("After unsqueeze(0): shape = {:?}", grad_out_step1.shape());
            let grad_out_step2 = grad_out_step1.unsqueeze(2);
            log::debug!("After unsqueeze(2): shape = {:?}", grad_out_step2.shape());
            let grad_out_unsqueezed = grad_out_step2.unsqueeze(3);
            log::debug!(
                "After unsqueeze(3): shape = {:?}",
                grad_out_unsqueezed.shape()
            );
            let grad_out_expanded = grad_output.data.view(grad_out_unsqueezed);

            let kernel_t_step1 = kernel_transposed.view.clone().unsqueeze(4);
            log::debug!(
                "kernel_t after unsqueeze(4): shape = {:?}",
                kernel_t_step1.shape()
            );
            let kernel_t_unsqueezed = kernel_t_step1.unsqueeze(5);
            log::debug!(
                "kernel_t after unsqueeze(5): shape = {:?}",
                kernel_t_unsqueezed.shape()
            );
            let kernel_t_expanded = kernel_transposed.view(kernel_t_unsqueezed);
            log::debug!(
                "kernel_t_expanded.view.shape() = {:?}",
                kernel_t_expanded.view.shape()
            );

            let c_out_expr = &kernel_shape[0];
            let h_out_expr = &grad_output_shape[1];
            let w_out_expr = &grad_output_shape[2];
            let shape_for_mult = vec![
                crate::graph::shape::Expr::from(c_in as isize),
                c_out_expr.clone(),
                crate::graph::shape::Expr::from(kernel_h as isize),
                crate::graph::shape::Expr::from(kernel_w as isize),
                h_out_expr.clone(),
                w_out_expr.clone(),
            ];

            log::debug!(
                "grad_out_expanded.view.shape() = {:?}",
                grad_out_expanded.view.shape()
            );
            log::debug!("shape_for_mult = {:?}", shape_for_mult);
            let grad_out_bc = grad_out_expanded.expand(shape_for_mult.clone());
            log::debug!("After grad_out expand");
            let kernel_t_bc = kernel_t_expanded.expand(shape_for_mult);

            let multiplied = &grad_out_bc * &kernel_t_bc;
            let reduced_for_fold = multiplied.reduce_sum(1);

            // reshape for fold: (C_in, k_h, k_w, H_out, W_out) -> (1, C_in*k_h*k_w, H_out*W_out)
            let reduced_shape = reduced_for_fold.view.shape();
            let c_in_val = reduced_shape[0].as_usize().unwrap_or(c_in);
            let kh_val = reduced_shape[1].as_usize().unwrap_or(kernel_h);
            let kw_val = reduced_shape[2].as_usize().unwrap_or(kernel_w);
            let h_out_val = reduced_shape[3]
                .expect_usize("Conv2d backward requires constant grad_output shape");
            let w_out_val = reduced_shape[4]
                .expect_usize("Conv2d backward requires constant grad_output shape");

            let reshaped_for_fold_view = reduced_for_fold.view.clone().reshape(vec![
                crate::graph::shape::Expr::from(1),
                crate::graph::shape::Expr::from((c_in_val * kh_val * kw_val) as isize),
                crate::graph::shape::Expr::from((h_out_val * w_out_val) as isize),
            ]);
            let reshaped_for_fold = reduced_for_fold.view(reshaped_for_fold_view);

            let grad_input_folded = reshaped_for_fold.fold2d(
                vec![c_in, h_in, w_in],
                (kernel_h, kernel_w),
                self.stride,
                self.dilation,
                1,
            );
            let grad_input_data = grad_input_folded;

            // === grad_kernel: correlation using unfold ===
            let spatial_dims_product = h_out_expr.clone() * w_out_expr.clone();
            let input_unfolded_raw =
                input
                    .data
                    .unfold2d((kernel_h, kernel_w), self.stride, self.dilation, 1);
            // unfoldの出力をcontiguous化
            let input_unf_contiguous_view =
                crate::graph::shape::View::contiguous(input_unfolded_raw.view.shape().to_vec());
            let input_unf_contiguous = crate::graph::GraphNode::new(
                input_unfolded_raw.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![input_unfolded_raw.clone()],
                input_unf_contiguous_view,
            );
            // unfold2dの出力: (C_in, kH, kW, H_out, W_out) -> (C_in, kH, kW, H_out*W_out)
            let input_unfolded_reshaped_view = input_unf_contiguous.view.clone().reshape(vec![
                input_shape[0].clone(),
                crate::graph::shape::Expr::from(kernel_h as isize),
                crate::graph::shape::Expr::from(kernel_w as isize),
                spatial_dims_product.clone(),
            ]);
            let input_unfolded = input_unf_contiguous.view(input_unfolded_reshaped_view);

            // reshape grad_output: (C_out, H_out, W_out) -> (C_out, H_out*W_out)
            // まずcontiguous化
            let grad_out_contiguous_view =
                crate::graph::shape::View::contiguous(grad_output_shape.to_vec());
            let grad_out_contiguous = crate::graph::GraphNode::new(
                grad_output.data.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![grad_output.data.clone()],
                grad_out_contiguous_view,
            );
            // reshape
            let grad_out_reshaped_view = grad_out_contiguous.view.clone().reshape(vec![
                grad_output_shape[0].clone(),
                spatial_dims_product.clone(),
            ]);
            let grad_out_reshaped = grad_out_contiguous.view(grad_out_reshaped_view);

            // unsqueeze: (C_out, H_out*W_out) -> (C_out, 1, 1, 1, H_out*W_out)
            let grad_out_tmp1 = grad_out_reshaped.view.clone().unsqueeze(1);
            let grad_out_tmp2 = grad_out_tmp1.unsqueeze(2);
            let grad_out_tmp3 = grad_out_tmp2.unsqueeze(3);
            let grad_output_expanded_view = grad_out_reshaped.view(grad_out_tmp3);

            let input_unf_tmp = input_unfolded.view.clone().unsqueeze(0);
            let input_unfolded_expanded = input_unfolded.view(input_unf_tmp);

            let c_out =
                kernel_shape[0].expect_const("Conv2d backward requires constant kernel shape");

            let common_shape = vec![
                crate::graph::shape::Expr::from(c_out),
                crate::graph::shape::Expr::from(c_in as isize),
                crate::graph::shape::Expr::from(kernel_h as isize),
                crate::graph::shape::Expr::from(kernel_w as isize),
                spatial_dims_product,
            ];

            let grad_out_broadcasted = grad_output_expanded_view.expand(common_shape.clone());
            let input_unf_broadcasted = input_unfolded_expanded.expand(common_shape);

            let grad_kernel_data = (&grad_out_broadcasted * &input_unf_broadcasted).reduce_sum(4);

            (grad_input_data, grad_kernel_data)
        } else {
            // === groups>1: group convolution backward ===
            // Conv1dBackwardと同じパターンでgroups対応

            let c_in_per_group = c_in / self.groups;
            let c_out =
                kernel_shape[0].expect_usize("Conv2d backward requires constant kernel shape");
            let c_out_per_group = c_out / self.groups;

            // === grad_input: transposed conv with groups ===
            // kernel: (C_out, C_in/groups, kH, kW) -> reshape -> (groups, C_out/groups, C_in/groups, kH, kW)
            // -> flip and permute -> (groups, C_in/groups, C_out/groups, kH, kW)

            // まずcontiguous化
            let kernel_contiguous_view =
                crate::graph::shape::View::contiguous(kernel.data.view.shape().to_vec());
            let kernel_contiguous = crate::graph::GraphNode::new(
                kernel.data.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![kernel.data.clone()],
                kernel_contiguous_view,
            );

            let kernel_reshaped_view = kernel_contiguous.view.clone().reshape(vec![
                crate::graph::shape::Expr::from(self.groups as isize),
                crate::graph::shape::Expr::from(c_out_per_group as isize),
                crate::graph::shape::Expr::from(c_in_per_group as isize),
                crate::graph::shape::Expr::from(kernel_h as isize),
                crate::graph::shape::Expr::from(kernel_w as isize),
            ]);
            let kernel_reshaped = kernel_contiguous.view(kernel_reshaped_view);

            // flip and permute: (groups, C_out/groups, C_in/groups, kH, kW) -> (groups, C_in/groups, C_out/groups, kH, kW)
            let kernel_flipped = kernel_reshaped.view(kernel_reshaped.view.clone().flip(3).flip(4));
            let kernel_transposed =
                kernel_flipped.view(kernel_flipped.view.clone().permute(vec![0, 2, 1, 3, 4]));

            // grad_output: (C_out, H', W') -> reshape -> (groups, C_out/groups, H', W')
            let grad_out_contiguous_view =
                crate::graph::shape::View::contiguous(grad_output.data.view.shape().to_vec());
            let grad_out_contiguous = crate::graph::GraphNode::new(
                grad_output.data.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![grad_output.data.clone()],
                grad_out_contiguous_view,
            );

            let grad_out_reshaped_view = grad_out_contiguous.view.clone().reshape(vec![
                crate::graph::shape::Expr::from(self.groups as isize),
                crate::graph::shape::Expr::from(c_out_per_group as isize),
                grad_output_shape[1].clone(),
                grad_output_shape[2].clone(),
            ]);
            let grad_out_reshaped = grad_out_contiguous.view(grad_out_reshaped_view);

            // expand and multiply: (groups, C_in/groups, C_out/groups, kH, kW, H', W')
            let grad_out_tmp = grad_out_reshaped
                .view
                .clone()
                .unsqueeze(1)
                .unsqueeze(3)
                .unsqueeze(4);
            let grad_out_expanded = grad_out_reshaped.view(grad_out_tmp);

            let kernel_t_tmp = kernel_transposed.view.clone().unsqueeze(5).unsqueeze(6);
            let kernel_t_expanded = kernel_transposed.view(kernel_t_tmp);

            let h_out_expr = &grad_output_shape[1];
            let w_out_expr = &grad_output_shape[2];
            let shape_for_mult = vec![
                crate::graph::shape::Expr::from(self.groups as isize),
                crate::graph::shape::Expr::from(c_in_per_group as isize),
                crate::graph::shape::Expr::from(c_out_per_group as isize),
                crate::graph::shape::Expr::from(kernel_h as isize),
                crate::graph::shape::Expr::from(kernel_w as isize),
                h_out_expr.clone(),
                w_out_expr.clone(),
            ];

            let grad_out_bc = grad_out_expanded.expand(shape_for_mult.clone());
            let kernel_t_bc = kernel_t_expanded.expand(shape_for_mult);

            let multiplied = &grad_out_bc * &kernel_t_bc;
            let reduced_for_fold = multiplied.reduce_sum(2); // reduce C_out/groups

            // fold with groups: (groups, C_in/groups, kH, kW, H', W') -> (C_in, H, W)
            let grad_input_folded = reduced_for_fold.fold2d(
                vec![c_in, h_in, w_in],
                (kernel_h, kernel_w),
                self.stride,
                self.dilation,
                self.groups,
            );
            let grad_input_data = grad_input_folded;

            // === grad_kernel: correlation with groups ===
            // unfold input with groups: (C_in, H, W) -> (groups, C_in/groups, kH, kW, H', W')
            let input_unfolded = input.data.unfold2d(
                (kernel_h, kernel_w),
                self.stride,
                self.dilation,
                self.groups,
            );

            // reshape grad_output for spatial flatten: (groups, C_out/groups, H', W') -> (groups, C_out/groups, H'*W')
            let spatial_dims_product = h_out_expr.clone() * w_out_expr.clone();
            let grad_out_for_kernel_view = grad_out_reshaped.view.clone().reshape(vec![
                crate::graph::shape::Expr::from(self.groups as isize),
                crate::graph::shape::Expr::from(c_out_per_group as isize),
                spatial_dims_product.clone(),
            ]);
            // contiguous化してからreshape
            let grad_out_cont_for_reshape_view =
                crate::graph::shape::View::contiguous(grad_out_reshaped.view.shape().to_vec());
            let grad_out_cont_for_reshape = crate::graph::GraphNode::new(
                grad_out_reshaped.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![grad_out_reshaped.clone()],
                grad_out_cont_for_reshape_view,
            );
            let grad_out_for_kernel = grad_out_cont_for_reshape.view(grad_out_for_kernel_view);

            // reshape input_unfolded: (groups, C_in/groups, kH, kW, H', W') -> (groups, C_in/groups, kH, kW, H'*W')
            let input_unf_contiguous_view =
                crate::graph::shape::View::contiguous(input_unfolded.view.shape().to_vec());
            let input_unf_contiguous = crate::graph::GraphNode::new(
                input_unfolded.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![input_unfolded.clone()],
                input_unf_contiguous_view,
            );
            let input_unf_reshaped_view = input_unf_contiguous.view.clone().reshape(vec![
                crate::graph::shape::Expr::from(self.groups as isize),
                crate::graph::shape::Expr::from(c_in_per_group as isize),
                crate::graph::shape::Expr::from(kernel_h as isize),
                crate::graph::shape::Expr::from(kernel_w as isize),
                spatial_dims_product.clone(),
            ]);
            let input_unf_reshaped = input_unf_contiguous.view(input_unf_reshaped_view);

            // expand: (groups, C_out/groups, C_in/groups, kH, kW, H'*W')
            let grad_out_tmp1 = grad_out_for_kernel
                .view
                .clone()
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4);
            let grad_out_for_kernel_expanded = grad_out_for_kernel.view(grad_out_tmp1);

            let input_unf_tmp = input_unf_reshaped.view.clone().unsqueeze(1);
            let input_unf_for_kernel = input_unf_reshaped.view(input_unf_tmp);

            let common_shape_kernel = vec![
                crate::graph::shape::Expr::from(self.groups as isize),
                crate::graph::shape::Expr::from(c_out_per_group as isize),
                crate::graph::shape::Expr::from(c_in_per_group as isize),
                crate::graph::shape::Expr::from(kernel_h as isize),
                crate::graph::shape::Expr::from(kernel_w as isize),
                spatial_dims_product,
            ];

            let grad_out_bc_kernel =
                grad_out_for_kernel_expanded.expand(common_shape_kernel.clone());
            let input_unf_bc_kernel = input_unf_for_kernel.expand(common_shape_kernel);

            let grad_kernel_grouped = (&grad_out_bc_kernel * &input_unf_bc_kernel).reduce_sum(5);

            // reshape back: (groups, C_out/groups, C_in/groups, kH, kW) -> (C_out, C_in/groups, kH, kW)
            // contiguous化してからreshape
            let grad_kernel_grouped_cont_view =
                crate::graph::shape::View::contiguous(grad_kernel_grouped.view.shape().to_vec());
            let grad_kernel_grouped_cont = crate::graph::GraphNode::new(
                grad_kernel_grouped.dtype.clone(),
                crate::graph::GraphOp::Contiguous {
                    elementwise_strategies: None,
                },
                vec![grad_kernel_grouped.clone()],
                grad_kernel_grouped_cont_view,
            );

            let grad_kernel_data_view = grad_kernel_grouped_cont.view.clone().reshape(vec![
                crate::graph::shape::Expr::from(c_out as isize),
                crate::graph::shape::Expr::from(c_in_per_group as isize),
                crate::graph::shape::Expr::from(kernel_h as isize),
                crate::graph::shape::Expr::from(kernel_w as isize),
            ]);
            let grad_kernel_data = grad_kernel_grouped_cont.view(grad_kernel_data_view);

            (grad_input_data, grad_kernel_data)
        };

        vec![
            Some(Tensor::from_graph_node(grad_input_data, false)),
            Some(Tensor::from_graph_node(grad_kernel_data, false)),
        ]
    }
}

/// Conv3d演算の勾配
///
/// fold演算（col2im）を使用してstride対応の勾配計算を実装
#[derive(Debug)]
#[allow(dead_code)]
pub struct Conv3dBackward {
    pub stride: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
    pub groups: usize,
}

impl GradFn for Conv3dBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 2, "Conv3d requires 2 inputs (input, kernel)");
        let input = &inputs[0];
        let kernel = &inputs[1];

        // dilationとgroupsのサポートは後回し
        if self.dilation != (1, 1, 1) || self.groups != 1 {
            panic!("Conv3d backward currently only supports dilation=(1,1,1), groups=1");
        }

        // 入力とカーネルのshape取得
        let input_shape = input.data.view.shape();
        let kernel_shape = kernel.data.view.shape();
        let grad_output_shape = grad_output.data.view.shape();

        log::debug!(
            "Conv3dBackward: grad_output_shape = {:?} (ndim={})",
            grad_output_shape,
            grad_output_shape.len()
        );

        // 定数のみサポート
        let c_in = input_shape[0].expect_usize("Conv3d backward requires constant input shape");
        let d_in = input_shape[1].expect_usize("Conv3d backward requires constant input shape");
        let h_in = input_shape[2].expect_usize("Conv3d backward requires constant input shape");
        let w_in = input_shape[3].expect_usize("Conv3d backward requires constant input shape");

        let kernel_d =
            kernel_shape[2].expect_usize("Conv3d backward requires constant kernel shape");
        let kernel_h =
            kernel_shape[3].expect_usize("Conv3d backward requires constant kernel shape");
        let kernel_w =
            kernel_shape[4].expect_usize("Conv3d backward requires constant kernel shape");

        // === grad_input: transposed convolution using fold ===
        let kernel_transposed_view = kernel
            .data
            .view
            .clone()
            .flip(2)
            .flip(3)
            .flip(4)
            .permute(vec![1, 0, 2, 3, 4]);
        let kernel_transposed = kernel.data.view(kernel_transposed_view);

        // grad_outputをexpandしてkernelとの積を計算
        let grad_out_unsqueezed = grad_output
            .data
            .view
            .clone()
            .unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(4);
        let grad_out_expanded = grad_output.data.view(grad_out_unsqueezed);

        let kernel_t_unsqueezed = kernel_transposed
            .view
            .clone()
            .unsqueeze(5)
            .unsqueeze(6)
            .unsqueeze(7);
        let kernel_t_expanded = kernel_transposed.view(kernel_t_unsqueezed);

        let c_out_expr = &kernel_shape[0];
        let d_out_expr = &grad_output_shape[1];
        let h_out_expr = &grad_output_shape[2];
        let w_out_expr = &grad_output_shape[3];
        let shape_for_mult = vec![
            crate::graph::shape::Expr::from(c_in as isize),
            c_out_expr.clone(),
            crate::graph::shape::Expr::from(kernel_d as isize),
            crate::graph::shape::Expr::from(kernel_h as isize),
            crate::graph::shape::Expr::from(kernel_w as isize),
            d_out_expr.clone(),
            h_out_expr.clone(),
            w_out_expr.clone(),
        ];

        let grad_out_bc = grad_out_expanded.expand(shape_for_mult.clone());
        let kernel_t_bc = kernel_t_expanded.expand(shape_for_mult);

        let multiplied = &grad_out_bc * &kernel_t_bc;
        let reduced_for_fold = multiplied.reduce_sum(1);

        // reshape for fold
        let reduced_shape = reduced_for_fold.view.shape();
        let c_in_val = reduced_shape[0].as_usize().unwrap_or(c_in);
        let kd_val = reduced_shape[1].as_usize().unwrap_or(kernel_d);
        let kh_val = reduced_shape[2].as_usize().unwrap_or(kernel_h);
        let kw_val = reduced_shape[3].as_usize().unwrap_or(kernel_w);
        let d_out_val =
            reduced_shape[4].expect_usize("Conv3d backward requires constant grad_output shape");
        let h_out_val =
            reduced_shape[5].expect_usize("Conv3d backward requires constant grad_output shape");
        let w_out_val =
            reduced_shape[6].expect_usize("Conv3d backward requires constant grad_output shape");

        let reshaped_for_fold_view = reduced_for_fold.view.clone().reshape(vec![
            crate::graph::shape::Expr::from(1),
            crate::graph::shape::Expr::from((c_in_val * kd_val * kh_val * kw_val) as isize),
            crate::graph::shape::Expr::from((d_out_val * h_out_val * w_out_val) as isize),
        ]);
        let reshaped_for_fold = reduced_for_fold.view(reshaped_for_fold_view);

        let grad_input_folded = reshaped_for_fold.fold3d(
            vec![c_in, d_in, h_in, w_in],
            (kernel_d, kernel_h, kernel_w),
            self.stride,
            self.dilation,
            1,
        );
        let grad_input_data = grad_input_folded;

        // === grad_kernel: correlation using unfold ===
        let spatial_dims_product = d_out_expr.clone() * h_out_expr.clone() * w_out_expr.clone();
        let input_unfolded_raw = input.data.unfold3d(
            (kernel_d, kernel_h, kernel_w),
            self.stride,
            self.dilation,
            1,
        );
        // unfoldの出力をcontiguous化
        let input_unf_contiguous_view =
            crate::graph::shape::View::contiguous(input_unfolded_raw.view.shape().to_vec());
        let input_unf_contiguous = crate::graph::GraphNode::new(
            input_unfolded_raw.dtype.clone(),
            crate::graph::GraphOp::Contiguous {
                elementwise_strategies: None,
            },
            vec![input_unfolded_raw.clone()],
            input_unf_contiguous_view,
        );
        // unfold3dの出力: (C_in, kD, kH, kW, D_out, H_out, W_out) -> (C_in, kD, kH, kW, D_out*H_out*W_out)
        let input_unfolded_reshaped_view = input_unf_contiguous.view.clone().reshape(vec![
            input_shape[0].clone(),
            crate::graph::shape::Expr::from(kernel_d as isize),
            crate::graph::shape::Expr::from(kernel_h as isize),
            crate::graph::shape::Expr::from(kernel_w as isize),
            spatial_dims_product.clone(),
        ]);
        let input_unfolded = input_unf_contiguous.view(input_unfolded_reshaped_view);

        // reshape grad_output: (C_out, D_out, H_out, W_out) -> (C_out, D_out*H_out*W_out)
        // まずcontiguous化
        let grad_out_contiguous_view =
            crate::graph::shape::View::contiguous(grad_output_shape.to_vec());
        let grad_out_contiguous = crate::graph::GraphNode::new(
            grad_output.data.dtype.clone(),
            crate::graph::GraphOp::Contiguous {
                elementwise_strategies: None,
            },
            vec![grad_output.data.clone()],
            grad_out_contiguous_view,
        );
        // reshape
        let grad_out_reshaped_view = grad_out_contiguous.view.clone().reshape(vec![
            grad_output_shape[0].clone(),
            spatial_dims_product.clone(),
        ]);
        let grad_out_reshaped = grad_out_contiguous.view(grad_out_reshaped_view);

        // unsqueeze: (C_out, D_out*H_out*W_out) -> (C_out, 1, 1, 1, 1, D_out*H_out*W_out)
        let grad_out_tmp1 = grad_out_reshaped.view.clone().unsqueeze(1);
        let grad_out_tmp2 = grad_out_tmp1.unsqueeze(2);
        let grad_out_tmp3 = grad_out_tmp2.unsqueeze(3);
        let grad_out_tmp4 = grad_out_tmp3.unsqueeze(4);
        let grad_output_expanded_view = grad_out_reshaped.view(grad_out_tmp4);

        let input_unf_tmp = input_unfolded.view.clone().unsqueeze(0);
        let input_unfolded_expanded = input_unfolded.view(input_unf_tmp);

        let c_out = kernel_shape[0].expect_const("Conv3d backward requires constant kernel shape");

        let common_shape = vec![
            crate::graph::shape::Expr::from(c_out),
            crate::graph::shape::Expr::from(c_in as isize),
            crate::graph::shape::Expr::from(kernel_d as isize),
            crate::graph::shape::Expr::from(kernel_h as isize),
            crate::graph::shape::Expr::from(kernel_w as isize),
            spatial_dims_product,
        ];

        let grad_out_broadcasted = grad_output_expanded_view.expand(common_shape.clone());
        let input_unf_broadcasted = input_unfolded_expanded.expand(common_shape);

        let grad_kernel_data = (&grad_out_broadcasted * &input_unf_broadcasted).reduce_sum(5);

        vec![
            Some(Tensor::from_graph_node(grad_input_data, false)),
            Some(Tensor::from_graph_node(grad_kernel_data, false)),
        ]
    }
}
