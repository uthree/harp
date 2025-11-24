//! 勾配計算関数
//!
//! 各演算に対する勾配計算の規則を定義します。
//! 設計方針: 演算の種類を最小限に抑え、複雑な演算は基本演算の組み合わせで表現します。

use super::tensor::Tensor;
use std::fmt::Debug;

/// 勾配計算関数のtrait
///
/// 各演算のbackward規則を実装します。
pub trait GradFn: Debug {
    /// 出力の勾配から入力の勾配を計算
    ///
    /// # 引数
    /// - `grad_output`: 出力に対する勾配
    /// - `inputs`: 前向き計算時の入力テンソル
    ///
    /// # 戻り値
    /// 各入力に対する勾配（requires_grad=falseの入力にはNoneを返す）
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>>;
}

// === 基本演算の勾配関数 ===

/// Add演算の勾配: ∂L/∂a = ∂L/∂out, ∂L/∂b = ∂L/∂out
#[derive(Debug)]
pub struct AddBackward;

impl GradFn for AddBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 2, "Add requires 2 inputs");
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }
}

/// Mul演算の勾配: ∂L/∂a = ∂L/∂out * b, ∂L/∂b = ∂L/∂out * a
#[derive(Debug)]
pub struct MulBackward;

impl GradFn for MulBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 2, "Mul requires 2 inputs");
        let a = &inputs[0];
        let b = &inputs[1];
        vec![Some(grad_output * b), Some(grad_output * a)]
    }
}

/// Neg演算の勾配: ∂L/∂a = -∂L/∂out
#[derive(Debug)]
pub struct NegBackward;

impl GradFn for NegBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Neg requires 1 input");
        vec![Some(-grad_output)]
    }
}

/// Recip演算の勾配: ∂L/∂a = -∂L/∂out / (a²)
#[derive(Debug)]
pub struct RecipBackward;

impl GradFn for RecipBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Recip requires 1 input");
        let a = &inputs[0];
        // -grad_output / (a²) = -grad_output * recip(a²) = -grad_output * recip(a) * recip(a)
        let recip_a = a.recip();
        let grad = -grad_output * &recip_a;
        let grad = &grad * &recip_a;
        vec![Some(grad)]
    }
}

/// ReduceSum演算の勾配: ∂L/∂a = expand(∂L/∂out)
///
/// 縮約した軸を元のサイズに展開します。
#[derive(Debug)]
pub struct ReduceSumBackward {
    pub axis: usize,
}

impl GradFn for ReduceSumBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "ReduceSum requires 1 input");
        let input_shape = inputs[0].data.view.shape();

        // 縮約された軸を復元（サイズ1で追加）
        let grad_unsqueezed = grad_output.data.view.clone().unsqueeze(self.axis);

        // 元のshapeに展開
        let grad_expanded_view = grad_unsqueezed.expand(input_shape.to_vec());

        // Viewを適用したGraphNodeを作成
        let grad_expanded = grad_output.data.view(grad_expanded_view);

        vec![Some(Tensor::from_graph_node(grad_expanded, false))]
    }
}

/// Max演算（要素ごと）の勾配: ∂L/∂a = ∂L/∂out * (a >= b), ∂L/∂b = ∂L/∂out * (a < b)
///
/// 注意: 等しい場合の勾配は慣例的に最初の入力に流します
#[derive(Debug)]
pub struct MaxBackward;

impl GradFn for MaxBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 2, "Max requires 2 inputs");
        // TODO: 後で実装（比較演算が必要）
        // 現時点では簡易実装として両方に勾配を流す
        vec![Some(grad_output.clone()), Some(grad_output.clone())]
    }
}

/// 定数との加算の勾配: ∂L/∂a = ∂L/∂out （定数項は勾配なし）
#[derive(Debug)]
pub struct AddConstBackward;

impl GradFn for AddConstBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "AddConst requires 1 input");
        vec![Some(grad_output.clone())]
    }
}

/// 定数との乗算の勾配: ∂L/∂a = ∂L/∂out * c
#[derive(Debug)]
pub struct MulConstBackward {
    pub constant: f32,
}

impl GradFn for MulConstBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "MulConst requires 1 input");
        vec![Some(grad_output * self.constant)]
    }
}

// === 数学関数の勾配関数 ===

/// Log2演算の勾配: ∂L/∂x = ∂L/∂out / (x * ln(2))
#[derive(Debug)]
pub struct Log2Backward;

impl GradFn for Log2Backward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Log2 requires 1 input");
        let x = &inputs[0];
        // ∂log2(x)/∂x = 1 / (x * ln(2))
        const INV_LN2: f32 = 1.0 / std::f32::consts::LN_2;
        vec![Some(grad_output / x * INV_LN2)]
    }
}

/// Exp2演算の勾配: ∂L/∂x = ∂L/∂out * 2^x * ln(2)
#[derive(Debug)]
pub struct Exp2Backward;

impl GradFn for Exp2Backward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Exp2 requires 1 input");
        let x = &inputs[0];
        // ∂2^x/∂x = 2^x * ln(2)
        const LN2: f32 = std::f32::consts::LN_2;
        vec![Some(grad_output * &x.exp2() * LN2)]
    }
}

/// Sin演算の勾配: ∂L/∂x = ∂L/∂out * cos(x)
#[derive(Debug)]
pub struct SinBackward;

impl GradFn for SinBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Sin requires 1 input");
        let x = &inputs[0];
        // ∂sin(x)/∂x = cos(x)
        vec![Some(grad_output * &x.cos())]
    }
}

/// Sqrt演算の勾配: ∂L/∂x = ∂L/∂out / (2 * sqrt(x))
#[derive(Debug)]
pub struct SqrtBackward;

impl GradFn for SqrtBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Sqrt requires 1 input");
        let x = &inputs[0];
        // ∂sqrt(x)/∂x = 1 / (2 * sqrt(x)) = 0.5 * rsqrt(x) * recip(x)
        // rsqrt(x) = 1 / sqrt(x) なので、 0.5 / sqrt(x)
        vec![Some(grad_output / &x.sqrt() * 0.5)]
    }
}

/// Pad演算の勾配: パディング部分を除去して元のサイズに戻す
///
/// grad_output から padding 部分を除去するslice操作
#[derive(Debug)]
pub struct PadBackward {
    pub padding: Vec<(usize, usize)>,
}

impl GradFn for PadBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Pad requires 1 input");

        let input = &inputs[0];
        let input_shape = input.data.view.shape();

        // パディングを除去する範囲を計算
        // 各軸について: [padding_before, padding_before + original_size]
        let ranges: Vec<(usize, usize)> = self
            .padding
            .iter()
            .enumerate()
            .map(|(i, (before, _after))| {
                // input_shape[i]は式なので、定数の場合のみ処理
                // TODO: 式の評価が必要な場合は対応が必要
                let size = match &input_shape[i] {
                    crate::graph::shape::Expr::Const(val) => *val as usize,
                    _ => panic!("PadBackward requires constant input shape"),
                };
                (*before, before + size)
            })
            .collect();

        vec![Some(grad_output.slice(ranges))]
    }
}

/// Slice演算の勾配: 元のサイズのゼロテンソルを作り、slice部分に勾配を配置
///
/// Padの逆操作として実装
#[derive(Debug)]
pub struct SliceBackward {
    pub ranges: Vec<(usize, usize)>,
}

impl GradFn for SliceBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Slice requires 1 input");

        let input = &inputs[0];
        let input_shape = input.data.view.shape();

        // Sliceの勾配は、元のサイズに戻すためにパディングを追加
        // 各軸について: padding_before = ranges[i].0, padding_after = input_shape[i] - ranges[i].1

        let padding: Vec<(usize, usize)> = self
            .ranges
            .iter()
            .enumerate()
            .map(|(i, (start, end))| {
                // input_shape[i]から実際のサイズを取得
                let input_size = match &input_shape[i] {
                    crate::graph::shape::Expr::Const(val) => *val as usize,
                    _ => {
                        // 式の場合は計算（start + (end - start) + padding_after = input_size）
                        // つまり padding_after = input_size - end
                        // しかしinput_sizeが式なので、endとの差分を計算する必要がある
                        // とりあえず、定数のみサポート
                        panic!("SliceBackward requires constant input shape")
                    }
                };
                let padding_before = *start;
                let padding_after = input_size - end;
                (padding_before, padding_after)
            })
            .collect();

        vec![Some(grad_output.pad(padding, 0.0))]
    }
}

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
        let c_in = match &input_shape[0] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv1d backward requires constant input shape"),
        };
        let _l_in = match &input_shape[1] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv1d backward requires constant input shape"),
        };

        let kernel_size = match &kernel_shape[2] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv1d backward requires constant kernel shape"),
        };

        // dilationとgroupsのサポートは後回し
        if self.dilation != 1 || self.groups != 1 {
            panic!("Conv1d backward currently only supports dilation=1, groups=1");
        }

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
        let c_in_val = match &reduced_shape[0] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => c_in,
        };
        let k_val = match &reduced_shape[1] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => kernel_size,
        };

        let reshaped_for_fold_view = reduced_for_fold.view.clone().reshape(vec![
            crate::graph::shape::Expr::from(1),
            crate::graph::shape::Expr::from((c_in_val * k_val) as isize),
            l_out_expr.clone(),
        ]);
        let reshaped_for_fold = reduced_for_fold.view(reshaped_for_fold_view);

        // foldで元の入力形状に戻す
        let l_in = match &input_shape[1] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv1d backward requires constant input shape"),
        };
        let grad_input_folded =
            reshaped_for_fold.fold1d(vec![c_in, l_in], kernel_size, self.stride, self.dilation, 1);
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
        let c_out = match &kernel_shape[0] {
            crate::graph::shape::Expr::Const(v) => *v,
            _ => panic!("Conv1d backward requires constant kernel shape"),
        };
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

        // dilationとgroupsのサポートは後回し
        if self.dilation != (1, 1) || self.groups != 1 {
            panic!("Conv2d backward currently only supports dilation=(1,1), groups=1");
        }

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
        let c_in = match &input_shape[0] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv2d backward requires constant input shape"),
        };
        let h_in = match &input_shape[1] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv2d backward requires constant input shape"),
        };
        let w_in = match &input_shape[2] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv2d backward requires constant input shape"),
        };

        let kernel_h = match &kernel_shape[2] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv2d backward requires constant kernel shape"),
        };
        let kernel_w = match &kernel_shape[3] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv2d backward requires constant kernel shape"),
        };

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
        let c_in_val = match &reduced_shape[0] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => c_in,
        };
        let kh_val = match &reduced_shape[1] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => kernel_h,
        };
        let kw_val = match &reduced_shape[2] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => kernel_w,
        };
        let h_out_val = match &reduced_shape[3] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv2d backward requires constant grad_output shape"),
        };
        let w_out_val = match &reduced_shape[4] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv2d backward requires constant grad_output shape"),
        };

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

        let c_out = match &kernel_shape[0] {
            crate::graph::shape::Expr::Const(v) => *v,
            _ => panic!("Conv2d backward requires constant kernel shape"),
        };

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
        let c_in = match &input_shape[0] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv3d backward requires constant input shape"),
        };
        let d_in = match &input_shape[1] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv3d backward requires constant input shape"),
        };
        let h_in = match &input_shape[2] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv3d backward requires constant input shape"),
        };
        let w_in = match &input_shape[3] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv3d backward requires constant input shape"),
        };

        let kernel_d = match &kernel_shape[2] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv3d backward requires constant kernel shape"),
        };
        let kernel_h = match &kernel_shape[3] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv3d backward requires constant kernel shape"),
        };
        let kernel_w = match &kernel_shape[4] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv3d backward requires constant kernel shape"),
        };

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
        let c_in_val = match &reduced_shape[0] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => c_in,
        };
        let kd_val = match &reduced_shape[1] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => kernel_d,
        };
        let kh_val = match &reduced_shape[2] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => kernel_h,
        };
        let kw_val = match &reduced_shape[3] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => kernel_w,
        };
        let d_out_val = match &reduced_shape[4] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv3d backward requires constant grad_output shape"),
        };
        let h_out_val = match &reduced_shape[5] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv3d backward requires constant grad_output shape"),
        };
        let w_out_val = match &reduced_shape[6] {
            crate::graph::shape::Expr::Const(v) => *v as usize,
            _ => panic!("Conv3d backward requires constant grad_output shape"),
        };

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

        let c_out = match &kernel_shape[0] {
            crate::graph::shape::Expr::Const(v) => *v,
            _ => panic!("Conv3d backward requires constant kernel shape"),
        };

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
