//! Fold演算のlowering
//!
//! Unfoldの逆操作（col2im）を行うカーネル関数を生成します。
//! Conv演算のbackward計算などで使用されます。

use crate::ast::{AstNode, Mutability, Scope, helper::*};
use crate::graph::GraphNode;

use super::Lowerer;

impl Lowerer {
    /// Fold演算をカーネル関数に変換
    ///
    /// アルゴリズム:
    /// 1. 出力バッファを0で初期化
    /// 2. 入力の各要素を適切な出力位置に加算（accumulate）
    ///
    /// # 引数
    /// - `node`: Foldノード
    /// - `node_id`: カーネル関数のID
    /// - `output_size`: 出力サイズ（unfold前のサイズ）
    /// - `kernel_size`: カーネルサイズ
    /// - `stride`: ストライド
    /// - `dilation`: 膨張率
    /// - `groups`: グループ数
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_fold_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        output_size: &[usize],
        kernel_size: &[usize],
        stride: &[usize],
        dilation: &[usize],
        groups: usize,
    ) -> Result<AstNode, String> {
        let input_node = &node.src[0];
        let output_view = &node.view;
        let output_shape = output_view.shape();

        let ndim = kernel_size.len();

        // パラメータを生成
        let mut params = Vec::new();
        params.push(self.create_input_param(0, &input_node.dtype)?);
        params.push(self.create_output_param(&node.dtype)?);
        params.extend(self.extract_shape_params(output_shape));

        let mut body_statements = Vec::new();

        // === ステップ1: 出力バッファをゼロで初期化 ===
        let init_statements = self.generate_fold_init_loops(node)?;
        body_statements.extend(init_statements);

        // === ステップ2: 入力から出力へ加算 ===
        let accumulate_statements = self.generate_fold_accumulate_loops(
            node,
            input_node,
            output_size,
            kernel_size,
            stride,
            dilation,
            ndim,
            groups,
        )?;
        body_statements.extend(accumulate_statements);

        // カーネル関数を作成
        Ok(self.create_kernel_function(node_id, params, body_statements, Scope::new()))
    }

    /// Fold初期化ループを生成
    ///
    /// 出力バッファ全体を0で初期化
    fn generate_fold_init_loops(&mut self, node: &GraphNode) -> Result<Vec<AstNode>, String> {
        let output_view = &node.view;
        let output_shape = output_view.shape();
        let ndim = output_view.ndim();

        if ndim == 0 {
            // スカラーの場合
            let output_offset = self.compute_offset_from_view(node, &[]);
            return Ok(vec![store(var("output"), output_offset, const_f32(0.0))]);
        }

        let axes: Vec<usize> = (0..ndim).collect();

        // 最内側のループ本体: output[offset] = 0.0
        let output_offset = self.compute_offset_for_output(&axes, node);
        let body_statements = vec![store(var("output"), output_offset, const_f32(0.0))];

        // ネストされたループを生成
        let (loop_statements, _) =
            self.generate_nested_loops(ndim, output_shape, "oidx", body_statements, Scope::new());

        Ok(loop_statements)
    }

    /// Fold accumulate ループを生成
    ///
    /// 入力の各要素を適切な出力位置に加算
    #[allow(clippy::too_many_arguments)]
    fn generate_fold_accumulate_loops(
        &mut self,
        node: &GraphNode,
        input_node: &GraphNode,
        output_size: &[usize],
        kernel_size: &[usize],
        stride: &[usize],
        dilation: &[usize],
        ndim: usize,
        groups: usize,
    ) -> Result<Vec<AstNode>, String> {
        let input_view = &input_node.view;
        let input_shape = input_view.shape();

        // 入力の形状を確認
        // Fold1dの場合: [C_out, C_in * kernel_size, L_out]
        // Fold2dの場合: [C_out, C_in * kernel_h * kernel_w, H_out * W_out]
        // Fold3dの場合: [C_out, C_in * kernel_d * kernel_h * kernel_w, D_out * H_out * W_out]

        let mut scope = Scope::new();
        let mut body_statements = Vec::new();

        // 入力のループ用のインデックス
        let axes: Vec<usize> = (0..input_view.ndim()).collect();

        // 入力からロード（iidxベースのオフセット計算）
        let input_offset = self.compute_input_offset_with_prefix(input_node, &axes, "iidx");
        let alu_var = self.fresh_alu();
        let input_dtype = self.graph_dtype_to_ast(&input_node.dtype)?;

        scope
            .declare(alu_var.clone(), input_dtype.clone(), Mutability::Mutable)
            .map_err(|e| format!("Failed to declare variable: {:?}", e))?;

        // 入力から読み込み
        body_statements.push(assign(
            &alu_var,
            load(var("input0"), input_offset, input_dtype.clone()),
        ));

        // 出力位置を計算
        // Fold1dの場合の例:
        // 入力インデックス: [c_out, k_idx, l_out_idx]
        // 出力位置: [c_in, l_in]
        // c_in = k_idx / kernel_size
        // k = k_idx % kernel_size
        // l_in = l_out_idx * stride + k * dilation

        // 出力インデックスの計算
        let output_indices = self.compute_fold_output_indices(
            &axes,
            output_size,
            kernel_size,
            stride,
            dilation,
            ndim,
            groups,
        );

        // 出力のオフセットを計算
        let output_offset = self.compute_output_offset_from_indices(node, &output_indices);

        // 出力の現在値を読み込み
        let acc_var = self.fresh_acc();
        scope
            .declare(acc_var.clone(), input_dtype.clone(), Mutability::Mutable)
            .map_err(|e| format!("Failed to declare accumulator: {:?}", e))?;

        body_statements.push(assign(
            &acc_var,
            load(var("output"), output_offset.clone(), input_dtype.clone()),
        ));

        // 加算
        body_statements.push(assign(&acc_var, var(&acc_var) + var(&alu_var)));

        // 出力に書き戻し
        body_statements.push(store(var("output"), output_offset, var(&acc_var)));

        // ネストされたループを生成
        let (loop_statements, _) = self.generate_nested_loops(
            input_view.ndim(),
            input_shape,
            "iidx",
            body_statements,
            scope,
        );

        Ok(loop_statements)
    }

    /// Foldの出力インデックスを計算
    #[allow(clippy::too_many_arguments)]
    fn compute_fold_output_indices(
        &self,
        axes: &[usize],
        output_size: &[usize],
        _kernel_size: &[usize],
        stride: &[usize],
        dilation: &[usize],
        ndim: usize,
        groups: usize,
    ) -> Vec<AstNode> {
        match ndim {
            1 => {
                if groups == 1 {
                    // groups=1: 入力 (C, k, L')
                    // 軸0: C
                    // 軸1: k
                    // 軸2: L'

                    let c_in = var(format!("iidx{}", axes[0]));
                    let k = var(format!("iidx{}", axes[1]));
                    let l_out_idx = var(format!("iidx{}", axes[2]));

                    // l_in = l_out_idx * stride + k * dilation
                    let l_in = l_out_idx * const_int(stride[0] as isize)
                        + k * const_int(dilation[0] as isize);

                    vec![c_in, l_in]
                } else {
                    // groups>1: 入力 (groups, C/groups, k, L')
                    // 軸0: groups
                    // 軸1: C/groups
                    // 軸2: k
                    // 軸3: L'

                    let g_idx = var(format!("iidx{}", axes[0]));
                    let c_per_group_idx = var(format!("iidx{}", axes[1]));
                    let k = var(format!("iidx{}", axes[2]));
                    let l_out_idx = var(format!("iidx{}", axes[3]));

                    // c_in = g_idx * (C/groups) + c_per_group_idx
                    let c_per_group = (output_size[0] / groups) as isize;
                    let c_in = g_idx * const_int(c_per_group) + c_per_group_idx;

                    // l_in = l_out_idx * stride + k * dilation
                    let l_in = l_out_idx * const_int(stride[0] as isize)
                        + k * const_int(dilation[0] as isize);

                    vec![c_in, l_in]
                }
            }
            2 => {
                if groups == 1 {
                    // groups=1: 入力 (C, kH, kW, H', W')
                    // 軸0: C
                    // 軸1: kH
                    // 軸2: kW
                    // 軸3: H'
                    // 軸4: W'

                    let c_in = var(format!("iidx{}", axes[0]));
                    let kh = var(format!("iidx{}", axes[1]));
                    let kw = var(format!("iidx{}", axes[2]));
                    let h_out = var(format!("iidx{}", axes[3]));
                    let w_out = var(format!("iidx{}", axes[4]));

                    let stride_h = stride[0];
                    let stride_w = stride[1];
                    let dilation_h = dilation[0];
                    let dilation_w = dilation[1];

                    // h_in = h_out * stride_h + kh * dilation_h
                    let h_in =
                        h_out * const_int(stride_h as isize) + kh * const_int(dilation_h as isize);

                    // w_in = w_out * stride_w + kw * dilation_w
                    let w_in =
                        w_out * const_int(stride_w as isize) + kw * const_int(dilation_w as isize);

                    vec![c_in, h_in, w_in]
                } else {
                    // groups>1: 入力 (groups, C/groups, kH, kW, H', W')
                    // 軸0: groups
                    // 軸1: C/groups
                    // 軸2: kH
                    // 軸3: kW
                    // 軸4: H'
                    // 軸5: W'

                    let g_idx = var(format!("iidx{}", axes[0]));
                    let c_per_group_idx = var(format!("iidx{}", axes[1]));
                    let kh = var(format!("iidx{}", axes[2]));
                    let kw = var(format!("iidx{}", axes[3]));
                    let h_out = var(format!("iidx{}", axes[4]));
                    let w_out = var(format!("iidx{}", axes[5]));

                    // c_in = g_idx * (C/groups) + c_per_group_idx
                    let c_per_group = (output_size[0] / groups) as isize;
                    let c_in = g_idx * const_int(c_per_group) + c_per_group_idx;

                    let stride_h = stride[0];
                    let stride_w = stride[1];
                    let dilation_h = dilation[0];
                    let dilation_w = dilation[1];

                    // h_in = h_out * stride_h + kh * dilation_h
                    let h_in =
                        h_out * const_int(stride_h as isize) + kh * const_int(dilation_h as isize);

                    // w_in = w_out * stride_w + kw * dilation_w
                    let w_in =
                        w_out * const_int(stride_w as isize) + kw * const_int(dilation_w as isize);

                    vec![c_in, h_in, w_in]
                }
            }
            3 => {
                if groups == 1 {
                    // groups=1: 入力 (C, kD, kH, kW, D', H', W')
                    // 軸0: C
                    // 軸1: kD
                    // 軸2: kH
                    // 軸3: kW
                    // 軸4: D'
                    // 軸5: H'
                    // 軸6: W'

                    let c_in = var(format!("iidx{}", axes[0]));
                    let kd = var(format!("iidx{}", axes[1]));
                    let kh = var(format!("iidx{}", axes[2]));
                    let kw = var(format!("iidx{}", axes[3]));
                    let d_out = var(format!("iidx{}", axes[4]));
                    let h_out = var(format!("iidx{}", axes[5]));
                    let w_out = var(format!("iidx{}", axes[6]));

                    let stride_d = stride[0];
                    let stride_h = stride[1];
                    let stride_w = stride[2];
                    let dilation_d = dilation[0];
                    let dilation_h = dilation[1];
                    let dilation_w = dilation[2];

                    // d_in = d_out * stride_d + kd * dilation_d
                    let d_in =
                        d_out * const_int(stride_d as isize) + kd * const_int(dilation_d as isize);

                    // h_in = h_out * stride_h + kh * dilation_h
                    let h_in =
                        h_out * const_int(stride_h as isize) + kh * const_int(dilation_h as isize);

                    // w_in = w_out * stride_w + kw * dilation_w
                    let w_in =
                        w_out * const_int(stride_w as isize) + kw * const_int(dilation_w as isize);

                    vec![c_in, d_in, h_in, w_in]
                } else {
                    // groups>1: 入力 (groups, C/groups, kD, kH, kW, D', H', W')
                    // 軸0: groups
                    // 軸1: C/groups
                    // 軸2: kD
                    // 軸3: kH
                    // 軸4: kW
                    // 軸5: D'
                    // 軸6: H'
                    // 軸7: W'

                    let g_idx = var(format!("iidx{}", axes[0]));
                    let c_per_group_idx = var(format!("iidx{}", axes[1]));
                    let kd = var(format!("iidx{}", axes[2]));
                    let kh = var(format!("iidx{}", axes[3]));
                    let kw = var(format!("iidx{}", axes[4]));
                    let d_out = var(format!("iidx{}", axes[5]));
                    let h_out = var(format!("iidx{}", axes[6]));
                    let w_out = var(format!("iidx{}", axes[7]));

                    // c_in = g_idx * (C/groups) + c_per_group_idx
                    let c_per_group = (output_size[0] / groups) as isize;
                    let c_in = g_idx * const_int(c_per_group) + c_per_group_idx;

                    let stride_d = stride[0];
                    let stride_h = stride[1];
                    let stride_w = stride[2];
                    let dilation_d = dilation[0];
                    let dilation_h = dilation[1];
                    let dilation_w = dilation[2];

                    // d_in = d_out * stride_d + kd * dilation_d
                    let d_in =
                        d_out * const_int(stride_d as isize) + kd * const_int(dilation_d as isize);

                    // h_in = h_out * stride_h + kh * dilation_h
                    let h_in =
                        h_out * const_int(stride_h as isize) + kh * const_int(dilation_h as isize);

                    // w_in = w_out * stride_w + kw * dilation_w
                    let w_in =
                        w_out * const_int(stride_w as isize) + kw * const_int(dilation_w as isize);

                    vec![c_in, d_in, h_in, w_in]
                }
            }
            _ => panic!("Unsupported ndim for fold: {}", ndim),
        }
    }
}
