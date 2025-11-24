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
    /// - `_groups`: グループ数（現在は未使用）
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_fold_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        output_size: &[usize],
        kernel_size: &[usize],
        stride: &[usize],
        dilation: &[usize],
        _groups: usize,
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
    fn compute_fold_output_indices(
        &self,
        axes: &[usize],
        output_size: &[usize],
        kernel_size: &[usize],
        stride: &[usize],
        dilation: &[usize],
        ndim: usize,
    ) -> Vec<AstNode> {
        // 簡略化のため、現在は1次元のfoldのみサポート
        // TODO: 2D/3D foldのサポート

        match ndim {
            1 => {
                // Fold1d: 入力 [C_out, C_in * kernel_size, L_out]
                // 軸0 (c_out): そのまま出力の軸0になる可能性があるが、
                //              実際にはC_inに変換する必要がある
                // 軸1 (k_idx): C_in と k に分解
                // 軸2 (l_out_idx): l_in に変換

                let _c_out_idx = var(format!("iidx{}", axes[0]));
                let k_idx = var(format!("iidx{}", axes[1]));
                let l_out_idx = var(format!("iidx{}", axes[2]));

                // k_idx = c_in * kernel_size + k
                // c_in = k_idx / kernel_size
                let kernel_size_expr = const_int(kernel_size[0] as isize);
                let c_in = idiv(k_idx.clone(), kernel_size_expr.clone());

                // k = k_idx % kernel_size
                let k = rem(k_idx, kernel_size_expr);

                // l_in = l_out_idx * stride + k * dilation
                let l_in =
                    l_out_idx * const_int(stride[0] as isize) + k * const_int(dilation[0] as isize);

                vec![c_in, l_in]
            }
            2 => {
                // Fold2d: 入力 [C_out, C_in * kernel_h * kernel_w, H_out * W_out]
                let _c_out_idx = var(format!("iidx{}", axes[0]));
                let k_idx = var(format!("iidx{}", axes[1]));
                let spatial_idx = var(format!("iidx{}", axes[2]));

                let kernel_h = kernel_size[0];
                let kernel_w = kernel_size[1];
                let _output_h = output_size[0];
                let _output_w = output_size[1];
                let stride_h = stride[0];
                let stride_w = stride[1];
                let dilation_h = dilation[0];
                let dilation_w = dilation[1];

                // k_idx = c_in * kernel_h * kernel_w + kh * kernel_w + kw
                let kernel_hw = (kernel_h * kernel_w) as isize;
                let c_in = idiv(k_idx.clone(), const_int(kernel_hw));

                let k_hw = rem(k_idx, const_int(kernel_hw));
                let kh = idiv(k_hw.clone(), const_int(kernel_w as isize));
                let kw = rem(k_hw, const_int(kernel_w as isize));

                // spatial_idx = h_out * W_out + w_out
                let h_out = idiv(spatial_idx.clone(), const_int(output_size[1] as isize));
                let w_out = rem(spatial_idx, const_int(output_size[1] as isize));

                // h_in = h_out * stride_h + kh * dilation_h
                let h_in =
                    h_out * const_int(stride_h as isize) + kh * const_int(dilation_h as isize);

                // w_in = w_out * stride_w + kw * dilation_w
                let w_in =
                    w_out * const_int(stride_w as isize) + kw * const_int(dilation_w as isize);

                vec![c_in, h_in, w_in]
            }
            3 => {
                // Fold3d: 入力 [C_out, C_in * kernel_d * kernel_h * kernel_w, D_out * H_out * W_out]
                let _c_out_idx = var(format!("iidx{}", axes[0]));
                let k_idx = var(format!("iidx{}", axes[1]));
                let spatial_idx = var(format!("iidx{}", axes[2]));

                let kernel_d = kernel_size[0];
                let kernel_h = kernel_size[1];
                let kernel_w = kernel_size[2];
                let stride_d = stride[0];
                let stride_h = stride[1];
                let stride_w = stride[2];
                let dilation_d = dilation[0];
                let dilation_h = dilation[1];
                let dilation_w = dilation[2];

                // k_idx = c_in * kernel_d * kernel_h * kernel_w + kd * kernel_h * kernel_w + kh * kernel_w + kw
                let kernel_dhw = (kernel_d * kernel_h * kernel_w) as isize;
                let c_in = idiv(k_idx.clone(), const_int(kernel_dhw));

                let k_dhw = rem(k_idx, const_int(kernel_dhw));
                let kernel_hw = (kernel_h * kernel_w) as isize;
                let kd = idiv(k_dhw.clone(), const_int(kernel_hw));
                let k_hw = rem(k_dhw, const_int(kernel_hw));
                let kh = idiv(k_hw.clone(), const_int(kernel_w as isize));
                let kw = rem(k_hw, const_int(kernel_w as isize));

                // spatial_idx = d_out * H_out * W_out + h_out * W_out + w_out
                let output_hw = (output_size[1] * output_size[2]) as isize;
                let d_out = idiv(spatial_idx.clone(), const_int(output_hw));
                let hw_idx = rem(spatial_idx, const_int(output_hw));
                let h_out = idiv(hw_idx.clone(), const_int(output_size[2] as isize));
                let w_out = rem(hw_idx, const_int(output_size[2] as isize));

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
            _ => panic!("Unsupported ndim for fold: {}", ndim),
        }
    }
}
