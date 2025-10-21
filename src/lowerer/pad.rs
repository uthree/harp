use super::utils::LowererUtils;
use crate::ast::helper::{block_with_statements, store};
use crate::ast::{AstNode, DType};
use crate::graph::shape::view::View;
use crate::graph::shape::Expr;

/// Pad演算のloweringを担当
pub(super) struct PadLowerer;

impl PadLowerer {
    /// Pad演算をlowerする
    ///
    /// パディング処理は以下の手順で行う：
    /// 1. 出力配列全体を0で初期化
    /// 2. 入力データを適切な位置にコピー
    pub fn lower(
        input_view: &View,
        output_view: &View,
        _pad_axis: usize,
        _pad_amount: &Expr,
        input_var: &str,
        output_var: &str,
        dtype: &DType,
    ) -> AstNode {
        // 初期化ループと入力コピーループを生成
        let init_loop = Self::create_init_loop(output_view, output_var, dtype, 0);
        let copy_loop = Self::create_copy_loop(input_view, output_view, input_var, output_var, 0);

        block_with_statements(vec![init_loop, copy_loop])
    }

    /// 出力配列を0で初期化するループを生成
    fn create_init_loop(
        output_view: &View,
        output_var: &str,
        _dtype: &DType,
        dim: usize,
    ) -> AstNode {
        let View::Linear {
            shape,
            strides,
            offset,
            ..
        } = output_view;

        if dim >= shape.len() {
            // 最内レベル: 0で初期化
            let index = LowererUtils::compute_memory_index(strides, offset, dim);
            // F32型の0.0リテラルを使用
            let zero_value = AstNode::from(0.0f32);

            return store(AstNode::Var(output_var.to_string()), index, zero_value);
        }

        // ループを生成
        let loop_var = format!("pidx{}", dim);
        let inner_body = Self::create_init_loop(output_view, output_var, _dtype, dim + 1);

        LowererUtils::create_dimension_loop(loop_var, &shape[dim], inner_body, None)
    }

    /// 入力データを出力配列の適切な位置にコピーするループを生成
    fn create_copy_loop(
        input_view: &View,
        output_view: &View,
        input_var: &str,
        output_var: &str,
        dim: usize,
    ) -> AstNode {
        let View::Linear {
            shape: input_shape,
            strides: input_strides,
            offset: input_offset,
            ..
        } = input_view;

        let View::Linear {
            strides: output_strides,
            offset: output_offset,
            pad: output_pad,
            ..
        } = output_view;

        if dim >= input_shape.len() {
            // 最内レベル: コピーを実行
            let input_index = LowererUtils::compute_memory_index(input_strides, input_offset, dim);

            // 出力インデックスはパディングを考慮する必要がある
            // 各次元のループ変数はridx0, ridx1, ... だが、
            // 出力配列へのインデックス計算時にはパディングオフセットを加える
            let output_index =
                Self::compute_padded_output_index(output_strides, output_offset, output_pad, dim);

            return store(
                AstNode::Var(output_var.to_string()),
                output_index,
                AstNode::Load {
                    target: Box::new(AstNode::Var(input_var.to_string())),
                    index: Box::new(input_index),
                    vector_width: 1,
                },
            );
        }

        // ループを生成
        let loop_var = format!("pidx{}", dim);
        let inner_body =
            Self::create_copy_loop(input_view, output_view, input_var, output_var, dim + 1);

        LowererUtils::create_dimension_loop(loop_var, &input_shape[dim], inner_body, None)
    }

    /// パディングを考慮した出力インデックスを計算
    fn compute_padded_output_index(
        strides: &[Expr],
        offset: &Expr,
        pad: &[(Expr, Expr)],
        ndim: usize,
    ) -> AstNode {
        let mut index_expr = offset.clone();

        for i in 0..ndim {
            let loop_var_expr = Expr::Var(format!("pidx{}", i));
            let stride = &strides[i];

            // パディングがある場合、前パディング分をオフセットに加算
            let adjusted_index = if i < pad.len() && (!pad[i].0.is_zero() || !pad[i].1.is_zero()) {
                let front_pad = &pad[i].0;
                loop_var_expr + front_pad.clone()
            } else {
                loop_var_expr
            };

            let term = (adjusted_index * stride.clone()).simplify();
            index_expr = (index_expr + term).simplify();
        }

        index_expr.into()
    }
}
