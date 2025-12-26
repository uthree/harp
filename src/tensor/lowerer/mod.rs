//! TensorLowerer - TensorをASTへ変換
//!
//! Tensorツリーをトラバースし、ASTプログラムを生成する。
//! 統一Compute演算を使用して全ての計算を処理。
//!
//! # 設計
//!
//! 全ての計算演算をCompute形式で統一:
//! - Elementwise: reduce_op = None, axes = []
//! - Reduce: expr = Wildcard("0"), reduce_op = Some(op), axes = [...]
//! - Fused: 任意のexpr + reduce_op
//!
//! これにより、axes=[]ならElementwiseパス、そうでなければReduceパスで処理。
//!
//! # 使用例
//!
//! ```ignore
//! use harp::tensor::{Tensor, Dim2};
//! use harp::tensor::lowerer::TensorLowerer;
//!
//! let a = Tensor::<f32, Dim2>::input("a", [2, 3]);
//! let b = Tensor::<f32, Dim2>::input("b", [2, 3]);
//! let c = &a + &b;
//!
//! let mut lowerer = TensorLowerer::new();
//! let ast = lowerer.lower(&c.clone().into_dyn());
//! ```

pub mod expr_builder;
pub mod helpers;

use std::collections::HashMap;
use std::sync::Arc;

use crate::ast::{AstKernelCallInfo, AstNode, DType as AstDType, Mutability, Scope, helper::*};
use crate::tensor::ops::{ErasedTensorInner, ReduceOp, TensorOp};
use crate::tensor::shape::Expr;
use crate::tensor::{DimDyn, Tensor, TensorInner};

use helpers::*;

/// TensorをASTに変換するLowerer
///
/// Tensorツリーをトラバースし、ASTプログラムを直接生成する。
pub struct TensorLowerer {
    /// カーネル名カウンタ
    kernel_counter: usize,
    /// 収集された入力バッファ名
    input_buffer_names: Vec<String>,
    /// 処理済みTensorInnerキャッシュ (ptr -> processed flag)
    visited: HashMap<*const TensorInner, bool>,
    /// Executedテンソルの名前マッピング (ptr -> name)
    executed_names: HashMap<*const TensorInner, String>,
    /// Executedテンソルカウンタ
    executed_counter: usize,
}

impl TensorLowerer {
    /// 新しいTensorLowererを作成
    pub fn new() -> Self {
        Self {
            kernel_counter: 0,
            input_buffer_names: Vec::new(),
            visited: HashMap::new(),
            executed_names: HashMap::new(),
            executed_counter: 0,
        }
    }

    /// TensorをASTに変換
    ///
    /// # Arguments
    /// * `tensor` - 変換するテンソル
    ///
    /// # Returns
    /// ASTプログラム
    pub fn lower(&mut self, tensor: &Tensor<f32, DimDyn>) -> AstNode {
        // 入力バッファを収集
        self.collect_input_buffers(tensor.inner.as_ref());

        // メインカーネル関数を生成
        let kernel_name = self.next_kernel_name();
        let kernel_fn = self.lower_node(&tensor.inner, &kernel_name);

        // 出力形状を取得
        let output_shape = tensor.shape().to_vec();
        let numel: usize = output_shape.iter().product();

        // Programとしてラップ
        self.wrap_as_program(vec![kernel_fn], &kernel_name, numel)
    }

    /// 入力バッファを収集
    fn collect_input_buffers(&mut self, inner: &dyn ErasedTensorInner) {
        let ptr = inner as *const dyn ErasedTensorInner as *const () as *const TensorInner;
        if self.visited.contains_key(&ptr) {
            return;
        }
        self.visited.insert(ptr, true);

        match inner.op() {
            TensorOp::Buffer { name } => {
                if !self.input_buffer_names.contains(name) {
                    self.input_buffer_names.push(name.clone());
                }
            }
            TensorOp::Executed => {
                // Executedテンソルは既存データを持つので入力バッファとして扱う
                if !self.executed_names.contains_key(&ptr) {
                    let name = format!("data{}", self.executed_counter);
                    self.executed_counter += 1;
                    self.executed_names.insert(ptr, name.clone());
                    self.input_buffer_names.push(name);
                }
            }
            _ => {
                // 再帰的に子ノードを処理
                for input in inner.op().inputs() {
                    self.collect_input_buffers(input.as_ref());
                }
            }
        }
    }

    /// 次のカーネル名を生成
    fn next_kernel_name(&mut self) -> String {
        let name = format!("kernel_{}", self.kernel_counter);
        self.kernel_counter += 1;
        name
    }

    /// TensorInnerをASTにlower
    fn lower_node(&self, inner: &Arc<TensorInner>, name: &str) -> AstNode {
        // View/Contiguousの場合は入力を辿って実際の計算ノードを処理
        // 最終的な出力形状はルートノードの形状を使用
        let (compute_inner, output_shape) = self.find_compute_node(inner);

        // 正規化形式に変換
        let (expr, reduce_op, axes) = self.normalize_op(&compute_inner);
        let compute_ndim = compute_inner.view.shape().len();
        let compute_shape = compute_inner.view.shape();

        // 出力形状が異なる場合はViewコピーを生成
        let output_ndim = output_shape.len();
        let needs_reshape = output_ndim != compute_ndim;

        if needs_reshape {
            // View経由で形状が変わる場合は、計算ノードの形状でカーネルを生成し、
            // 出力時にreshape相当の処理を行う
            // ただし、最終出力形状に合わせたループ構造にする必要がある
            self.lower_with_reshape(
                &compute_inner,
                &expr,
                reduce_op.as_ref(),
                &axes,
                compute_shape,
                &output_shape,
                name,
            )
        } else if axes.is_empty() {
            // Elementwiseパス
            self.lower_elementwise_path(&compute_inner, &expr, compute_ndim, compute_shape, name)
        } else {
            // Reduceパス
            self.lower_reduce_path(
                &compute_inner,
                &expr,
                reduce_op.as_ref().unwrap(),
                &axes,
                compute_ndim,
                name,
            )
        }
    }

    /// View/Contiguousを辿って実際の計算ノードを見つける
    fn find_compute_node(&self, inner: &Arc<TensorInner>) -> (Arc<TensorInner>, Vec<Expr>) {
        let output_shape = inner.view.shape().to_vec();

        match &inner.op {
            TensorOp::View { input } | TensorOp::Contiguous { input } => {
                // 入力を辿る（出力形状は保持）
                // InputRefからArc<TensorInner>として再取得する必要がある
                // 一時的にviewとshapeを取得して再構築
                let (_, _) = self.find_compute_node_erased(input.as_ref());
                // Note: This is a workaround - we return the current inner
                // since we can't easily convert InputRef back to Arc<TensorInner>
                (inner.clone(), output_shape)
            }
            _ => (inner.clone(), output_shape),
        }
    }

    /// View/Contiguousを辿って実際の計算ノードを見つける (ErasedTensorInner版)
    fn find_compute_node_erased(&self, inner: &dyn ErasedTensorInner) -> (Vec<Expr>, Vec<Expr>) {
        let output_shape = inner.view().shape().to_vec();

        match inner.op() {
            TensorOp::View { input } | TensorOp::Contiguous { input } => {
                let (_, compute_shape) = self.find_compute_node_erased(input.as_ref());
                (output_shape, compute_shape)
            }
            _ => (output_shape.clone(), output_shape),
        }
    }

    /// 形状変換を伴うlower
    fn lower_with_reshape(
        &self,
        inner: &Arc<TensorInner>,
        expr: &AstNode,
        reduce_op: Option<&ReduceOp>,
        axes: &[usize],
        _compute_shape: &[Expr],
        output_shape: &[Expr],
        name: &str,
    ) -> AstNode {
        // 形状変換を伴う場合、出力形状でループを生成し、
        // 入力オフセットは線形インデックスで計算

        let load_dtype = dtype_to_ast(&inner.dtype);
        let output_ndim = output_shape.len();

        if reduce_op.is_some() || !axes.is_empty() {
            // Reduce演算は形状変換と組み合わせない（現時点では未対応）
            panic!("Reduce with reshape is not supported yet");
        }

        // 各入力の式を再帰的に構築
        // ただし、線形インデックスでアクセス
        let mut mappings = HashMap::new();
        let mut buffer_index = 0;

        let inputs = inner.op().inputs();
        for (i, input) in inputs.iter().enumerate() {
            let input_node = self.build_input_expr_linear(
                (*input).as_ref(),
                output_ndim,
                output_shape,
                &mut buffer_index,
                &load_dtype,
            );
            mappings.insert(i.to_string(), input_node);
        }

        // Wildcardを置換して値式を作成
        let value_expr = expr.substitute(&mappings);

        // Store文を作成（出力形状で）
        let output_offset = build_contiguous_offset_with_shape(output_ndim, Some(output_shape));
        let store_stmt = store(var(ph::OUTPUT), output_offset, value_expr);

        // 出力形状でループ
        let body = wrap_with_loops_with_shape(output_ndim, vec![store_stmt], Some(output_shape));

        function(
            Some(name.to_string()),
            vec![],
            AstDType::Tuple(vec![]),
            body,
        )
    }

    /// 線形インデックスで入力テンソルの式を構築
    fn build_input_expr_linear(
        &self,
        input: &dyn ErasedTensorInner,
        output_ndim: usize,
        output_shape: &[Expr],
        buffer_index: &mut usize,
        load_dtype: &AstDType,
    ) -> AstNode {
        match input.op() {
            TensorOp::Const(lit) | TensorOp::ConstFill(lit) => AstNode::Const(lit.clone()),
            TensorOp::Compute {
                inputs: nested_inputs,
                expr: nested_expr,
                reduce_op: None,
                axes,
                ..
            } if axes.is_empty() => {
                // Elementwise Compute演算の場合、式を再帰的に展開
                let mut nested_mappings = HashMap::new();
                for (j, nested_input) in nested_inputs.iter().enumerate() {
                    let nested_node = self.build_input_expr_linear(
                        nested_input.as_ref(),
                        output_ndim,
                        output_shape,
                        buffer_index,
                        load_dtype,
                    );
                    nested_mappings.insert(j.to_string(), nested_node);
                }
                nested_expr.substitute(&nested_mappings)
            }
            TensorOp::View { input: view_input } => {
                // Viewの次元数とループの次元数が一致するか確認
                let view_ndim = input.view().ndim();
                if view_ndim == output_ndim {
                    // 次元数が一致する場合はストライドを使用
                    let strided_offset = build_strided_offset(input.view(), output_ndim);
                    self.build_input_expr_with_view(
                        view_input.as_ref(),
                        strided_offset,
                        buffer_index,
                        load_dtype,
                    )
                } else {
                    // 次元数が異なる場合（reshape）は線形インデックスで再帰
                    // 出力の線形インデックスがそのまま入力の線形インデックスになる
                    self.build_input_expr_linear(
                        view_input.as_ref(),
                        output_ndim,
                        output_shape,
                        buffer_index,
                        load_dtype,
                    )
                }
            }
            TensorOp::Contiguous {
                input: contiguous_input,
            } => {
                // Contiguousは入力のView情報を考慮してコピー
                // 入力がViewの場合、そのstride情報を使う
                self.build_input_expr_linear(
                    contiguous_input.as_ref(),
                    output_ndim,
                    output_shape,
                    buffer_index,
                    load_dtype,
                )
            }
            TensorOp::Buffer { .. } => {
                // Bufferからload（線形インデックス使用）
                let linear_offset = build_linear_offset_with_shape(output_ndim, output_shape);
                let idx = *buffer_index;
                *buffer_index += 1;
                load(var(ph::input(idx)), linear_offset, load_dtype.clone())
            }
            _ => {
                // その他は線形オフセットでload
                let linear_offset = build_linear_offset_with_shape(output_ndim, output_shape);
                let idx = *buffer_index;
                *buffer_index += 1;
                load(var(ph::input(idx)), linear_offset, load_dtype.clone())
            }
        }
    }

    /// View経由でBufferにアクセスする式を構築
    fn build_input_expr_with_view(
        &self,
        input: &dyn ErasedTensorInner,
        offset: AstNode,
        buffer_index: &mut usize,
        load_dtype: &AstDType,
    ) -> AstNode {
        match input.op() {
            TensorOp::Buffer { .. } => {
                let idx = *buffer_index;
                *buffer_index += 1;
                load(var(ph::input(idx)), offset, load_dtype.clone())
            }
            TensorOp::View { input: nested } | TensorOp::Contiguous { input: nested } => {
                // さらにViewがネストしている場合は入力を辿る
                self.build_input_expr_with_view(nested.as_ref(), offset, buffer_index, load_dtype)
            }
            _ => {
                let idx = *buffer_index;
                *buffer_index += 1;
                load(var(ph::input(idx)), offset, load_dtype.clone())
            }
        }
    }

    /// TensorOpを正規化形式に変換
    ///
    /// Returns: (expr, reduce_op, axes)
    fn normalize_op(&self, inner: &Arc<TensorInner>) -> (AstNode, Option<ReduceOp>, Vec<usize>) {
        match &inner.op {
            // 統一Compute演算
            TensorOp::Compute {
                expr,
                reduce_op,
                axes,
                ..
            } => (expr.clone(), *reduce_op, axes.clone()),

            TensorOp::ConstFill(lit) => (AstNode::Const(lit.clone()), None, vec![]),

            TensorOp::Rand => {
                // Rand演算は特殊な式としてrand()を使用
                (rand(), None, vec![])
            }

            TensorOp::Arange => {
                // Arangeは単一の軸に対してインデックスを返す
                // 0次元目のインデックスを使用
                (var(ph::ridx(0)), None, vec![])
            }

            // View, Reshape, Contiguous等は「コピー」演算として処理
            // 入力要素をそのまま出力にコピーする
            TensorOp::View { .. } | TensorOp::Contiguous { .. } => {
                // identity copy: Wildcard("0") を使用して入力をそのまま出力
                (wildcard("0"), None, vec![])
            }

            // Bufferは入力のコピーとして処理
            TensorOp::Buffer { .. } => (wildcard("0"), None, vec![]),

            // 未対応の演算
            _ => {
                panic!("Cannot normalize op: {:?}", inner.op);
            }
        }
    }

    /// 入力テンソルの式を再帰的に構築
    ///
    /// Compute演算の入力を辿り、Bufferに達するまで式を展開する。
    /// 入力インデックスを適切に再マッピングする。
    fn build_input_expr(
        &self,
        input: &dyn ErasedTensorInner,
        ndim: usize,
        buffer_index: &mut usize,
        load_dtype: &AstDType,
    ) -> AstNode {
        match input.op() {
            TensorOp::Const(lit) | TensorOp::ConstFill(lit) => {
                // 定数は直接埋め込み
                AstNode::Const(lit.clone())
            }
            TensorOp::Compute {
                inputs: nested_inputs,
                expr: nested_expr,
                reduce_op: None,
                axes,
                ..
            } if axes.is_empty() => {
                // Elementwise Compute演算の場合、式を再帰的に展開
                let mut nested_mappings = HashMap::new();
                for (j, nested_input) in nested_inputs.iter().enumerate() {
                    let nested_node = self.build_input_expr(
                        nested_input.as_ref(),
                        ndim,
                        buffer_index,
                        load_dtype,
                    );
                    nested_mappings.insert(j.to_string(), nested_node);
                }
                nested_expr.substitute(&nested_mappings)
            }
            TensorOp::Buffer { .. } | TensorOp::View { .. } | TensorOp::Contiguous { .. } => {
                // Bufferまたはメモリ操作の場合、loadを生成
                let src_offset = self.build_input_offset(input, ndim);
                let idx = *buffer_index;
                *buffer_index += 1;
                load(var(ph::input(idx)), src_offset, load_dtype.clone())
            }
            _ => {
                // その他の演算（Reduce結果など）もloadで処理
                let src_offset = self.build_input_offset(input, ndim);
                let idx = *buffer_index;
                *buffer_index += 1;
                load(var(ph::input(idx)), src_offset, load_dtype.clone())
            }
        }
    }

    /// Elementwiseパスでlower
    fn lower_elementwise_path(
        &self,
        inner: &Arc<TensorInner>,
        expr: &AstNode,
        ndim: usize,
        shape: &[Expr],
        name: &str,
    ) -> AstNode {
        let load_dtype = dtype_to_ast(&inner.dtype);

        // 各入力の式を再帰的に構築
        let mut mappings = HashMap::new();
        let mut buffer_index = 0;

        let inputs = inner.op().inputs();
        for (i, input) in inputs.iter().enumerate() {
            let input_node =
                self.build_input_expr(input.as_ref(), ndim, &mut buffer_index, &load_dtype);
            mappings.insert(i.to_string(), input_node);
        }

        // Wildcardを置換して値式を作成
        let value_expr = expr.substitute(&mappings);

        // Store文を作成
        let output_offset = build_contiguous_offset_with_shape(ndim, Some(shape));
        let store_stmt = store(var(ph::OUTPUT), output_offset, value_expr);

        // ループでラップ
        let body = wrap_with_loops_with_shape(ndim, vec![store_stmt], Some(shape));

        function(
            Some(name.to_string()),
            vec![],
            AstDType::Tuple(vec![]),
            body,
        )
    }

    /// Reduceパスでlower
    fn lower_reduce_path(
        &self,
        inner: &Arc<TensorInner>,
        expr: &AstNode,
        reduce_op: &ReduceOp,
        axes: &[usize],
        _ndim: usize,
        name: &str,
    ) -> AstNode {
        // 入力ノードの形状を取得
        let inputs = inner.op().inputs();
        let input_shape = if let Some(input) = inputs.first() {
            input.view().shape()
        } else {
            inner.view.shape()
        };

        // 入力テンソルの次元数を使用（出力テンソルの次元数ではなく）
        let input_ndim = input_shape.len();

        let load_dtype = dtype_to_ast(&inner.dtype);

        // 各入力のload式を構築（elementwise Computeは展開）
        let mut mappings = HashMap::new();
        let mut buffer_index = 0;

        for (i, input) in inputs.iter().enumerate() {
            let input_node =
                self.build_input_expr(input.as_ref(), input_ndim, &mut buffer_index, &load_dtype);
            mappings.insert(i.to_string(), input_node);
        }

        let value_expr = expr.substitute(&mappings);

        // アキュムレータ初期化と更新
        let (init_value, accumulate_fn) = build_reduce_accumulator(reduce_op, &inner.dtype);

        // 出力オフセット（縮約軸を除く）
        let output_offset =
            build_contiguous_offset_excluding_axes_with_shape(input_ndim, axes, Some(input_shape));

        let acc_var = "acc";
        let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));

        // Reduce軸のループを生成
        let mut reduce_loops = block(vec![acc_update], Scope::new());
        for &axis in axes.iter().rev() {
            reduce_loops = range(
                ph::ridx(axis),
                const_int(0),
                const_int(1),
                shape_dim_to_ast(Some(input_shape), axis),
                reduce_loops,
            );
        }

        // アキュムレータ変数を宣言
        let mut scope = Scope::new();
        let _ = scope.declare(
            acc_var.to_string(),
            dtype_to_ast(&inner.dtype),
            Mutability::Mutable,
        );

        let acc_init = assign(acc_var, init_value);
        let store_stmt = store(var(ph::OUTPUT), output_offset, var(acc_var));

        let inner_body = vec![acc_init, reduce_loops, store_stmt];
        let body = wrap_with_loops_excluding_axes_with_scope_and_shape(
            input_ndim,
            axes,
            inner_body,
            scope,
            Some(input_shape),
        );

        function(
            Some(name.to_string()),
            vec![],
            AstDType::Tuple(vec![]),
            body,
        )
    }

    /// 入力テンソルのオフセットを構築
    ///
    /// ブロードキャストを考慮：入力の次元数がループ次元数より小さい場合、
    /// 先頭の軸を無視してオフセットを計算する
    fn build_input_offset(&self, src: &dyn ErasedTensorInner, loop_ndim: usize) -> AstNode {
        let src_ndim = src.view().shape().len();

        if src_ndim == loop_ndim {
            // 次元数が同じ場合はそのまま
            self.build_offset_for_tensor(src, loop_ndim)
        } else if src_ndim < loop_ndim {
            // ブロードキャスト: 入力の次元数 < ループの次元数
            // 入力は後ろの軸にマッピング（先頭の軸は無視）
            self.build_broadcast_offset(src, src_ndim, loop_ndim)
        } else {
            // src_ndim > loop_ndim: 通常発生しないが、念のためそのまま
            self.build_offset_for_tensor(src, src_ndim)
        }
    }

    /// ブロードキャスト用オフセットを構築
    ///
    /// 入力の次元数 < ループの次元数の場合、
    /// 入力は最後のsrc_ndim軸にマッピングされる
    fn build_broadcast_offset(
        &self,
        inner: &dyn ErasedTensorInner,
        src_ndim: usize,
        loop_ndim: usize,
    ) -> AstNode {
        let axis_offset = loop_ndim - src_ndim;
        let shape = inner.view().shape();

        // 入力の各軸を、ループの対応する軸にマッピング
        // 例: 入力[3]、ループ[2,3] → ridx1のみ使用
        if src_ndim == 0 {
            return const_int(0);
        }

        // strideを計算（後ろからの連続アクセス）
        let mut offset = var(ph::ridx(axis_offset + src_ndim - 1));

        for axis in (0..src_ndim - 1).rev() {
            let loop_axis = axis_offset + axis;
            let mut stride = shape_dim_to_ast(Some(shape), axis + 1);
            for inner_axis in (axis + 2)..src_ndim {
                stride = stride * shape_dim_to_ast(Some(shape), inner_axis);
            }
            offset = var(ph::ridx(loop_axis)) * stride + offset;
        }

        offset
    }

    /// TensorInnerのオフセットを構築
    fn build_offset_for_tensor(&self, inner: &dyn ErasedTensorInner, ndim: usize) -> AstNode {
        match inner.op() {
            TensorOp::Buffer { .. } => {
                // Bufferノードは自身のviewを使用
                build_strided_offset(inner.view(), ndim)
            }
            TensorOp::View { .. } => {
                // Viewノードは自身のviewを使用
                build_strided_offset(inner.view(), ndim)
            }
            TensorOp::Contiguous { input } => {
                // Contiguousは入力のオフセットを使用（まだ実体化されていない）
                // 入力がViewの場合、そのstride情報を使ってアクセス
                self.build_offset_for_tensor(input.as_ref(), ndim)
            }
            TensorOp::Compute { .. } => {
                // Compute演算の結果は連続メモリ
                let shape = inner.view().shape();
                build_contiguous_offset_with_shape(ndim, Some(shape))
            }
            _ => {
                // その他の演算は自身のviewを使用
                build_strided_offset(inner.view(), ndim)
            }
        }
    }

    /// Programとしてラップ
    fn wrap_as_program(&self, functions: Vec<AstNode>, kernel_name: &str, numel: usize) -> AstNode {
        // 入力バッファをソート（安定した順序のため）
        let mut input_names = self.input_buffer_names.clone();
        input_names.sort();

        // スレッドグループサイズを計算
        let local_size = 64.min(numel);
        let grid_size = numel.div_ceil(local_size);

        // カーネル呼び出し情報
        let call_info = AstKernelCallInfo {
            kernel_name: kernel_name.to_string(),
            inputs: input_names.clone(),
            outputs: vec!["output".to_string()],
            grid_size: [
                Expr::from(grid_size as i64),
                Expr::from(1i64),
                Expr::from(1i64),
            ],
            local_size: [
                Expr::from(local_size as i64),
                Expr::from(1i64),
                Expr::from(1i64),
            ],
        };

        AstNode::Program {
            functions,
            execution_waves: vec![vec![call_info]],
        }
    }
}

impl Default for TensorLowerer {
    fn default() -> Self {
        Self::new()
    }
}

/// TensorをASTに変換する簡易関数
pub fn lower_tensor(tensor: &Tensor<f32, DimDyn>) -> AstNode {
    let mut lowerer = TensorLowerer::new();
    lowerer.lower(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dim2, Recip, Sqrt};

    #[test]
    fn test_lower_simple_add() {
        let a = Tensor::<f32, Dim2>::input("a", [2, 3]);
        let b = Tensor::<f32, Dim2>::input("b", [2, 3]);
        let c = &a + &b;

        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&c.clone().into_dyn());

        // ASTがProgramであることを確認
        match ast {
            AstNode::Program { functions, .. } => {
                assert!(
                    !functions.is_empty(),
                    "Program should have at least one function"
                );
            }
            _ => panic!("Expected AstNode::Program"),
        }
    }

    #[test]
    fn test_lower_fused_operations() {
        let a = Tensor::<f32, Dim2>::input("a", [4, 4]);
        let b = a.recip().sqrt(); // Fused: recip -> sqrt

        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&b.clone().into_dyn());

        match ast {
            AstNode::Program { functions, .. } => {
                assert!(!functions.is_empty());
            }
            _ => panic!("Expected AstNode::Program"),
        }
    }

    #[test]
    fn test_lower_reduce() {
        let a = Tensor::<f32, Dim2>::input("a", [4, 4]);
        let b = a.reduce_sum(&[1], false);

        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&b);

        match ast {
            AstNode::Program { functions, .. } => {
                assert!(!functions.is_empty());
            }
            _ => panic!("Expected AstNode::Program"),
        }
    }

    #[test]
    fn test_lower_const_fill() {
        let a = Tensor::<f32, Dim2>::full([2, 3], 1.0);

        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&a.into_dyn());

        match ast {
            AstNode::Program { functions, .. } => {
                assert!(!functions.is_empty());
            }
            _ => panic!("Expected AstNode::Program"),
        }
    }
}
