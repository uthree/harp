//! TensorLowerer - TensorをASTへ変換
//!
//! Tensorツリーをトラバースし、ASTプログラムを生成する。
//! 統一MapReduce演算を使用して全ての計算を処理。
//!
//! # 設計
//!
//! 全ての計算演算をMapReduce形式で統一:
//! - Elementwise: reduce_op = None, axes = []
//! - Reduce: expr = Wildcard("0"), reduce_op = Some(op), axes = [...]
//! - Fused: 任意のexpr + reduce_op
//!
//! これにより、axes=[]ならElementwiseパス、そうでなければReduceパスで処理。
//!
//! # 使用例
//!
//! ```ignore
//! use harp_core::tensor::{Tensor, Dim2};
//! use harp_core::tensor::lowerer::TensorLowerer;
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

use crate::ast::{AstKernelCallInfo, AstNode, DType as AstDType, Mutability, Scope, helper::*};
use crate::tensor::ops::{ReduceOp, TensorOp};
use crate::tensor::shape::Expr;
use crate::tensor::shape::View;
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
    /// テンソルポインタ -> バッファインデックスのマッピング
    /// 同じテンソルが複数回参照される場合に同じインデックスを再利用するため
    buffer_index_map: HashMap<*const TensorInner, usize>,
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
            buffer_index_map: HashMap::new(),
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
        self.lower_inner(tensor.inner.as_ref())
    }

    /// TensorInnerからASTに変換（内部用）
    ///
    /// realize_core処理で使用するため、TensorInnerを直接受け取る版
    ///
    /// # Arguments
    /// * `inner` - 変換するテンソル内部表現
    ///
    /// # Returns
    /// ASTプログラム
    pub fn lower_inner(&mut self, inner: &TensorInner) -> AstNode {
        // 入力バッファを収集
        self.collect_input_buffers(inner);

        // メインカーネル関数を生成
        let kernel_name = self.next_kernel_name();
        let kernel_fn = self.lower_node_erased(inner, &kernel_name);

        // 出力形状を取得
        let output_shape = inner.view().shape().to_vec();
        let numel: usize = output_shape
            .iter()
            .map(|e| e.as_const().unwrap_or(1) as usize)
            .product();

        // Programとしてラップ
        self.wrap_as_program(vec![kernel_fn], &kernel_name, numel)
    }

    /// 入力バッファを収集
    fn collect_input_buffers(&mut self, inner: &TensorInner) {
        let ptr = inner as *const TensorInner;
        if self.visited.contains_key(&ptr) {
            return;
        }
        self.visited.insert(ptr, true);

        // バッファを持つノードは入力バッファとして扱う（Compute含む）
        // これにより collect_input_data_inner と同じ順序で入力を収集する
        if inner.has_buffer() {
            if !self.executed_names.contains_key(&ptr) {
                let name = format!("data{}", self.executed_counter);
                self.executed_counter += 1;
                self.executed_names.insert(ptr, name.clone());
                self.input_buffer_names.push(name);
            }
            return;
        }

        match inner.op() {
            TensorOp::Buffer { name } => {
                if !self.input_buffer_names.contains(name) {
                    self.input_buffer_names.push(name.clone());
                }
            }
            TensorOp::Executed => {
                // Executedテンソルは既存データを持つので入力バッファとして扱う
                // （通常は上のhas_buffer()で処理されるが、バッファがない場合もある）
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

    /// TensorInnerをASTにlower（内部用）
    fn lower_node_erased(&mut self, inner: &TensorInner, name: &str) -> AstNode {
        // View::Maskedは特別処理（Expr条件分岐が必要）
        // Note: View::padded()もMasked Viewを返すようになったため、ここで処理される
        if let TensorOp::View { input } = inner.op()
            && let View::Masked {
                condition,
                default_value,
                ..
            } = inner.view()
        {
            return self.lower_masked_erased(
                inner,
                input.as_ref(),
                condition,
                *default_value,
                name,
            );
        }

        // Concatは特別処理（複数入力から条件分岐で選択）
        if let TensorOp::Concat { inputs, axis } = inner.op() {
            return self.lower_concat_erased(inner, inputs, *axis, name);
        }

        // 最終的な出力形状はルートノードの形状を使用
        let output_shape = inner.view().shape().to_vec();
        let output_ndim = output_shape.len();

        // compute_shape を決定
        // - Contiguous: 出力形状を使用（新しいcontiguousバッファを作成するため）
        // - View: 入力の compute_shape を使用
        // - その他: 出力形状を使用
        let compute_shape = match inner.op() {
            TensorOp::Contiguous { .. } => {
                // Contiguousは出力形状でループする（新しいcontiguousバッファを作成）
                output_shape.clone()
            }
            TensorOp::View { input } => {
                let (_, compute_shape) = self.find_compute_node_erased(input.as_ref());
                compute_shape
            }
            _ => output_shape.clone(),
        };
        let compute_ndim = compute_shape.len();

        // 正規化形式に変換
        let (expr, reduce_op, axes) = self.normalize_op_erased(inner);

        // 出力形状が異なる場合はViewコピーを生成
        let needs_reshape = output_ndim != compute_ndim;

        if needs_reshape {
            // View経由で形状が変わる場合は、計算ノードの形状でカーネルを生成し、
            // 出力時にreshape相当の処理を行う
            // ただし、最終出力形状に合わせたループ構造にする必要がある
            self.lower_with_reshape_erased(
                inner,
                &expr,
                reduce_op.as_ref(),
                &axes,
                &compute_shape,
                &output_shape,
                name,
            )
        } else if axes.is_empty() {
            // Elementwiseパス
            self.lower_elementwise_path_erased(inner, &expr, compute_ndim, &compute_shape, name)
        } else {
            // Reduceパス
            self.lower_reduce_path_erased(
                inner,
                &expr,
                reduce_op.as_ref().unwrap(),
                &axes,
                compute_ndim,
                name,
            )
        }
    }

    /// View/Contiguousを辿って実際の計算ノードを見つける (TensorInner版)
    fn find_compute_node_erased(&self, inner: &TensorInner) -> (Vec<Expr>, Vec<Expr>) {
        let output_shape = inner.view().shape().to_vec();

        match inner.op() {
            TensorOp::View { input } | TensorOp::Contiguous { input } => {
                let (_, compute_shape) = self.find_compute_node_erased(input.as_ref());
                (output_shape, compute_shape)
            }
            _ => (output_shape.clone(), output_shape),
        }
    }

    /// 形状変換を伴うlower（TensorInner版）
    #[allow(clippy::too_many_arguments)]
    fn lower_with_reshape_erased(
        &mut self,
        inner: &TensorInner,
        expr: &AstNode,
        reduce_op: Option<&ReduceOp>,
        axes: &[usize],
        _compute_shape: &[Expr],
        output_shape: &[Expr],
        name: &str,
    ) -> AstNode {
        // 形状変換を伴う場合、出力形状でループを生成し、
        // 入力オフセットは線形インデックスで計算

        let load_dtype = dtype_to_ast(&inner.dtype());
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
        &mut self,
        input: &TensorInner,
        output_ndim: usize,
        output_shape: &[Expr],
        buffer_index: &mut usize,
        load_dtype: &AstDType,
    ) -> AstNode {
        let ptr = input as *const TensorInner;

        // バッファを持つノードは入力バッファとして扱う（realized済みCompute含む）
        // バッファのデータは常にcontiguousなので、適切なオフセット計算を使用
        if input.has_buffer() {
            let input_shape = input.view().shape();
            let input_ndim = input_shape.len();

            let src_offset = if input_ndim == output_ndim {
                // 次元数が同じ場合はcontiguousオフセット
                build_contiguous_offset_with_shape(output_ndim, Some(&input_shape))
            } else if input_ndim < output_ndim {
                // ブロードキャスト: 入力の次元数 < ループの次元数
                self.build_broadcast_offset(input, input_ndim, output_ndim)
            } else {
                // input_ndim > output_ndim: 入力のshapeを使用
                build_contiguous_offset_with_shape(input_ndim, Some(&input_shape))
            };

            // 既にこのテンソルにインデックスが割り当てられているか確認
            let idx = if let Some(&existing_idx) = self.buffer_index_map.get(&ptr) {
                existing_idx
            } else {
                let idx = *buffer_index;
                *buffer_index += 1;
                self.buffer_index_map.insert(ptr, idx);
                idx
            };
            return load(var(ph::input(idx)), src_offset, load_dtype.clone());
        }

        match input.op() {
            TensorOp::Const(lit) | TensorOp::ConstFill(lit) => AstNode::Const(lit.clone()),
            TensorOp::MapReduce {
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
                    // 次元数が異なる場合（reshape）は線形インデックスを計算して直接アクセス
                    // 出力の線形インデックスがそのまま入力の線形インデックスになる
                    let linear_offset = build_linear_offset_with_shape(output_ndim, output_shape);
                    self.build_input_expr_with_view(
                        view_input.as_ref(),
                        linear_offset,
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
                // 既にこのテンソルにインデックスが割り当てられているか確認
                let idx = if let Some(&existing_idx) = self.buffer_index_map.get(&ptr) {
                    existing_idx
                } else {
                    let idx = *buffer_index;
                    *buffer_index += 1;
                    self.buffer_index_map.insert(ptr, idx);
                    idx
                };
                load(var(ph::input(idx)), linear_offset, load_dtype.clone())
            }
            _ => {
                // その他は線形オフセットでload
                let linear_offset = build_linear_offset_with_shape(output_ndim, output_shape);
                // 既にこのテンソルにインデックスが割り当てられているか確認
                let idx = if let Some(&existing_idx) = self.buffer_index_map.get(&ptr) {
                    existing_idx
                } else {
                    let idx = *buffer_index;
                    *buffer_index += 1;
                    self.buffer_index_map.insert(ptr, idx);
                    idx
                };
                load(var(ph::input(idx)), linear_offset, load_dtype.clone())
            }
        }
    }

    /// View経由でBufferにアクセスする式を構築
    fn build_input_expr_with_view(
        &mut self,
        input: &TensorInner,
        offset: AstNode,
        buffer_index: &mut usize,
        load_dtype: &AstDType,
    ) -> AstNode {
        let ptr = input as *const TensorInner;

        // バッファを持つノードは入力バッファとして扱う（realized済みCompute含む）
        // バッファのデータは常にcontiguousなので、contiguousオフセットを使用
        if input.has_buffer() {
            let input_shape = input.view().shape();
            let input_ndim = input_shape.len();
            // contiguousオフセットを計算（バッファはcontiguousデータを持つ）
            let src_offset = build_contiguous_offset_with_shape(input_ndim, Some(&input_shape));
            // 既にこのテンソルにインデックスが割り当てられているか確認
            let idx = if let Some(&existing_idx) = self.buffer_index_map.get(&ptr) {
                existing_idx
            } else {
                let idx = *buffer_index;
                *buffer_index += 1;
                self.buffer_index_map.insert(ptr, idx);
                idx
            };
            return load(var(ph::input(idx)), src_offset, load_dtype.clone());
        }

        match input.op() {
            TensorOp::Const(lit) | TensorOp::ConstFill(lit) => {
                // 定数は直接埋め込み（他のbuild_input_expr_*と同様）
                AstNode::Const(lit.clone())
            }
            TensorOp::Buffer { .. } => {
                // 既にこのテンソルにインデックスが割り当てられているか確認
                let idx = if let Some(&existing_idx) = self.buffer_index_map.get(&ptr) {
                    existing_idx
                } else {
                    let idx = *buffer_index;
                    *buffer_index += 1;
                    self.buffer_index_map.insert(ptr, idx);
                    idx
                };
                load(var(ph::input(idx)), offset, load_dtype.clone())
            }
            TensorOp::View { input: nested } | TensorOp::Contiguous { input: nested } => {
                // さらにViewがネストしている場合は入力を辿る
                self.build_input_expr_with_view(nested.as_ref(), offset, buffer_index, load_dtype)
            }
            _ => {
                // 既にこのテンソルにインデックスが割り当てられているか確認
                let idx = if let Some(&existing_idx) = self.buffer_index_map.get(&ptr) {
                    existing_idx
                } else {
                    let idx = *buffer_index;
                    *buffer_index += 1;
                    self.buffer_index_map.insert(ptr, idx);
                    idx
                };
                load(var(ph::input(idx)), offset, load_dtype.clone())
            }
        }
    }

    /// TensorOpを正規化形式に変換（TensorInner版）
    ///
    /// Returns: (expr, reduce_op, axes)
    fn normalize_op_erased(&self, inner: &TensorInner) -> (AstNode, Option<ReduceOp>, Vec<usize>) {
        match inner.op() {
            // 統一Compute演算
            TensorOp::MapReduce {
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
            op => {
                panic!("Cannot normalize op: {:?}", op);
            }
        }
    }

    /// 入力テンソルの式を再帰的に構築
    ///
    /// Compute演算の入力を辿り、Bufferに達するまで式を展開する。
    /// 入力インデックスを適切に再マッピングする。
    ///
    /// 重要: 同じテンソルが複数回参照される場合（例: a + a）、
    /// 同じバッファインデックスを再利用する。これにより、カーネル引数の数と
    /// 実際に渡されるバッファの数が一致する。
    fn build_input_expr(
        &mut self,
        input: &TensorInner,
        ndim: usize,
        buffer_index: &mut usize,
        load_dtype: &AstDType,
    ) -> AstNode {
        let ptr = input as *const TensorInner;

        // バッファを持つノードは入力バッファとして扱う（realized済みCompute含む）
        // バッファのデータは常にcontiguousなので、適切なオフセット計算を使用
        if input.has_buffer() {
            let input_shape = input.view().shape();
            let input_ndim = input_shape.len();

            let src_offset = if input_ndim == ndim {
                // 次元数が同じ場合はcontiguousオフセット
                build_contiguous_offset_with_shape(ndim, Some(&input_shape))
            } else if input_ndim < ndim {
                // ブロードキャスト: 入力の次元数 < ループの次元数
                self.build_broadcast_offset(input, input_ndim, ndim)
            } else {
                // input_ndim > ndim: 通常発生しないが、入力のshapeを使用
                build_contiguous_offset_with_shape(input_ndim, Some(&input_shape))
            };

            // 既にこのテンソルにインデックスが割り当てられているか確認
            let idx = if let Some(&existing_idx) = self.buffer_index_map.get(&ptr) {
                existing_idx
            } else {
                let idx = *buffer_index;
                *buffer_index += 1;
                self.buffer_index_map.insert(ptr, idx);
                idx
            };
            return load(var(ph::input(idx)), src_offset, load_dtype.clone());
        }

        match input.op() {
            TensorOp::Const(lit) | TensorOp::ConstFill(lit) => {
                // 定数は直接埋め込み
                AstNode::Const(lit.clone())
            }
            TensorOp::MapReduce {
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
            TensorOp::View { input: view_input } | TensorOp::Contiguous { input: view_input } => {
                // Viewチェーンの最終入力がインライン化可能かチェック
                // - ConstFill/Const: 定数として埋め込み
                // - Fusable MapReduce: 再帰的に展開
                // - バッファを持つノード/Buffer/Executed: バッファからload
                if Self::is_inlineable_in_view(view_input.as_ref()) {
                    // 再帰的に入力を処理（ConstFillやfusable MapReduceを展開）
                    self.build_input_expr(view_input.as_ref(), ndim, buffer_index, load_dtype)
                } else {
                    // インライン化できない場合はバッファからloadとして扱う
                    let src_offset = self.build_input_offset(input, ndim);
                    let idx = if let Some(&existing_idx) = self.buffer_index_map.get(&ptr) {
                        existing_idx
                    } else {
                        let idx = *buffer_index;
                        *buffer_index += 1;
                        self.buffer_index_map.insert(ptr, idx);
                        idx
                    };
                    load(var(ph::input(idx)), src_offset, load_dtype.clone())
                }
            }
            TensorOp::Buffer { .. } => {
                // Bufferからload
                let src_offset = self.build_input_offset(input, ndim);
                // 既にこのテンソルにインデックスが割り当てられているか確認
                let idx = if let Some(&existing_idx) = self.buffer_index_map.get(&ptr) {
                    existing_idx
                } else {
                    let idx = *buffer_index;
                    *buffer_index += 1;
                    self.buffer_index_map.insert(ptr, idx);
                    idx
                };
                load(var(ph::input(idx)), src_offset, load_dtype.clone())
            }
            _ => {
                // その他の演算（Reduce結果など）もloadで処理
                let src_offset = self.build_input_offset(input, ndim);
                // 既にこのテンソルにインデックスが割り当てられているか確認
                let idx = if let Some(&existing_idx) = self.buffer_index_map.get(&ptr) {
                    existing_idx
                } else {
                    let idx = *buffer_index;
                    *buffer_index += 1;
                    self.buffer_index_map.insert(ptr, idx);
                    idx
                };
                load(var(ph::input(idx)), src_offset, load_dtype.clone())
            }
        }
    }

    /// Elementwiseパスでlower（TensorInner版）
    fn lower_elementwise_path_erased(
        &mut self,
        inner: &TensorInner,
        expr: &AstNode,
        ndim: usize,
        shape: &[Expr],
        name: &str,
    ) -> AstNode {
        let load_dtype = dtype_to_ast(&inner.dtype());

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

    /// Reduceパスでlower（TensorInner版）
    fn lower_reduce_path_erased(
        &mut self,
        inner: &TensorInner,
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
            inner.view().shape()
        };

        // 入力テンソルの次元数を使用（出力テンソルの次元数ではなく）
        let input_ndim = input_shape.len();

        let load_dtype = dtype_to_ast(&inner.dtype());

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
        let (init_value, accumulate_fn) = build_reduce_accumulator(reduce_op, &inner.dtype());

        // 出力オフセット（縮約軸を除く）
        let output_offset =
            build_contiguous_offset_excluding_axes_with_shape(input_ndim, axes, Some(&input_shape));

        let acc_var = "acc";
        let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));

        // Reduce軸のループを生成
        let mut reduce_loops = block(vec![acc_update], Scope::new());
        for &axis in axes.iter().rev() {
            reduce_loops = range(
                ph::ridx(axis),
                const_int(0),
                const_int(1),
                shape_dim_to_ast(Some(&input_shape), axis),
                reduce_loops,
            );
        }

        // アキュムレータ変数を宣言
        let mut scope = Scope::new();
        let _ = scope.declare(
            acc_var.to_string(),
            dtype_to_ast(&inner.dtype()),
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
            Some(&input_shape),
        );

        function(
            Some(name.to_string()),
            vec![],
            AstDType::Tuple(vec![]),
            body,
        )
    }

    /// View::Masked演算をlower
    ///
    /// Expr条件に基づいて、条件が非0なら入力からロード、0ならdefault_valueを出力
    fn lower_masked_erased(
        &self,
        outer: &TensorInner,
        input: &TensorInner,
        condition: &crate::tensor::shape::Expr,
        default_value: crate::tensor::ops::PadValue,
        name: &str,
    ) -> AstNode {
        let output_shape = outer.view().shape();
        let ndim = output_shape.len();
        let load_dtype = dtype_to_ast(&outer.dtype());

        // デフォルト値をリテラルに変換
        let default_lit = match default_value {
            crate::tensor::ops::PadValue::Zero => AstNode::Const(crate::ast::Literal::F32(0.0)),
            crate::tensor::ops::PadValue::One => AstNode::Const(crate::ast::Literal::F32(1.0)),
            crate::tensor::ops::PadValue::NegInf => {
                AstNode::Const(crate::ast::Literal::F32(f32::NEG_INFINITY))
            }
        };

        // 出力オフセット（contiguous）
        let output_offset = build_contiguous_offset_with_shape(ndim, Some(&output_shape));

        // 条件式をAstNodeに変換
        // Expr::Idx(n) は ridx{n} に変換される
        let condition_ast: AstNode = condition.clone().into();

        // 条件が非0かどうかをチェック（boolではなく整数なのでNe(0)で比較）
        // condition != 0 → !(condition < 1 && 1 < condition + 1)
        // 簡略化: condition をそのままSelect条件として使う（0は偽、非0は真）
        // AstNode::Selectはbool型を期待するので、Ne(condition, 0) を使う
        let cond_bool = ne(condition_ast, const_int(0));

        // 入力オフセットを計算（inner viewを考慮）
        let inner_view = input.view();
        let input_offset = build_strided_offset(inner_view, ndim);

        // 入力バッファからのload
        let load_expr = load(var(ph::input(0)), input_offset, load_dtype);

        // 条件分岐: 条件が真ならload、偽ならdefault_value
        let value_expr = AstNode::Select {
            cond: Box::new(cond_bool),
            then_val: Box::new(load_expr),
            else_val: Box::new(default_lit),
        };

        // Store文を作成
        let store_stmt = store(var(ph::OUTPUT), output_offset, value_expr);

        // ループでラップ
        let body = wrap_with_loops_with_shape(ndim, vec![store_stmt], Some(&output_shape));

        function(
            Some(name.to_string()),
            vec![],
            AstDType::Tuple(vec![]),
            body,
        )
    }

    /// Concat演算をlower
    ///
    /// 複数の入力テンソルをaxis軸に沿って結合
    /// 条件分岐で適切な入力からロードする
    fn lower_concat_erased(
        &self,
        outer: &TensorInner,
        inputs: &[crate::tensor::ops::InputRef],
        axis: usize,
        name: &str,
    ) -> AstNode {
        let output_shape = outer.view().shape();
        let ndim = output_shape.len();
        let load_dtype = dtype_to_ast(&outer.dtype());

        // 出力オフセット（contiguous）
        let output_offset = build_contiguous_offset_with_shape(ndim, Some(&output_shape));

        // 各入力テンソルのaxis方向のサイズと累積オフセットを計算
        let mut cumulative_offset: i64 = 0;
        let mut boundaries: Vec<i64> = Vec::new();

        for input in inputs.iter() {
            let input_shape = input.view().shape();
            let axis_size = input_shape[axis].expect_const("concat input axis size must be const");
            cumulative_offset += axis_size;
            boundaries.push(cumulative_offset);
        }

        // 条件分岐を構築（後ろから構築してネスト）
        // if ridx[axis] < boundary[0]: load from input[0]
        // else if ridx[axis] < boundary[1]: load from input[1] (offset adjusted)
        // ...

        // まず最後の入力のロードを構築（else節）
        let last_idx = inputs.len() - 1;
        let last_offset_adj = if last_idx > 0 {
            boundaries[last_idx - 1]
        } else {
            0
        };
        let mut result_expr = self.build_concat_load(
            &inputs[last_idx],
            axis,
            ndim,
            last_offset_adj,
            &load_dtype,
            last_idx,
        );

        // 残りの入力について条件分岐を追加（後ろから前へ）
        for i in (0..last_idx).rev() {
            let offset_adj = if i > 0 { boundaries[i - 1] } else { 0 };
            let boundary = boundaries[i];

            // 条件: ridx[axis] < boundary
            let cond = AstNode::Lt(Box::new(var(ph::ridx(axis))), Box::new(const_int(boundary)));

            // then節: この入力からロード
            let then_expr =
                self.build_concat_load(&inputs[i], axis, ndim, offset_adj, &load_dtype, i);

            // Select文で条件分岐
            result_expr = AstNode::Select {
                cond: Box::new(cond),
                then_val: Box::new(then_expr),
                else_val: Box::new(result_expr),
            };
        }

        // Store文を作成
        let store_stmt = store(var(ph::OUTPUT), output_offset, result_expr);

        // ループでラップ
        let body = wrap_with_loops_with_shape(ndim, vec![store_stmt], Some(&output_shape));

        function(
            Some(name.to_string()),
            vec![],
            AstDType::Tuple(vec![]),
            body,
        )
    }

    /// Concat用の入力ロード式を構築
    fn build_concat_load(
        &self,
        input: &crate::tensor::ops::InputRef,
        axis: usize,
        ndim: usize,
        offset_adjustment: i64,
        load_dtype: &AstDType,
        input_idx: usize,
    ) -> AstNode {
        let input_shape = input.view().shape();

        // 入力オフセットを計算
        // ridx[i] をそのまま使用、ただしaxis方向は offset_adjustment を引く
        let mut input_offset = const_int(0);

        for dim in (0..ndim).rev() {
            let idx = if dim == axis && offset_adjustment != 0 {
                var(ph::ridx(dim)) - const_int(offset_adjustment)
            } else {
                var(ph::ridx(dim))
            };

            if dim == ndim - 1 {
                input_offset = idx;
            } else {
                // stride = product of inner dimensions
                let mut stride: AstNode = input_shape[dim + 1].clone().into();
                for inner_shape in input_shape.iter().take(ndim).skip(dim + 2) {
                    stride = stride * Into::<AstNode>::into(inner_shape.clone());
                }
                input_offset = idx * stride + input_offset;
            }
        }

        load(var(ph::input(input_idx)), input_offset, load_dtype.clone())
    }

    /// Viewチェーン内でインライン化可能かどうかをチェック
    ///
    /// 以下の場合にインライン化可能:
    /// - ConstFill/Const: 定数として埋め込み可能
    /// - View/Contiguous: 再帰的にチェック
    /// - Fusable MapReduce: 全入力がインライン化可能な場合
    ///
    /// バッファを持つノード、Buffer、Executed、Reduce演算は不可
    fn is_inlineable_in_view(inner: &TensorInner) -> bool {
        // バッファを持つノードはインライン化不可（loadが必要）
        if inner.has_buffer() {
            return false;
        }

        match inner.op() {
            // 定数は常にインライン化可能
            TensorOp::Const(_) | TensorOp::ConstFill(_) => true,
            // View/Contiguous: 再帰的にチェック
            TensorOp::View { input } | TensorOp::Contiguous { input } => {
                Self::is_inlineable_in_view(input.as_ref())
            }
            // Elementwise MapReduce: 全入力がインライン化可能な場合
            TensorOp::MapReduce {
                inputs,
                reduce_op: None,
                axes,
                ..
            } if axes.is_empty() => {
                // 全入力がインライン化可能かチェック
                inputs
                    .iter()
                    .all(|inp| Self::is_inlineable_in_view(inp.as_ref()))
            }
            // Buffer/Executed/Reduce演算などはインライン化不可
            _ => false,
        }
    }

    /// 入力テンソルのオフセットを構築
    ///
    /// ブロードキャストを考慮：入力の次元数がループ次元数より小さい場合、
    /// 先頭の軸を無視してオフセットを計算する
    fn build_input_offset(&self, src: &TensorInner, loop_ndim: usize) -> AstNode {
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
        inner: &TensorInner,
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
            let mut stride = shape_dim_to_ast(Some(&shape), axis + 1);
            for inner_axis in (axis + 2)..src_ndim {
                stride = stride * shape_dim_to_ast(Some(&shape), inner_axis);
            }
            offset = var(ph::ridx(loop_axis)) * stride + offset;
        }

        offset
    }

    /// TensorInnerのオフセットを構築
    fn build_offset_for_tensor(&self, inner: &TensorInner, ndim: usize) -> AstNode {
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
            TensorOp::MapReduce { .. } => {
                // Compute演算の結果は連続メモリ
                let shape = inner.view().shape();
                build_contiguous_offset_with_shape(ndim, Some(&shape))
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

/// TensorInnerからASTに変換する簡易関数（内部用）
///
/// realize_core処理で使用するため、TensorInnerを直接受け取る版
pub fn lower_tensor_inner(inner: &TensorInner) -> AstNode {
    let mut lowerer = TensorLowerer::new();
    lowerer.lower_inner(inner)
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
        use crate::tensor::Dim1;
        let a = Tensor::<f32, Dim2>::input("a", [4, 4]);
        let b: Tensor<f32, Dim1> = a.sum(1);

        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&b.into_dyn());

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

    /// Test that the same tensor referenced multiple times uses the same buffer index
    ///
    /// This verifies that `a + a` generates a kernel with only 1 input buffer,
    /// not 2 input buffers. This is critical for correct kernel argument counts.
    #[test]
    fn test_lower_same_tensor_multiple_references() {
        let a = Tensor::<f32, Dim2>::input("a", [2, 3]);
        // a + a should use the same buffer index for both references
        let c = &a + &a;

        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&c.clone().into_dyn());

        // Verify AST is a Program
        match &ast {
            AstNode::Program { functions, .. } => {
                assert!(!functions.is_empty());
            }
            _ => panic!("Expected AstNode::Program"),
        }

        // Verify only 1 input buffer was collected
        // (since both inputs reference the same tensor)
        assert_eq!(
            lowerer.input_buffer_names.len(),
            1,
            "Same tensor referenced twice should result in 1 input buffer, not 2"
        );
    }

    /// Test that different tensors get different buffer indices
    #[test]
    fn test_lower_different_tensors() {
        let a = Tensor::<f32, Dim2>::input("a", [2, 3]);
        let b = Tensor::<f32, Dim2>::input("b", [2, 3]);
        let c = &a + &b;

        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&c.clone().into_dyn());

        // Verify AST is a Program
        match &ast {
            AstNode::Program { functions, .. } => {
                assert!(!functions.is_empty());
            }
            _ => panic!("Expected AstNode::Program"),
        }

        // Verify 2 input buffers were collected
        assert_eq!(
            lowerer.input_buffer_names.len(),
            2,
            "Two different tensors should result in 2 input buffers"
        );
    }

    /// Test mixed case: some same, some different tensors
    #[test]
    fn test_lower_mixed_tensor_references() {
        let a = Tensor::<f32, Dim2>::input("a", [2, 3]);
        let b = Tensor::<f32, Dim2>::input("b", [2, 3]);
        // (a + b) + a: should have 2 inputs (a and b), with 'a' referenced twice
        let sum1 = &a + &b;
        let c = &sum1 + &a;

        let mut lowerer = TensorLowerer::new();
        let ast = lowerer.lower(&c.clone().into_dyn());

        // Verify AST is a Program
        match &ast {
            AstNode::Program { functions, .. } => {
                assert!(!functions.is_empty());
            }
            _ => panic!("Expected AstNode::Program"),
        }

        // Verify 2 input buffers were collected (a and b, not 3)
        assert_eq!(
            lowerer.input_buffer_names.len(),
            2,
            "Two different tensors (a and b) should result in 2 input buffers, not 3"
        );
    }
}
