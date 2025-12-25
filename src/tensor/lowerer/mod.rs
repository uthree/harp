//! TensorLowerer - TensorNodeから直接ASTへの変換
//!
//! TensorNodeツリーをASTに変換する。
//! FusedElementwiseReduceパターンを使用して全演算を統一的に処理する。
//!
//! # 設計
//!
//! 全ての計算演算を正規化形式 (expr, reduce_op, axes) に変換:
//! - Elementwise: (expr, None, [])
//! - FusedElementwise: (expr, None, [])
//! - Reduce: (Wildcard("0"), Some(op), axes)
//! - FusedElementwiseReduce: (expr, Some(op), axes)
//!
//! これにより、axes=[]ならElementwiseパス、そうでなければReduceパスで処理。
//!
//! # 使用例
//!
//! ```ignore
//! use harp::tensor::{Tensor, Dim2};
//! use harp::tensor::lowerer::TensorLowerer;
//!
//! let a = Tensor::<Dim2>::input("a", [2, 3]);
//! let b = Tensor::<Dim2>::input("b", [2, 3]);
//! let c = &a + &b;
//!
//! let mut lowerer = TensorLowerer::new();
//! let ast = lowerer.lower(&c.clone().into_dyn());
//! ```

pub mod expr_builder;
pub mod helpers;

use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::{AstKernelCallInfo, AstNode, DType as AstDType, Mutability, Scope, helper::*};
use crate::core::DType;
use crate::core::shape::{Expr, View};
use crate::tensor::ops::{ReduceOp, TensorOp};
use crate::tensor::{DimDyn, Tensor, TensorNode};

use expr_builder::{build_elementwise_expr, build_reduce_identity_expr};
use helpers::*;

/// 入力バッファのメタデータ
#[derive(Debug, Clone)]
struct InputMeta {
    name: String,
    dtype: DType,
    shape: Vec<Expr>,
    view: View,
}

/// TensorをASTに変換するLowerer
///
/// TensorNodeツリーをトラバースし、ASTプログラムを直接生成する。
pub struct TensorLowerer {
    /// カーネル名カウンタ
    kernel_counter: usize,
    /// 収集された入力バッファ (name -> meta)
    input_buffers: HashMap<String, InputMeta>,
    /// 処理済みTensorNodeキャッシュ (ptr -> processed flag)
    visited: HashMap<*const TensorNode, bool>,
}

impl TensorLowerer {
    /// 新しいTensorLowererを作成
    pub fn new() -> Self {
        Self {
            kernel_counter: 0,
            input_buffers: HashMap::new(),
            visited: HashMap::new(),
        }
    }

    /// TensorをASTに変換
    ///
    /// # Arguments
    /// * `tensor` - 変換するテンソル
    ///
    /// # Returns
    /// ASTプログラム
    pub fn lower(&mut self, tensor: &Tensor<DimDyn>) -> AstNode {
        // 入力バッファを収集
        self.collect_input_buffers(&tensor.inner);

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
    fn collect_input_buffers(&mut self, node: &Rc<TensorNode>) {
        let ptr = Rc::as_ptr(node);
        if self.visited.contains_key(&ptr) {
            return;
        }
        self.visited.insert(ptr, true);

        match &node.op {
            TensorOp::Buffer { name } => {
                if !self.input_buffers.contains_key(name) {
                    self.input_buffers.insert(
                        name.clone(),
                        InputMeta {
                            name: name.clone(),
                            dtype: node.dtype.clone(),
                            shape: node.view.shape().to_vec(),
                            view: node.view.clone(),
                        },
                    );
                }
            }
            _ => {
                // 再帰的に子ノードを処理
                for src in &node.src {
                    self.collect_input_buffers(&src.inner);
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

    /// TensorNodeをASTにlower
    fn lower_node(&self, node: &Rc<TensorNode>, name: &str) -> AstNode {
        // 正規化形式に変換
        let (expr, reduce_op, axes) = self.normalize_op(node);
        let ndim = node.view.shape().len();
        let shape = node.view.shape();

        if axes.is_empty() {
            // Elementwiseパス
            self.lower_elementwise_path(node, &expr, ndim, shape, name)
        } else {
            // Reduceパス
            self.lower_reduce_path(node, &expr, reduce_op.as_ref().unwrap(), &axes, ndim, name)
        }
    }

    /// TensorOpを正規化形式に変換
    ///
    /// Returns: (expr, reduce_op, axes)
    fn normalize_op(&self, node: &Rc<TensorNode>) -> (AstNode, Option<ReduceOp>, Vec<usize>) {
        match &node.op {
            TensorOp::Elementwise { op } => {
                let expr = build_elementwise_expr(op);
                (expr, None, vec![])
            }
            TensorOp::FusedElementwise { expr } => (expr.clone(), None, vec![]),
            TensorOp::Reduce { op, axes, .. } => {
                let expr = build_reduce_identity_expr();
                (expr, Some(op.clone()), axes.clone())
            }
            TensorOp::FusedElementwiseReduce {
                expr,
                reduce_op,
                axes,
                ..
            } => (expr.clone(), Some(reduce_op.clone()), axes.clone()),
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
            // Buffer, View, Clone, Contiguous等は直接lowerしない
            _ => {
                // これらのケースはlower_nodeで到達すべきでない
                // 通常はsrcを辿って計算ノードに到達する
                panic!("Cannot normalize op: {:?}", node.op);
            }
        }
    }

    /// Elementwiseパスでlower
    fn lower_elementwise_path(
        &self,
        node: &Rc<TensorNode>,
        expr: &AstNode,
        ndim: usize,
        shape: &[Expr],
        name: &str,
    ) -> AstNode {
        let load_dtype = dtype_to_ast(&node.dtype);

        // 各入力のload式を構築
        let mut mappings = HashMap::new();
        let mut non_const_idx = 0;

        for (i, src) in node.src.iter().enumerate() {
            if let TensorOp::Const(lit) = &src.inner.op {
                // 定数は直接埋め込み
                mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
            } else {
                // 入力バッファからload
                let src_offset = self.build_input_offset(src, ndim);
                let load_node = load(
                    var(ph::input(non_const_idx)),
                    src_offset,
                    load_dtype.clone(),
                );
                mappings.insert(i.to_string(), load_node);
                non_const_idx += 1;
            }
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
        node: &Rc<TensorNode>,
        expr: &AstNode,
        reduce_op: &ReduceOp,
        axes: &[usize],
        ndim: usize,
        name: &str,
    ) -> AstNode {
        // 入力ノードの形状を取得（srcが存在する場合）
        let input_shape = if let Some(src) = node.src.first() {
            src.inner.view.shape()
        } else {
            node.view.shape()
        };

        let load_dtype = dtype_to_ast(&node.dtype);

        // 各入力のload式を構築
        let mut mappings = HashMap::new();
        let mut non_const_idx = 0;

        for (i, src) in node.src.iter().enumerate() {
            if let TensorOp::Const(lit) = &src.inner.op {
                mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
            } else {
                let src_offset = self.build_input_offset(src, ndim);
                let load_node = load(
                    var(ph::input(non_const_idx)),
                    src_offset,
                    load_dtype.clone(),
                );
                mappings.insert(i.to_string(), load_node);
                non_const_idx += 1;
            }
        }

        let value_expr = expr.substitute(&mappings);

        // アキュムレータ初期化と更新
        let (init_value, accumulate_fn) = build_reduce_accumulator(reduce_op, &node.dtype);

        // 出力オフセット（縮約軸を除く）
        let output_offset =
            build_contiguous_offset_excluding_axes_with_shape(ndim, axes, Some(input_shape));

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
            dtype_to_ast(&node.dtype),
            Mutability::Mutable,
        );

        let acc_init = assign(acc_var, init_value);
        let store_stmt = store(var(ph::OUTPUT), output_offset, var(acc_var));

        let inner_body = vec![acc_init, reduce_loops, store_stmt];
        let body = wrap_with_loops_excluding_axes_with_scope_and_shape(
            ndim,
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
    fn build_input_offset(&self, src: &Tensor<DimDyn>, ndim: usize) -> AstNode {
        // srcのviewを辿ってBufferまで到達
        self.build_offset_for_tensor(&src.inner, ndim)
    }

    /// TensorNodeのオフセットを構築
    fn build_offset_for_tensor(&self, node: &Rc<TensorNode>, ndim: usize) -> AstNode {
        match &node.op {
            TensorOp::Buffer { .. } => {
                // Bufferノードは自身のviewを使用
                build_strided_offset(&node.view, ndim)
            }
            TensorOp::View => {
                // Viewノードは自身のviewを使用
                build_strided_offset(&node.view, ndim)
            }
            TensorOp::Contiguous => {
                // Contiguousは連続メモリアクセス
                let shape = node.view.shape();
                build_contiguous_offset_with_shape(ndim, Some(shape))
            }
            _ => {
                // その他の演算は自身のviewを使用
                build_strided_offset(&node.view, ndim)
            }
        }
    }

    /// Programとしてラップ
    fn wrap_as_program(&self, functions: Vec<AstNode>, kernel_name: &str, numel: usize) -> AstNode {
        // 入力バッファをソート（安定した順序のため）
        let mut input_names: Vec<_> = self.input_buffers.keys().cloned().collect();
        input_names.sort();

        // スレッドグループサイズを計算
        let local_size = 64.min(numel);
        let grid_size = (numel + local_size - 1) / local_size;

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
pub fn lower_tensor(tensor: &Tensor<DimDyn>) -> AstNode {
    let mut lowerer = TensorLowerer::new();
    lowerer.lower(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_lower_simple_add() {
        let a = Tensor::<Dim2>::input("a", [2, 3]);
        let b = Tensor::<Dim2>::input("b", [2, 3]);
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
        let a = Tensor::<Dim2>::input("a", [4, 4]);
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
        let a = Tensor::<Dim2>::input("a", [4, 4]);
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
        let a = Tensor::<Dim2>::full([2, 3], 1.0);

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
