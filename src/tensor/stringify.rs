//! グラフの文字列化
//!
//! Tensorグラフを一意な文字列表現に変換する。
//! キャッシュキーとして使用可能で、デバッグ表示にも流用できる。

use std::collections::HashMap;
use std::fmt::Write;

use crate::ast::AstNode;
use crate::tensor::Tensor;
use crate::tensor::TensorDType;
use crate::tensor::TensorInner;
use crate::tensor::dimension::Dimension;
use crate::tensor::ops::{InputRef, TensorOp};
use crate::tensor::shape::View;

/// グラフを文字列化する
///
/// # Example
/// ```text
/// Compute(Add(Load($0), Mul(Load($1), Const(2.0))), shape=[4,4], dtype=f32)
/// ```
/// - `$0`, `$1` は入力バッファの位置インデックス（出現順）
pub fn stringify_graph<T: TensorDType, D: Dimension>(tensor: &Tensor<T, D>) -> String {
    let mut stringifier = GraphStringifier::new();
    stringifier.visit(tensor.inner.as_ref())
}

/// TensorInnerからグラフを文字列化する（内部用）
///
/// realize処理で使用するため、TensorInnerを直接受け取る版
pub fn stringify_graph_inner(inner: &TensorInner) -> String {
    let mut stringifier = GraphStringifier::new();
    stringifier.visit(inner)
}

/// グラフ文字列化のための訪問者
struct GraphStringifier {
    /// ポインタ → 入力位置インデックスのマッピング
    input_positions: HashMap<usize, usize>,
    /// 次の入力位置インデックス
    next_input_position: usize,
    /// 訪問済みノード → 文字列表現のキャッシュ（DAG対応）
    visited: HashMap<usize, String>,
}

impl GraphStringifier {
    fn new() -> Self {
        Self {
            input_positions: HashMap::new(),
            next_input_position: 0,
            visited: HashMap::new(),
        }
    }

    /// ノードを訪問して文字列化
    fn visit(&mut self, inner: &TensorInner) -> String {
        let ptr = inner as *const TensorInner as usize;

        // DAG: 既に訪問済みなら参照を返す
        if let Some(cached) = self.visited.get(&ptr) {
            return cached.clone();
        }

        let result = self.stringify_op(inner);

        // キャッシュに保存（入力ノードは除く）
        if !matches!(inner.op(), TensorOp::Buffer { .. } | TensorOp::Executed) {
            self.visited.insert(ptr, result.clone());
        }

        result
    }

    /// TensorOpを文字列化
    fn stringify_op(&mut self, inner: &TensorInner) -> String {
        let shape = inner.shape();
        let dtype = inner.dtype();

        // バッファを持つノードは入力として扱う（Executedと同様）
        // これにより、中間結果がrealizeされた場合にキャッシュキーが変わる
        if inner.has_buffer() {
            let pos = self.get_or_assign_input_position(inner);
            return format!("Buffered(${}, shape={:?}, dtype={:?})", pos, shape, dtype);
        }

        match inner.op() {
            // ソース演算
            TensorOp::Buffer { name } => {
                let pos = self.get_or_assign_input_position(inner);
                format!(
                    "Buffer(${}:{}, shape={:?}, dtype={:?})",
                    pos, name, shape, dtype
                )
            }
            TensorOp::Executed => {
                let pos = self.get_or_assign_input_position(inner);
                format!("Executed(${}, shape={:?}, dtype={:?})", pos, shape, dtype)
            }
            TensorOp::Const(lit) => {
                format!("Const({:?})", lit)
            }
            TensorOp::ConstFill(lit) => {
                format!("ConstFill({:?}, shape={:?})", lit, shape)
            }
            TensorOp::Rand => {
                format!("Rand(shape={:?}, dtype={:?})", shape, dtype)
            }
            TensorOp::Arange => {
                format!("Arange(shape={:?}, dtype={:?})", shape, dtype)
            }

            // 単項演算
            TensorOp::View { input } => {
                let child = self.visit_input(input);
                let view = inner.view();
                format!("View({}, view={})", child, self.stringify_view(view))
            }
            TensorOp::Contiguous { input } => {
                let child = self.visit_input(input);
                format!("Contiguous({})", child)
            }
            TensorOp::Clone { input } => {
                let child = self.visit_input(input);
                format!("Clone({})", child)
            }

            // 統一計算演算
            TensorOp::MapReduce {
                inputs,
                expr,
                reduce_op,
                axes,
                keepdim,
            } => {
                let children: Vec<String> = inputs.iter().map(|i| self.visit_input(i)).collect();
                let children_str = children.join(", ");
                let expr_str = self.stringify_ast(expr);

                let mut result = format!("MapReduce([{}], expr={}", children_str, expr_str);

                if let Some(op) = reduce_op {
                    write!(result, ", reduce={:?}", op).unwrap();
                }
                if !axes.is_empty() {
                    write!(result, ", axes={:?}", axes).unwrap();
                }
                if *keepdim {
                    write!(result, ", keepdim").unwrap();
                }
                write!(result, ", shape={:?}, dtype={:?})", shape, dtype).unwrap();

                result
            }

            // 構造演算
            TensorOp::Concat { inputs, axis } => {
                let children: Vec<String> = inputs.iter().map(|i| self.visit_input(i)).collect();
                format!("Concat([{}], axis={})", children.join(", "), axis)
            }
        }
    }

    /// InputRefを訪問
    fn visit_input(&mut self, input: &InputRef) -> String {
        self.visit(input.as_ref())
    }

    /// 入力ノードに位置インデックスを割り当て
    fn get_or_assign_input_position(&mut self, inner: &TensorInner) -> usize {
        let ptr = inner as *const TensorInner as usize;
        *self.input_positions.entry(ptr).or_insert_with(|| {
            let pos = self.next_input_position;
            self.next_input_position += 1;
            pos
        })
    }

    /// Viewを文字列化
    fn stringify_view(&self, view: &View) -> String {
        match view {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                format!(
                    "Linear(shape={:?}, strides={:?}, offset={:?})",
                    shape, strides, offset
                )
            }
            View::IndexExpr { shape, index_expr } => {
                format!("IndexExpr(shape={:?}, expr={:?})", shape, index_expr)
            }
            View::Padded {
                inner,
                padding,
                default_value,
            } => {
                format!(
                    "Padded(inner={}, padding={:?}, default={:?})",
                    self.stringify_view(inner),
                    padding,
                    default_value
                )
            }
            View::Masked {
                inner,
                condition,
                default_value,
            } => {
                format!(
                    "Masked(inner={}, condition={}, default={:?})",
                    self.stringify_view(inner),
                    condition,
                    default_value
                )
            }
        }
    }

    /// AstNodeを簡潔に文字列化
    fn stringify_ast(&self, node: &AstNode) -> String {
        match node {
            // リテラル・変数
            AstNode::Const(lit) => format!("{:?}", lit),
            AstNode::Wildcard(name) => format!("${}", name),
            AstNode::Var(name) => format!("Var({})", name),

            // 二項演算
            AstNode::Add(l, r) => {
                format!("Add({}, {})", self.stringify_ast(l), self.stringify_ast(r))
            }
            AstNode::Mul(l, r) => {
                format!("Mul({}, {})", self.stringify_ast(l), self.stringify_ast(r))
            }
            AstNode::Max(l, r) => {
                format!("Max({}, {})", self.stringify_ast(l), self.stringify_ast(r))
            }
            AstNode::Rem(l, r) => {
                format!("Rem({}, {})", self.stringify_ast(l), self.stringify_ast(r))
            }
            AstNode::Idiv(l, r) => {
                format!("Idiv({}, {})", self.stringify_ast(l), self.stringify_ast(r))
            }

            // 単項演算
            AstNode::Recip(x) => format!("Recip({})", self.stringify_ast(x)),
            AstNode::Sqrt(x) => format!("Sqrt({})", self.stringify_ast(x)),
            AstNode::Log2(x) => format!("Log2({})", self.stringify_ast(x)),
            AstNode::Exp2(x) => format!("Exp2({})", self.stringify_ast(x)),
            AstNode::Sin(x) => format!("Sin({})", self.stringify_ast(x)),
            AstNode::Floor(x) => format!("Floor({})", self.stringify_ast(x)),

            // キャスト
            AstNode::Cast(value, dtype) => {
                format!("Cast({}, {:?})", self.stringify_ast(value), dtype)
            }

            // FMA
            AstNode::Fma { a, b, c } => {
                format!(
                    "Fma({}, {}, {})",
                    self.stringify_ast(a),
                    self.stringify_ast(b),
                    self.stringify_ast(c)
                )
            }

            // 比較・論理演算（プリミティブのみ）
            AstNode::Lt(l, r) => {
                format!("Lt({}, {})", self.stringify_ast(l), self.stringify_ast(r))
            }
            AstNode::And(l, r) => {
                format!("And({}, {})", self.stringify_ast(l), self.stringify_ast(r))
            }
            AstNode::Not(a) => {
                format!("Not({})", self.stringify_ast(a))
            }

            // その他のノードはDebug表示にフォールバック
            _ => format!("{:?}", node),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dim2, Tensor};

    #[test]
    fn test_stringify_simple_add() {
        let a = Tensor::<f32, Dim2>::input("a", [2, 3]);
        let b = Tensor::<f32, Dim2>::input("b", [2, 3]);
        let c = &a + &b;

        let repr = stringify_graph(&c);
        println!("Graph repr: {}", repr);

        // 基本的な構造が含まれていることを確認
        assert!(repr.contains("MapReduce"));
        assert!(repr.contains("Buffer"));
        assert!(repr.contains("$0"));
        assert!(repr.contains("$1"));
    }

    #[test]
    fn test_stringify_same_graph_same_output() {
        // 同じ構造のグラフは同じ文字列になることを確認
        let a1 = Tensor::<f32, Dim2>::input("a", [4, 4]);
        let b1 = Tensor::<f32, Dim2>::input("b", [4, 4]);
        let c1 = &a1 + &b1;

        let a2 = Tensor::<f32, Dim2>::input("a", [4, 4]);
        let b2 = Tensor::<f32, Dim2>::input("b", [4, 4]);
        let c2 = &a2 + &b2;

        let repr1 = stringify_graph(&c1);
        let repr2 = stringify_graph(&c2);

        assert_eq!(
            repr1, repr2,
            "Same graph structure should produce same string"
        );
    }

    #[test]
    fn test_stringify_different_shapes_different_output() {
        // 異なる形状のグラフは異なる文字列になることを確認
        let a1 = Tensor::<f32, Dim2>::input("a", [2, 3]);
        let b1 = Tensor::<f32, Dim2>::input("b", [2, 3]);
        let c1 = &a1 + &b1;

        let a2 = Tensor::<f32, Dim2>::input("a", [4, 4]);
        let b2 = Tensor::<f32, Dim2>::input("b", [4, 4]);
        let c2 = &a2 + &b2;

        let repr1 = stringify_graph(&c1);
        let repr2 = stringify_graph(&c2);

        assert_ne!(
            repr1, repr2,
            "Different shapes should produce different strings"
        );
    }
}
