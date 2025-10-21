use crate::graph::shape::view::View;
use crate::graph::{Graph, GraphNode};

/// ループ順序変更の提案を行う構造体
pub struct LoopPermutationSuggester;

impl LoopPermutationSuggester {
    /// メモリアクセスパターンを最適化するループ順序の提案を生成
    ///
    /// 戦略：
    /// 1. ストライドが小さい次元を最内ループに配置
    /// 2. 連続メモリアクセスを促進
    /// 3. キャッシュヒット率を向上
    pub fn suggest(graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();

        // 各出力ノードに対してループ順序の最適化を試みる
        for output in &graph.outputs {
            if let Some(optimized_graph) = Self::try_permute_for_better_access(graph, output) {
                suggestions.push(optimized_graph);
            }
        }

        suggestions
    }

    /// 特定のノードに対してメモリアクセスを改善するpermutationを試みる
    fn try_permute_for_better_access(_graph: &Graph, node: &GraphNode) -> Option<Graph> {
        // ノードのviewを分析
        let view = &node.view;

        // 最適なループ順序を計算
        let optimal_order = Self::compute_optimal_order(view)?;

        // 現在の順序と異なる場合のみpermutationを提案
        let current_order: Vec<usize> = (0..optimal_order.len()).collect();
        if optimal_order == current_order {
            return None;
        }

        // TODO: 実際のGraph変換を実装
        // 現在はプレースホルダー
        None
    }

    /// Viewからストライドに基づいて最適なループ順序を計算
    ///
    /// ストライドが小さい次元ほど内側のループに配置すべき
    fn compute_optimal_order(view: &View) -> Option<Vec<usize>> {
        match view {
            View::Linear { strides, .. } => {
                if strides.is_empty() {
                    return None;
                }

                // ストライドの大きさでソート（小さい順）
                let mut indexed_strides: Vec<(usize, &crate::graph::shape::Expr)> =
                    strides.iter().enumerate().collect();

                // ストライドを評価値に変換してソート
                indexed_strides.sort_by_key(|(_, stride)| Self::evaluate_stride(stride));

                // ソート結果から順序を取得
                let order: Vec<usize> = indexed_strides.iter().map(|(idx, _)| *idx).collect();

                Some(order)
            }
        }
    }

    /// ストライドの評価値を計算（小さいほど優先度が高い）
    fn evaluate_stride(stride: &crate::graph::shape::Expr) -> i64 {
        use crate::graph::shape::Expr;

        match stride {
            Expr::Const(val) => *val as i64,
            Expr::Var(_) => i64::MAX, // 変数は最も優先度が低い
            Expr::Add(lhs, rhs) => Self::evaluate_stride(lhs) + Self::evaluate_stride(rhs),
            Expr::Sub(lhs, rhs) => Self::evaluate_stride(lhs) - Self::evaluate_stride(rhs),
            Expr::Mul(lhs, rhs) => {
                Self::evaluate_stride(lhs).saturating_mul(Self::evaluate_stride(rhs))
            }
            Expr::Div(lhs, _) => Self::evaluate_stride(lhs), // 除算は近似的に扱う
            Expr::Rem(_, _) => i64::MAX / 2,                 // 剰余は不確定
        }
    }

    /// ループ順序を適用した新しいGraphを生成
    pub fn apply_permutation(graph: &Graph, node_id: usize, permutation: Vec<usize>) -> Graph {
        let new_graph = graph.clone();

        // TODO: 実際のpermutation適用ロジックを実装
        // 1. 対象ノードの前にViewノードを挿入
        // 2. Viewノードでpermuteを適用
        // 3. グラフを再構築

        let _ = (node_id, permutation); // 一時的に警告を抑制
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::shape::Expr;

    #[test]
    fn test_compute_optimal_order_contiguous() {
        // Contiguous view: strides = [12, 4, 1] for shape [2, 3, 4]
        // 最適順序は [2, 1, 0] (逆順)
        let view = View::new_contiguous(vec![2, 3, 4]);
        let order = LoopPermutationSuggester::compute_optimal_order(&view);

        assert!(order.is_some());
        let order = order.unwrap();
        // ストライド [12, 4, 1] を昇順ソート → インデックス [2, 1, 0]
        assert_eq!(order, vec![2, 1, 0]);
    }

    #[test]
    fn test_compute_optimal_order_permuted() {
        // Permuted view: strides = [1, 12, 4] for shape [4, 2, 3]
        let view = View::new_contiguous(vec![2, 3, 4]).permute(vec![2, 0, 1]);

        let order = LoopPermutationSuggester::compute_optimal_order(&view);
        assert!(order.is_some());
        let order = order.unwrap();
        // ストライド [1, 12, 4] を昇順ソート → インデックス [0, 2, 1]
        assert_eq!(order, vec![0, 2, 1]);
    }

    #[test]
    fn test_evaluate_stride_const() {
        assert_eq!(LoopPermutationSuggester::evaluate_stride(&Expr::from(1)), 1);
        assert_eq!(
            LoopPermutationSuggester::evaluate_stride(&Expr::from(100)),
            100
        );
    }

    #[test]
    fn test_evaluate_stride_variable() {
        let var_stride = Expr::Var("N".to_string());
        // 変数は最も優先度が低い
        assert_eq!(
            LoopPermutationSuggester::evaluate_stride(&var_stride),
            i64::MAX
        );
    }

    #[test]
    fn test_suggest_returns_empty_for_simple_graph() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![2.into(), 3.into(), 4.into()]);
        graph.output(input);

        let suggestions = LoopPermutationSuggester::suggest(&graph);
        // 単純なグラフでは最適化提案なし
        assert_eq!(suggestions.len(), 0);
    }
}
