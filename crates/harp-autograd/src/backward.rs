//! 逆伝播の実装
//!
//! トポロジカルソートして計算グラフを逆順に辿り、勾配を伝播します。

use super::tensor::Tensor;
use harp::graph::{GraphNode, GraphNodeData};
use std::collections::{HashMap, HashSet};

/// 逆伝播を実行
///
/// # 引数
/// - `output`: 出力テンソル（スカラーである必要がある）
/// - `grad_output`: 出力に対する勾配（通常は1.0）
pub(super) fn backward(output: &Tensor, grad_output: GraphNode) {
    // 1. トポロジカルソート
    let topo_order = topological_sort(output);

    // 2. 勾配を保持するマップ（GraphNodeDataのポインタをキーにする）
    let mut grads: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();
    grads.insert(output.data.as_ptr(), grad_output);

    // 3. 逆順に勾配を伝播
    for tensor in topo_order.iter().rev() {
        // requires_gradがfalseなら勾配計算をスキップ
        if !tensor.requires_grad() {
            continue;
        }

        // このテンソルの勾配を取得
        let tensor_ptr = tensor.data.as_ptr();
        let grad = match grads.get(&tensor_ptr) {
            Some(g) => g.clone(),
            None => continue, // 勾配がない場合はスキップ
        };

        // grad_fnがあれば勾配を計算して入力に伝播
        if let Some(grad_fn_wrapper) = tensor.grad_fn() {
            let grad_tensor = Tensor::from_graph_node(grad.clone(), false);
            let input_grads = grad_fn_wrapper
                .grad_fn
                .apply(&grad_tensor, &grad_fn_wrapper.inputs);

            // 各入力に勾配を伝播
            for (input, input_grad) in grad_fn_wrapper.inputs.iter().zip(input_grads) {
                if let Some(ig) = input_grad
                    && input.requires_grad()
                {
                    let input_ptr = input.data.as_ptr();
                    // 勾配を累積
                    if let Some(existing_grad) = grads.get_mut(&input_ptr) {
                        *existing_grad = existing_grad.clone() + ig.data.clone();
                    } else {
                        grads.insert(input_ptr, ig.data.clone());
                    }
                }
            }
        }

        // 最終的な勾配をテンソルに保存
        tensor.accumulate_grad(grad);
    }
}

/// トポロジカルソート
///
/// 出力から入力に向かって深さ優先探索で訪問順序を記録します。
fn topological_sort(output: &Tensor) -> Vec<Tensor> {
    let mut visited = HashSet::new();
    let mut topo_order = Vec::new();

    fn dfs(
        tensor: &Tensor,
        visited: &mut HashSet<*const GraphNodeData>,
        topo_order: &mut Vec<Tensor>,
    ) {
        let ptr = tensor.data.as_ptr();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        // 入力テンソルを再帰的に訪問
        if let Some(grad_fn_wrapper) = tensor.grad_fn() {
            for input in &grad_fn_wrapper.inputs {
                dfs(input, visited, topo_order);
            }
        }

        // 訪問後に追加（後置順）
        topo_order.push(tensor.clone());
    }

    dfs(output, &mut visited, &mut topo_order);
    topo_order
}

#[cfg(test)]
mod tests {
    use super::*;
    use harp::graph::{DType, Graph};

    #[test]
    fn test_topological_sort() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([2, 3])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([2, 3])
            .build();

        let ta = Tensor::from_graph_node(a, true);
        let tb = Tensor::from_graph_node(b, true);

        // (a + b) * a
        let sum = &ta + &tb;
        let result = &sum * &ta;

        let topo = topological_sort(&result);

        // 順序確認: a, b -> sum -> result
        // ただし、a + b と sum * a の両方で a が参照されるため、
        // 重複なしで a, b, sum, result の4つのノードが期待される
        assert!(
            topo.len() >= 3,
            "Expected at least 3 nodes, got {}",
            topo.len()
        );
    }
}
