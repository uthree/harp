use super::Lowerer;
use crate::graph::{Graph, GraphNode, GraphOp};
use std::collections::{HashMap, HashSet, VecDeque};

impl Lowerer {
    /// トポロジカルソートを実行し、世代（レベル）ごとにノードをグループ化
    /// 各世代は並列実行可能なノードのグループを表す
    pub(super) fn topological_sort_by_generation(&self, graph: &Graph) -> Vec<Vec<GraphNode>> {
        let mut in_degree: HashMap<GraphNode, usize> = HashMap::new();
        let mut adjacency: HashMap<GraphNode, Vec<GraphNode>> = HashMap::new();
        let mut all_nodes = HashSet::new();
        let mut node_level: HashMap<GraphNode, usize> = HashMap::new();

        // グラフを走査して依存関係を構築
        let mut queue = VecDeque::new();
        for output in &graph.outputs {
            queue.push_back(output.clone());
        }

        while let Some(node) = queue.pop_front() {
            if all_nodes.contains(&node) {
                continue;
            }
            all_nodes.insert(node.clone());

            let deps = self.get_dependencies(&node);
            in_degree.insert(node.clone(), deps.len());

            for dep in deps {
                adjacency.entry(dep.clone()).or_default().push(node.clone());
                queue.push_back(dep);
            }
        }

        // レベル（世代）を計算しながらトポロジカルソート実行
        let mut generations: Vec<Vec<GraphNode>> = Vec::new();
        let mut zero_in_degree: VecDeque<_> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(node, _)| node.clone())
            .collect();

        // 初期ノード（入力ノード等）のレベルを0に設定
        for node in &zero_in_degree {
            node_level.insert(node.clone(), 0);
        }

        while !zero_in_degree.is_empty() {
            // 現在の世代のノードを収集
            let current_generation: Vec<GraphNode> = zero_in_degree.drain(..).collect();

            // 次の世代の候補を収集
            let mut next_generation = Vec::new();

            for node in &current_generation {
                if let Some(neighbors) = adjacency.get(node) {
                    let current_level = *node_level.get(node).unwrap_or(&0);

                    for neighbor in neighbors {
                        if let Some(degree) = in_degree.get_mut(neighbor) {
                            *degree -= 1;

                            // ノードのレベルを更新（全ての依存元の最大レベル+1）
                            let neighbor_level = node_level.entry(neighbor.clone()).or_insert(0);
                            *neighbor_level = (*neighbor_level).max(current_level + 1);

                            if *degree == 0 {
                                next_generation.push(neighbor.clone());
                            }
                        }
                    }
                }
            }

            generations.push(current_generation);
            zero_in_degree = next_generation.into_iter().collect();
        }

        generations
    }

    pub(super) fn get_dependencies(&self, node: &GraphNode) -> Vec<GraphNode> {
        match &node.op {
            GraphOp::Input(_) => vec![],
            GraphOp::Const(_) => vec![],
            GraphOp::Elementwise(op) => {
                use crate::graph::ops::ElementwiseOp;
                match op {
                    ElementwiseOp::Add(lhs, rhs)
                    | ElementwiseOp::Mul(lhs, rhs)
                    | ElementwiseOp::Max(lhs, rhs)
                    | ElementwiseOp::Mod(lhs, rhs)
                    | ElementwiseOp::LessThan(lhs, rhs)
                    | ElementwiseOp::Eq(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
                    ElementwiseOp::Neg(n)
                    | ElementwiseOp::Recip(n)
                    | ElementwiseOp::Sin(n)
                    | ElementwiseOp::Sqrt(n)
                    | ElementwiseOp::Log2(n)
                    | ElementwiseOp::Exp2(n) => vec![n.clone()],
                    ElementwiseOp::Select(cond, true_val, false_val) => {
                        vec![cond.clone(), true_val.clone(), false_val.clone()]
                    }
                }
            }
            GraphOp::Reduce(_, _, input) => vec![input.clone()],
            GraphOp::Cumulative(_, _, input) => vec![input.clone()],
            GraphOp::View(n) => vec![n.clone()],
            GraphOp::Contiguous(input) => vec![input.clone()],
            GraphOp::Cast(input, _) => vec![input.clone()],
            GraphOp::Fold(_, _, _, _, input) => vec![input.clone()],
            GraphOp::FusedElementwise(_, nodes) => nodes.clone(),
            GraphOp::FusedReduce(_, _, input) => vec![input.clone()],
            GraphOp::FusedElementwiseReduce(_, nodes, _, _) => nodes.clone(),
            GraphOp::FusedElementwiseCumulative(_, nodes, _) => nodes.clone(),
        }
    }
}
