use crate::graph::{Graph, GraphNode, ops::GraphOp};
use std::collections::{HashMap, HashSet, VecDeque};

// モジュール宣言
mod contiguous;
mod cumulative;
mod elementwise;
mod fold;
mod fused_elementwise;
mod fused_elementwise_reduce;
mod fused_reduce;
mod reduce;
mod utils;

pub struct Lowerer {
    alu_counter: usize, // 一時変数のカウンター
}

/// トポロジカルソートの結果。各世代（Generation）は並列実行可能なノード群。
pub type TopologicalOrder = Vec<Vec<GraphNode>>;

/// GraphNodeから内部のポインタを取得するヘルパー関数
fn node_ptr(node: &GraphNode) -> *const () {
    node.as_ptr() as *const ()
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

/// GraphをProgramに変換する公開関数
///
/// Graphの全ノードをカーネル関数に変換し、Programとして返します。
/// 現時点では各ノードを個別のカーネル関数として生成し、
/// kernel_main関数による統合は未実装です。
pub(crate) fn lower(graph: Graph) -> crate::ast::Program {
    let mut lowerer = Lowerer::new();

    // トポロジカルソートでノードを取得
    let generations = Lowerer::topological_sort(&graph);

    // Programを作成（entry_pointはとりあえず"main"）
    let mut program = crate::ast::Program::new("main".to_string());

    // 各世代の各ノードをカーネル関数に変換
    let mut kernel_id = 0;
    let mut first_kernel_name = String::new();
    for generation in generations {
        for node in generation {
            // Input ノードはスキップ
            if matches!(node.op, GraphOp::Input) {
                continue;
            }

            // カーネル関数を生成
            if let Ok(function) = lowerer.lower_node_to_kernel(&node, kernel_id) {
                let kernel_name = format!("kernel_{}", kernel_id);
                if kernel_id == 0 {
                    first_kernel_name = kernel_name.clone();
                }
                let _ = program.add_function(kernel_name, function);
                kernel_id += 1;
            }
        }
    }

    // entry_pointを最初のカーネルに設定（もしあれば）
    if !first_kernel_name.is_empty() {
        program.entry_point = first_kernel_name;
    }

    program
}

impl Lowerer {
    pub fn new() -> Self {
        Self { alu_counter: 0 }
    }

    /// 新しい一時変数名を生成
    pub(super) fn fresh_alu(&mut self) -> String {
        let name = format!("alu{}", self.alu_counter);
        self.alu_counter += 1;
        name
    }

    /// GraphNodeを一つのカーネル関数に変換（最も単純なケース）
    /// 前提：contiguous, 全軸Sequential, SIMD未使用
    pub fn lower_node_to_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
    ) -> Result<crate::ast::Function, String> {
        match &node.op {
            GraphOp::Elementwise { op, .. } => self.lower_elementwise_kernel(node, node_id, op),
            GraphOp::Reduce { op, axis, .. } => self.lower_reduce_kernel(node, node_id, op, *axis),
            GraphOp::Contiguous { .. } => self.lower_contiguous_kernel(node, node_id),
            _ => Err(format!("Unsupported operation: {:?}", node.op)),
        }
    }

    // === トポロジカルソート関連 ===

    /// Kahnのアルゴリズムを使用してグラフをトポロジカルソートし、世代別にグループ化する。
    /// 各世代のノードは同時に計算可能。
    pub fn topological_sort(graph: &Graph) -> TopologicalOrder {
        // 1. すべてのノードを収集（出力ノードから再帰的に辿る）
        let all_nodes = Self::collect_all_nodes(graph);

        // 2. 各ノードの入次数を計算（何個のノードから参照されているか）
        let mut in_degree: HashMap<*const (), usize> = HashMap::new();
        for node in &all_nodes {
            let ptr = node_ptr(node);
            in_degree.entry(ptr).or_insert(0);

            // このノードが参照する各srcノードの入次数を増やす
            for src in &node.src {
                let src_ptr = node_ptr(src);
                *in_degree.entry(src_ptr).or_insert(0) += 1;
            }
        }

        // 3. Kahnのアルゴリズムで世代別にグループ化
        let mut result: TopologicalOrder = Vec::new();
        let mut queue: VecDeque<GraphNode> = VecDeque::new();

        // 入次数が0のノード（誰からも参照されていない=出力ノード）をキューに追加
        for node in &all_nodes {
            let ptr = node_ptr(node);
            if in_degree[&ptr] == 0 {
                queue.push_back(node.clone());
            }
        }

        // 世代ごとに処理
        while !queue.is_empty() {
            let generation_size = queue.len();
            let mut current_generation = Vec::new();

            // 現在の世代のノードをすべて処理
            for _ in 0..generation_size {
                let node = queue.pop_front().unwrap();
                current_generation.push(node.clone());

                // このノードが参照するsrcノードの入次数を減らす
                for src in &node.src {
                    let src_ptr = node_ptr(src);
                    let degree = in_degree.get_mut(&src_ptr).unwrap();
                    *degree -= 1;

                    // 入次数が0になったらキューに追加
                    if *degree == 0 {
                        queue.push_back(src.clone());
                    }
                }
            }

            result.push(current_generation);
        }

        result
    }

    /// グラフの出力ノードから再帰的にすべてのノードを収集する
    fn collect_all_nodes(graph: &Graph) -> Vec<GraphNode> {
        let mut visited: HashSet<*const ()> = HashSet::new();
        let mut nodes: Vec<GraphNode> = Vec::new();

        for output_node in graph.outputs().values() {
            Self::collect_nodes_recursive(output_node, &mut visited, &mut nodes);
        }

        nodes
    }

    /// 再帰的にノードを収集する（深さ優先探索）
    fn collect_nodes_recursive(
        node: &GraphNode,
        visited: &mut HashSet<*const ()>,
        nodes: &mut Vec<GraphNode>,
    ) {
        let ptr = node_ptr(node);

        if visited.contains(&ptr) {
            return;
        }

        visited.insert(ptr);

        // 先にsrcノードを訪問（依存関係の順序）
        for src in &node.src {
            Self::collect_nodes_recursive(src, visited, nodes);
        }

        nodes.push(node.clone());
    }
}

#[cfg(test)]
mod tests;
