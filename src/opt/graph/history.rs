//! グラフ最適化の履歴を記録するモジュール

use crate::graph::Graph;

/// 最適化の各ステップのスナップショット
#[derive(Clone, Debug)]
pub struct OptimizationSnapshot {
    /// ステップ番号
    pub step: usize,
    /// この時点でのグラフ
    pub graph: Graph,
    /// このグラフのコスト推定値
    pub cost: f32,
    /// このステップの説明
    pub description: String,
    /// このステップまでのログ
    pub logs: Vec<String>,
    /// このステップでSuggesterが提案した候補の数
    pub num_candidates: Option<usize>,
}

impl OptimizationSnapshot {
    /// 新しいスナップショットを作成
    pub fn new(step: usize, graph: Graph, cost: f32, description: String) -> Self {
        Self {
            step,
            graph,
            cost,
            description,
            logs: Vec::new(),
            num_candidates: None,
        }
    }

    /// ログ付きで新しいスナップショットを作成
    pub fn with_logs(
        step: usize,
        graph: Graph,
        cost: f32,
        description: String,
        logs: Vec<String>,
    ) -> Self {
        Self {
            step,
            graph,
            cost,
            description,
            logs,
            num_candidates: None,
        }
    }

    /// 候補数付きで新しいスナップショットを作成
    pub fn with_candidates(
        step: usize,
        graph: Graph,
        cost: f32,
        description: String,
        logs: Vec<String>,
        num_candidates: usize,
    ) -> Self {
        Self {
            step,
            graph,
            cost,
            description,
            logs,
            num_candidates: Some(num_candidates),
        }
    }
}

/// 最適化の履歴全体を保持
#[derive(Clone, Default, Debug)]
pub struct OptimizationHistory {
    /// スナップショットのリスト
    snapshots: Vec<OptimizationSnapshot>,
}

impl OptimizationHistory {
    /// 新しい履歴を作成
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    /// スナップショットを追加
    pub fn add_snapshot(&mut self, snapshot: OptimizationSnapshot) {
        self.snapshots.push(snapshot);
    }

    /// すべてのスナップショットを取得
    pub fn snapshots(&self) -> &[OptimizationSnapshot] {
        &self.snapshots
    }

    /// スナップショット数を取得
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// 履歴が空かどうか
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// 特定のステップのスナップショットを取得
    pub fn get(&self, step: usize) -> Option<&OptimizationSnapshot> {
        self.snapshots.get(step)
    }

    /// コストの遷移を取得
    pub fn cost_transition(&self) -> Vec<(usize, f32)> {
        self.snapshots.iter().map(|s| (s.step, s.cost)).collect()
    }

    /// 候補数の遷移を取得
    pub fn candidate_transition(&self) -> Vec<(usize, usize)> {
        self.snapshots
            .iter()
            .filter_map(|s| s.num_candidates.map(|n| (s.step, n)))
            .collect()
    }
}
