//! AST最適化の履歴を記録するモジュール

use crate::ast::AstNode;

/// 最適化の各ステップのスナップショット
#[derive(Clone, Debug)]
pub struct OptimizationSnapshot {
    /// ステップ番号
    pub step: usize,
    /// この時点でのAST
    pub ast: AstNode,
    /// このASTのコスト推定値
    pub cost: f32,
    /// このステップの説明
    pub description: String,
    /// ビーム内の順位（0が最良）
    pub rank: usize,
    /// このステップで適用されたルール名（あれば）
    pub applied_rule: Option<String>,
    /// このステップまでのログ
    pub logs: Vec<String>,
}

impl OptimizationSnapshot {
    /// 新しいスナップショットを作成
    pub fn new(
        step: usize,
        ast: AstNode,
        cost: f32,
        description: String,
        rank: usize,
        applied_rule: Option<String>,
    ) -> Self {
        Self {
            step,
            ast,
            cost,
            description,
            rank,
            applied_rule,
            logs: Vec::new(),
        }
    }

    /// ログ付きで新しいスナップショットを作成
    pub fn with_logs(
        step: usize,
        ast: AstNode,
        cost: f32,
        description: String,
        rank: usize,
        applied_rule: Option<String>,
        logs: Vec<String>,
    ) -> Self {
        Self {
            step,
            ast,
            cost,
            description,
            rank,
            applied_rule,
            logs,
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

    /// 特定のステップのすべてのスナップショット（ビーム内の全候補）を取得
    pub fn get_step(&self, step: usize) -> Vec<&OptimizationSnapshot> {
        self.snapshots.iter().filter(|s| s.step == step).collect()
    }

    /// コストの遷移を取得（最良候補のみ）
    pub fn cost_transition(&self) -> Vec<(usize, f32)> {
        let mut transitions = Vec::new();
        let mut seen_steps = std::collections::HashSet::new();

        for snapshot in &self.snapshots {
            if snapshot.rank == 0 && !seen_steps.contains(&snapshot.step) {
                transitions.push((snapshot.step, snapshot.cost));
                seen_steps.insert(snapshot.step);
            }
        }

        transitions.sort_by_key(|(step, _)| *step);
        transitions
    }
}
