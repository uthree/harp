//! AST最適化の履歴を記録するモジュール

use crate::ast::AstNode;

/// 選択されなかった候補の情報
#[derive(Clone, Debug)]
pub struct AlternativeCandidate {
    /// AST
    pub ast: AstNode,
    /// コスト推定値
    pub cost: f32,
    /// 提案したSuggesterの名前
    pub suggester_name: Option<String>,
    /// 提案の説明
    pub description: String,
    /// ビーム内の順位（0が最良 = 選択された候補）
    pub rank: usize,
}

/// 最適化の各ステップのスナップショット
#[derive(Clone, Debug)]
pub struct OptimizationSnapshot {
    /// ステップ番号
    pub step: usize,
    /// この時点でのAST（選択された候補）
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
    /// このステップでSuggesterが提案した候補の数
    pub num_candidates: Option<usize>,
    /// この候補を提案したSuggesterの名前
    pub suggester_name: Option<String>,
    /// 選択されなかった代替候補（rank > 0の候補）
    pub alternatives: Vec<AlternativeCandidate>,
    /// このASTに至るまでの完全なパス（各ステップの(suggester_name, description)）
    pub path: Vec<(String, String)>,
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
            num_candidates: None,
            suggester_name: None,
            alternatives: Vec::new(),
            path: Vec::new(),
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
            num_candidates: None,
            suggester_name: None,
            alternatives: Vec::new(),
            path: Vec::new(),
        }
    }

    /// 候補数付きで新しいスナップショットを作成
    #[allow(clippy::too_many_arguments)]
    pub fn with_candidates(
        step: usize,
        ast: AstNode,
        cost: f32,
        description: String,
        rank: usize,
        applied_rule: Option<String>,
        logs: Vec<String>,
        num_candidates: usize,
    ) -> Self {
        Self {
            step,
            ast,
            cost,
            description,
            rank,
            applied_rule,
            logs,
            num_candidates: Some(num_candidates),
            suggester_name: None,
            alternatives: Vec::new(),
            path: Vec::new(),
        }
    }

    /// 候補数とSuggester名付きで新しいスナップショットを作成
    #[allow(clippy::too_many_arguments)]
    pub fn with_suggester(
        step: usize,
        ast: AstNode,
        cost: f32,
        description: String,
        rank: usize,
        applied_rule: Option<String>,
        logs: Vec<String>,
        num_candidates: usize,
        suggester_name: Option<String>,
    ) -> Self {
        Self {
            step,
            ast,
            cost,
            description,
            rank,
            applied_rule,
            logs,
            num_candidates: Some(num_candidates),
            suggester_name,
            alternatives: Vec::new(),
            path: Vec::new(),
        }
    }

    /// 代替候補付きでスナップショットを作成
    #[allow(clippy::too_many_arguments)]
    pub fn with_alternatives(
        step: usize,
        ast: AstNode,
        cost: f32,
        description: String,
        rank: usize,
        applied_rule: Option<String>,
        logs: Vec<String>,
        num_candidates: usize,
        suggester_name: Option<String>,
        alternatives: Vec<AlternativeCandidate>,
        path: Vec<(String, String)>,
    ) -> Self {
        Self {
            step,
            ast,
            cost,
            description,
            rank,
            applied_rule,
            logs,
            num_candidates: Some(num_candidates),
            suggester_name,
            alternatives,
            path,
        }
    }
}

/// 最終結果に至る最適化パス
///
/// 各要素は (suggester_name, description) のタプル
pub type FinalOptimizationPath = Vec<(String, String)>;

/// 最適化の履歴全体を保持
#[derive(Clone, Default, Debug)]
pub struct OptimizationHistory {
    /// スナップショットのリスト
    snapshots: Vec<OptimizationSnapshot>,
    /// 最終結果に至る実際のパス
    ///
    /// ビームサーチでは各ステップの最良候補が最終結果に至るとは限らない。
    /// このフィールドは最終結果に実際に至ったパスを記録する。
    final_path: FinalOptimizationPath,
    /// 最適化のターゲットバックエンド
    ///
    /// どのバックエンドを対象として最適化が行われたかを示す。
    /// 可視化ツールでレンダラーを自動選択するために使用される。
    target_backend: crate::backend::TargetBackend,
}

impl OptimizationHistory {
    /// 新しい履歴を作成
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            final_path: Vec::new(),
            target_backend: crate::backend::TargetBackend::Generic,
        }
    }

    /// ターゲットバックエンドを指定して新しい履歴を作成
    pub fn with_target_backend(target_backend: crate::backend::TargetBackend) -> Self {
        Self {
            snapshots: Vec::new(),
            final_path: Vec::new(),
            target_backend,
        }
    }

    /// ターゲットバックエンドを取得
    pub fn target_backend(&self) -> crate::backend::TargetBackend {
        self.target_backend
    }

    /// ターゲットバックエンドを設定
    pub fn set_target_backend(&mut self, backend: crate::backend::TargetBackend) {
        self.target_backend = backend;
    }

    /// 最終結果に至るパスを設定
    ///
    /// ビームサーチ終了時に、最終結果に至った実際のパスを記録する。
    pub fn set_final_path(&mut self, path: FinalOptimizationPath) {
        self.final_path = path;
    }

    /// 最終結果に至るパスを取得
    pub fn final_path(&self) -> &FinalOptimizationPath {
        &self.final_path
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

    /// 候補数の遷移を取得（最良候補のみ）
    pub fn candidate_transition(&self) -> Vec<(usize, usize)> {
        let mut transitions = Vec::new();
        let mut seen_steps = std::collections::HashSet::new();

        for snapshot in &self.snapshots {
            if snapshot.rank == 0 && !seen_steps.contains(&snapshot.step) {
                if let Some(num_candidates) = snapshot.num_candidates {
                    transitions.push((snapshot.step, num_candidates));
                }
                seen_steps.insert(snapshot.step);
            }
        }

        transitions.sort_by_key(|(step, _)| *step);
        transitions
    }
}
