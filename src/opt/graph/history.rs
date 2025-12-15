//! グラフ最適化の履歴を記録するモジュール

use crate::graph::Graph;

/// 選択されなかった候補の情報
#[derive(Clone, Debug)]
pub struct AlternativeCandidate {
    /// グラフ
    pub graph: Graph,
    /// コスト推定値
    pub cost: f32,
    /// 提案したSuggesterの名前
    pub suggester_name: String,
    /// ビーム内の順位（0が最良 = 選択された候補）
    pub rank: usize,
}

/// 最適化の各ステップのスナップショット
#[derive(Clone, Debug)]
pub struct OptimizationSnapshot {
    /// ステップ番号
    pub step: usize,
    /// この時点でのグラフ（選択された候補）
    pub graph: Graph,
    /// このグラフのコスト推定値
    pub cost: f32,
    /// このステップの説明
    pub description: String,
    /// このステップまでのログ
    pub logs: Vec<String>,
    /// このステップでSuggesterが提案した候補の数
    pub num_candidates: Option<usize>,
    /// このグラフを提案したSuggesterの名前
    pub suggester_name: Option<String>,
    /// 選択されなかった代替候補（rank > 0の候補）
    pub alternatives: Vec<AlternativeCandidate>,
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
            suggester_name: None,
            alternatives: Vec::new(),
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
            suggester_name: None,
            alternatives: Vec::new(),
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
            suggester_name: None,
            alternatives: Vec::new(),
        }
    }

    /// Suggester名付きでスナップショットを作成
    pub fn with_suggester(
        step: usize,
        graph: Graph,
        cost: f32,
        description: String,
        logs: Vec<String>,
        num_candidates: usize,
        suggester_name: String,
    ) -> Self {
        Self {
            step,
            graph,
            cost,
            description,
            logs,
            num_candidates: Some(num_candidates),
            suggester_name: Some(suggester_name),
            alternatives: Vec::new(),
        }
    }

    /// 代替候補付きでスナップショットを作成
    #[allow(clippy::too_many_arguments)]
    pub fn with_alternatives(
        step: usize,
        graph: Graph,
        cost: f32,
        description: String,
        logs: Vec<String>,
        num_candidates: usize,
        suggester_name: String,
        alternatives: Vec<AlternativeCandidate>,
    ) -> Self {
        Self {
            step,
            graph,
            cost,
            description,
            logs,
            num_candidates: Some(num_candidates),
            suggester_name: Some(suggester_name),
            alternatives,
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

    /// 別の履歴を結合（ステップ番号を調整して連結）
    ///
    /// 結合時に、追加する履歴のステップ番号は現在の最大ステップ番号+1から開始するように調整されます。
    /// また、説明にはフェーズ名のプレフィックスを追加できます。
    ///
    /// # Arguments
    /// * `other` - 結合する履歴
    /// * `phase_name` - フェーズ名（descriptionのプレフィックスとして使用）
    pub fn extend_with_phase(&mut self, other: OptimizationHistory, phase_name: &str) {
        if other.is_empty() {
            return;
        }

        // 現在の最大ステップ番号を取得
        let base_step_offset = self.snapshots.last().map(|s| s.step + 1).unwrap_or(0);

        // Phase 2の初期スナップショット（step=0でsuggeseter_nameがNone）はPhase 1の最後と
        // 同じグラフなのでスキップする。スキップした場合、step番号を1減らして連続性を保つ。
        let skip_initial = !self.snapshots.is_empty()
            && other
                .snapshots
                .first()
                .map(|s| s.step == 0 && s.suggester_name.is_none())
                .unwrap_or(false);
        let step_offset = if skip_initial {
            base_step_offset.saturating_sub(1)
        } else {
            base_step_offset
        };

        // 他の履歴のスナップショットをステップ番号を調整して追加
        for snapshot in other.snapshots {
            // 初期スナップショット（suggester情報なし）をスキップ
            if skip_initial && snapshot.step == 0 && snapshot.suggester_name.is_none() {
                continue;
            }

            // 既存のdescriptionが[...]プレフィックスを持つ場合は置き換え、そうでなければ追加
            let description = if snapshot.description.starts_with('[') {
                if let Some(bracket_end) = snapshot.description.find(']') {
                    // 既存のプレフィックスをフェーズ名に置き換え
                    format!(
                        "[{}]{}",
                        phase_name,
                        &snapshot.description[bracket_end + 1..]
                    )
                } else {
                    format!("[{}] {}", phase_name, snapshot.description)
                }
            } else {
                format!("[{}] {}", phase_name, snapshot.description)
            };
            let adjusted_snapshot = OptimizationSnapshot {
                step: snapshot.step + step_offset,
                graph: snapshot.graph,
                cost: snapshot.cost,
                description,
                logs: snapshot.logs,
                num_candidates: snapshot.num_candidates,
                suggester_name: snapshot.suggester_name,
                alternatives: snapshot.alternatives,
            };
            self.snapshots.push(adjusted_snapshot);
        }
    }

    /// 複数の履歴をフェーズ名付きで結合
    ///
    /// # Arguments
    /// * `histories` - (フェーズ名, 履歴) のタプルのスライス
    pub fn from_phases(histories: &[(String, OptimizationHistory)]) -> Self {
        let mut combined = Self::new();
        for (phase_name, history) in histories {
            combined.extend_with_phase(history.clone(), phase_name);
        }
        combined
    }
}
