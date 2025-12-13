//! 候補選択のトレイトと実装
//!
//! ビームサーチなどの最適化アルゴリズムで、
//! コスト付き候補から上位n件を選択する処理を抽象化します。
//!
//! # 設計意図
//!
//! tinygradのような多段階評価を可能にするための抽象化です：
//! 1. 静的評価で明らかに悪い候補を足切り
//! 2. 中間的なヒューリスティクスで絞り込み
//! 3. 実行時間の実測値で精密に評価
//!
//! # Example
//!
//! ```ignore
//! use harp::opt::selector::{Selector, MultiStageSelector};
//!
//! // 3段階選択: 静的コスト→メモリ推定→実測
//! let selector = MultiStageSelector::new()
//!     .then(|c| estimate_static_cost(c), 1000)
//!     .then(|c| estimate_memory(c), 100)
//!     .then(|c| measure_runtime(c), 10);
//!
//! let selected = selector.select(candidates, 5);
//! ```

use std::cmp::Ordering;

/// 候補選択のトレイト
///
/// ビームサーチなどの最適化アルゴリズムにおいて、
/// コスト付き候補から上位n件を選択する処理を抽象化します。
///
/// # Type Parameters
///
/// * `T` - 候補の型
pub trait Selector<T> {
    /// 候補とコストのペアから上位n件を選択
    ///
    /// # Arguments
    ///
    /// * `candidates` - (候補, コスト) のペアのベクタ
    /// * `n` - 選択する最大件数
    ///
    /// # Returns
    ///
    /// 選択された (候補, コスト) のベクタ（最大n件）
    fn select(&self, candidates: Vec<(T, f32)>, n: usize) -> Vec<(T, f32)>;
}

/// 静的コストベースの選択器
///
/// コストの昇順でソートして上位n件を選択する最もシンプルな実装。
/// デフォルトの選択器として使用されます。
#[derive(Default, Clone, Copy, Debug)]
pub struct StaticCostSelector;

impl StaticCostSelector {
    /// 新しいStaticCostSelectorを作成
    pub fn new() -> Self {
        Self
    }
}

impl<T> Selector<T> for StaticCostSelector {
    fn select(&self, mut candidates: Vec<(T, f32)>, n: usize) -> Vec<(T, f32)> {
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        candidates.into_iter().take(n).collect()
    }
}

/// 選択ステージ
///
/// 各ステージは評価関数と残す候補数を持ちます。
struct SelectionStage<T> {
    /// 評価関数（候補からコストを計算）
    evaluator: Box<dyn Fn(&T) -> f32>,
    /// このステージで残す候補数
    keep_count: usize,
}

/// 多段階選択器
///
/// メソッドチェーンで複数のステージを構築し、段階的に候補を絞り込みます。
/// 各ステージでは評価関数によりコストを再計算し、上位keep_count件を残します。
///
/// # Example
///
/// ```ignore
/// use harp::opt::selector::{Selector, MultiStageSelector};
///
/// // 3段階選択
/// let selector = MultiStageSelector::new()
///     .then(|c| static_cost(c), 1000)   // 静的コストで1000件に足切り
///     .then(|c| memory_cost(c), 100)    // メモリコストで100件に絞り込み
///     .then(|c| runtime(c), 10);        // 実測で10件を最終選択
///
/// let selected = selector.select(candidates, 5);
/// ```
///
/// # 設計
///
/// [dagopt](https://github.com/uthree/dagopt)の設計を参考にしています。
pub struct MultiStageSelector<T> {
    stages: Vec<SelectionStage<T>>,
}

impl<T> Default for MultiStageSelector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MultiStageSelector<T> {
    /// 新しいMultiStageSelectorを作成
    pub fn new() -> Self {
        Self { stages: vec![] }
    }

    /// 選択ステージを追加
    ///
    /// # Arguments
    ///
    /// * `evaluator` - 候補からコストを計算する評価関数
    /// * `keep_count` - このステージで残す候補数
    ///
    /// # Example
    ///
    /// ```ignore
    /// let selector = MultiStageSelector::new()
    ///     .then(|c| c.node_count() as f32, 100)
    ///     .then(|c| measure_runtime(c), 10);
    /// ```
    pub fn then<F>(mut self, evaluator: F, keep_count: usize) -> Self
    where
        F: Fn(&T) -> f32 + 'static,
    {
        self.stages.push(SelectionStage {
            evaluator: Box::new(evaluator),
            keep_count,
        });
        self
    }

    /// ステージ数を取得
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl<T> Selector<T> for MultiStageSelector<T> {
    fn select(&self, mut candidates: Vec<(T, f32)>, n: usize) -> Vec<(T, f32)> {
        // ステージがない場合は静的コストで選択
        if self.stages.is_empty() {
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            return candidates.into_iter().take(n).collect();
        }

        for (i, stage) in self.stages.iter().enumerate() {
            // 各候補のコストを再計算
            candidates = candidates
                .into_iter()
                .map(|(candidate, _old_cost)| {
                    let new_cost = (stage.evaluator)(&candidate);
                    (candidate, new_cost)
                })
                .collect();

            // コストでソート
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            // 最後のステージでは n を考慮、それ以外は keep_count で截断
            let limit = if i == self.stages.len() - 1 {
                n.min(stage.keep_count)
            } else {
                stage.keep_count
            };

            candidates.truncate(limit);
        }

        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_cost_selector_basic() {
        let selector = StaticCostSelector::new();
        let candidates = vec![("a", 3.0), ("b", 1.0), ("c", 2.0)];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, "b"); // コスト1.0
        assert_eq!(selected[1].0, "c"); // コスト2.0
    }

    #[test]
    fn test_static_cost_selector_all() {
        let selector = StaticCostSelector::new();
        let candidates = vec![("a", 3.0), ("b", 1.0)];

        let selected = selector.select(candidates, 10);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_static_cost_selector_empty() {
        let selector = StaticCostSelector::new();
        let candidates: Vec<(&str, f32)> = vec![];

        let selected = selector.select(candidates, 5);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_static_cost_selector_zero_n() {
        let selector = StaticCostSelector::new();
        let candidates = vec![("a", 1.0), ("b", 2.0)];

        let selected = selector.select(candidates, 0);
        assert!(selected.is_empty());
    }

    #[test]
    fn test_static_cost_selector_with_nan() {
        let selector = StaticCostSelector::new();
        let candidates = vec![("a", f32::NAN), ("b", 1.0), ("c", 2.0)];

        let selected = selector.select(candidates, 2);
        // NaNの扱いは未定義だが、パニックしないことを確認
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_multi_stage_selector_single_stage() {
        // 単一ステージ: 静的コストとは異なる評価関数を使用
        let evaluator = |s: &&str| -> f32 {
            match *s {
                "a" => 1.0, // 静的コスト3.0だが評価では最良
                "b" => 2.0, // 静的コスト1.0だが評価では2番目
                "c" => 3.0, // 静的コスト2.0だが評価では最悪
                _ => f32::MAX,
            }
        };

        let selector = MultiStageSelector::new().then(evaluator, 3);
        let candidates = vec![("a", 3.0), ("b", 1.0), ("c", 2.0)];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, "a"); // 評価コスト1.0
        assert_eq!(selected[1].0, "b"); // 評価コスト2.0
    }

    #[test]
    fn test_multi_stage_selector_two_stages() {
        // 2段階選択
        // Stage 1: 静的コストで3件に足切り
        // Stage 2: 別の評価関数で最終選択
        let stage1_eval = |s: &&str| -> f32 {
            match *s {
                "a" => 1.0,
                "b" => 2.0,
                "c" => 3.0,
                "d" => 4.0,
                "e" => 5.0,
                _ => f32::MAX,
            }
        };
        let stage2_eval = |s: &&str| -> f32 {
            match *s {
                "a" => 3.0, // Stage1で最良だがStage2で最悪
                "b" => 2.0,
                "c" => 1.0, // Stage1で3番だがStage2で最良
                _ => f32::MAX,
            }
        };

        let selector = MultiStageSelector::new()
            .then(stage1_eval, 3) // a, b, c が残る
            .then(stage2_eval, 2); // c, b が選ばれる

        let candidates = vec![("a", 0.0), ("b", 0.0), ("c", 0.0), ("d", 0.0), ("e", 0.0)];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, "c"); // Stage2コスト1.0
        assert_eq!(selected[1].0, "b"); // Stage2コスト2.0
    }

    #[test]
    fn test_multi_stage_selector_three_stages() {
        // 3段階選択
        let stage1 = |s: &&str| -> f32 {
            match *s {
                "a" => 1.0,
                "b" => 2.0,
                "c" => 3.0,
                "d" => 4.0,
                _ => f32::MAX,
            }
        };
        let stage2 = |s: &&str| -> f32 {
            match *s {
                "a" => 3.0,
                "b" => 1.0,
                "c" => 2.0,
                _ => f32::MAX,
            }
        };
        let stage3 = |s: &&str| -> f32 {
            match *s {
                "b" => 2.0,
                "c" => 1.0, // 最終的に最良
                _ => f32::MAX,
            }
        };

        let selector = MultiStageSelector::new()
            .then(stage1, 3) // a, b, c
            .then(stage2, 2) // b, c
            .then(stage3, 1); // c

        let candidates = vec![("a", 0.0), ("b", 0.0), ("c", 0.0), ("d", 0.0)];

        let selected = selector.select(candidates, 1);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].0, "c");
    }

    #[test]
    fn test_multi_stage_selector_no_stages() {
        // ステージがない場合は静的コストで選択
        let selector: MultiStageSelector<&str> = MultiStageSelector::new();
        let candidates = vec![("a", 3.0), ("b", 1.0), ("c", 2.0)];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, "b"); // 静的コスト1.0
        assert_eq!(selected[1].0, "c"); // 静的コスト2.0
    }

    #[test]
    fn test_multi_stage_selector_cutoff() {
        // 足切りテスト: Stage1で "d" が除外される
        let stage1 = |s: &&str| -> f32 {
            match *s {
                "a" => 1.0,
                "b" => 2.0,
                "c" => 3.0,
                "d" => 10.0, // 足切り対象
                _ => f32::MAX,
            }
        };
        let stage2 = |s: &&str| -> f32 {
            match *s {
                "d" => 0.0, // 本来最良だが足切り済み
                "a" => 3.0,
                "b" => 2.0,
                "c" => 1.0,
                _ => f32::MAX,
            }
        };

        let selector = MultiStageSelector::new()
            .then(stage1, 3) // a, b, c (dは足切り)
            .then(stage2, 2);

        let candidates = vec![("a", 0.0), ("b", 0.0), ("c", 0.0), ("d", 0.0)];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        // dは足切りされたので、c, bが選ばれる
        assert_eq!(selected[0].0, "c"); // Stage2コスト1.0
        assert_eq!(selected[1].0, "b"); // Stage2コスト2.0
    }

    #[test]
    fn test_multi_stage_selector_n_smaller_than_keep() {
        // n が keep_count より小さい場合
        let eval = |s: &&str| -> f32 {
            match *s {
                "a" => 1.0,
                "b" => 2.0,
                "c" => 3.0,
                _ => f32::MAX,
            }
        };

        let selector = MultiStageSelector::new().then(eval, 10); // keep_count = 10

        let candidates = vec![("a", 0.0), ("b", 0.0), ("c", 0.0)];

        // n = 2 を指定
        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, "a");
        assert_eq!(selected[1].0, "b");
    }

    #[test]
    fn test_multi_stage_selector_stage_count() {
        let selector: MultiStageSelector<i32> = MultiStageSelector::new()
            .then(|_| 0.0, 10)
            .then(|_| 0.0, 5)
            .then(|_| 0.0, 2);

        assert_eq!(selector.stage_count(), 3);
    }

    #[test]
    fn test_multi_stage_selector_empty_candidates() {
        let selector = MultiStageSelector::new().then(|_: &&str| 0.0, 10);
        let candidates: Vec<(&str, f32)> = vec![];

        let selected = selector.select(candidates, 5);
        assert!(selected.is_empty());
    }
}
