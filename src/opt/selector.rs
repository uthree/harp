//! 候補選択のトレイトと実装
//!
//! ビームサーチなどの最適化アルゴリズムで、
//! コスト付き候補から上位n件を選択する処理を抽象化します。
//!
//! # 設計意図
//!
//! tinygradのような二段階評価を可能にするための抽象化です：
//! 1. 静的評価で明らかに悪い候補を足切り
//! 2. 実行時間の実測値で精密に評価
//!
//! # Example
//!
//! ```ignore
//! use harp::opt::selector::{Selector, StaticCostSelector};
//!
//! let selector = StaticCostSelector;
//! let candidates = vec![
//!     ("a", 3.0),
//!     ("b", 1.0),
//!     ("c", 2.0),
//! ];
//!
//! let selected = selector.select(candidates, 2);
//! assert_eq!(selected[0].0, "b"); // コスト1.0
//! assert_eq!(selected[1].0, "c"); // コスト2.0
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

/// 二段階選択器
///
/// 静的コストで候補を足切りした後、カスタム評価関数で再評価して最終選択を行います。
/// tinygradスタイルの最適化を実現するための選択器です。
///
/// # Example
///
/// ```ignore
/// use harp::opt::selector::{Selector, TwoStageSelector};
///
/// // 静的コストで10件に足切り → 実測で上位3件を選択
/// let selector = TwoStageSelector::new(10, |candidate| {
///     // 実際の実行時間を計測
///     measure_runtime(candidate)
/// });
///
/// let selected = selector.select(candidates, 3);
/// ```
pub struct TwoStageSelector<F> {
    /// 一次選択（足切り）で残す候補数
    first_stage_count: usize,
    /// 二次評価関数（候補から精密なコストを計算）
    evaluator: F,
}

impl<F> TwoStageSelector<F> {
    /// 新しいTwoStageSelectorを作成
    ///
    /// # Arguments
    ///
    /// * `first_stage_count` - 一次選択で残す候補数
    /// * `evaluator` - 二次評価関数 `Fn(&T) -> f32`
    pub fn new(first_stage_count: usize, evaluator: F) -> Self {
        Self {
            first_stage_count,
            evaluator,
        }
    }
}

impl<T, F> Selector<T> for TwoStageSelector<F>
where
    F: Fn(&T) -> f32,
{
    fn select(&self, mut candidates: Vec<(T, f32)>, n: usize) -> Vec<(T, f32)> {
        // 一次選択: 静的コストで足切り
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let first_stage: Vec<(T, f32)> = candidates
            .into_iter()
            .take(self.first_stage_count)
            .collect();

        // 二次選択: 精密評価で再ソート
        let mut re_evaluated: Vec<(T, f32)> = first_stage
            .into_iter()
            .map(|(candidate, _static_cost)| {
                let precise_cost = (self.evaluator)(&candidate);
                (candidate, precise_cost)
            })
            .collect();

        re_evaluated.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        re_evaluated.into_iter().take(n).collect()
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
    fn test_two_stage_selector_basic() {
        // 静的コストとは異なる精密コストを返す評価関数
        let evaluator = |s: &&str| -> f32 {
            match *s {
                "a" => 1.0, // 静的コスト3.0だが実測では最良
                "b" => 2.0, // 静的コスト1.0だが実測では2番目
                "c" => 3.0, // 静的コスト2.0だが実測では最悪
                _ => f32::MAX,
            }
        };

        let selector = TwoStageSelector::new(3, evaluator);
        let candidates = vec![("a", 3.0), ("b", 1.0), ("c", 2.0)];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, "a"); // 実測コスト1.0
        assert_eq!(selected[1].0, "b"); // 実測コスト2.0
    }

    #[test]
    fn test_two_stage_selector_cutoff() {
        // 足切りにより"a"は二次評価に進めない
        let evaluator = |s: &&str| -> f32 {
            match *s {
                "a" => 0.1, // 実測では最良だが足切り
                "b" => 2.0,
                "c" => 1.0, // 足切り後では最良
                "d" => 3.0,
                _ => f32::MAX,
            }
        };

        let selector = TwoStageSelector::new(2, evaluator);
        let candidates = vec![
            ("a", 10.0), // 静的コスト最悪→足切り
            ("b", 1.0),  // 静的コスト1番
            ("c", 2.0),  // 静的コスト2番
            ("d", 3.0),  // 静的コスト3番→足切り
        ];

        let selected = selector.select(candidates, 2);
        assert_eq!(selected.len(), 2);
        // b, c のみが二次評価に進む
        assert_eq!(selected[0].0, "c"); // 実測コスト1.0
        assert_eq!(selected[1].0, "b"); // 実測コスト2.0
    }

    #[test]
    fn test_two_stage_selector_first_stage_larger_than_candidates() {
        let evaluator = |s: &&str| -> f32 {
            match *s {
                "a" => 2.0,
                "b" => 1.0,
                _ => f32::MAX,
            }
        };

        let selector = TwoStageSelector::new(100, evaluator);
        let candidates = vec![("a", 1.0), ("b", 2.0)];

        let selected = selector.select(candidates, 1);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].0, "b"); // 実測コスト1.0
    }
}
