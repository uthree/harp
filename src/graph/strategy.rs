//! 並列化戦略の定義
//!
//! Reduce演算の各軸に対する
//! 並列化戦略とループアンローリングの設定を提供します。
//!
//! Note: ElementwiseStrategyは削除されました。
//! 並列化はASTレベルで直接制御されます。

/// Reduce演算の並列化戦略
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReduceStrategy {
    /// 逐次実行（アンローリング係数: デフォルト1）
    /// 将来的には並列リダクションアルゴリズムなどを追加予定
    Sequential { unroll_factor: usize },
}

impl Default for ReduceStrategy {
    fn default() -> Self {
        Self::Sequential { unroll_factor: 1 }
    }
}

/// ReduceStrategyのビルダーメソッドを生成するマクロ
macro_rules! impl_unroll_only_strategy_builders {
    ($type:ident, $variant:ident, $prefix:ident) => {
        paste::paste! {
            impl $type {
                #[doc = $variant "を作成"]
                pub fn $prefix() -> Self {
                    Self::$variant { unroll_factor: 1 }
                }

                #[doc = "アンローリングありの" $variant "を作成"]
                pub fn [<$prefix _unroll>](unroll_factor: usize) -> Self {
                    Self::$variant { unroll_factor }
                }

                /// アンローリング係数を取得
                pub fn unroll_factor(&self) -> usize {
                    match self {
                        Self::$variant { unroll_factor } => *unroll_factor,
                    }
                }
            }
        }
    };
}

impl_unroll_only_strategy_builders!(ReduceStrategy, Sequential, sequential);
