//! 並列化戦略の定義
//!
//! Element-wise演算、Reduce演算、Cumulative演算の各軸に対する
//! 並列化戦略とSIMD化、ループアンローリングの設定を提供します。

/// Element-wise演算の各軸の並列化戦略
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ElementwiseStrategy {
    /// 逐次実行（SIMD幅: 1=SIMD化なし、2以上=SIMD化、アンローリング係数: デフォルト1）
    Sequential {
        simd_width: usize,
        unroll_factor: usize,
    },
    /// スレッドで並列化（SIMD幅: 1=SIMD化なし、2以上=SIMD化、アンローリング係数: デフォルト1）
    Thread {
        simd_width: usize,
        unroll_factor: usize,
    },
    /// スレッドグループ/ブロック（SIMD幅: 1=SIMD化なし、2以上=SIMD化、アンローリング係数: デフォルト1）
    ThreadGroup {
        simd_width: usize,
        unroll_factor: usize,
    },
}

impl Default for ElementwiseStrategy {
    fn default() -> Self {
        Self::Sequential {
            simd_width: 1,
            unroll_factor: 1,
        }
    }
}

/// ElementwiseStrategyのビルダーメソッドを生成するマクロ
macro_rules! impl_elementwise_strategy_builders {
    ($variant:ident, $prefix:ident) => {
        paste::paste! {
            #[doc = "SIMD化なしの" $variant "を作成"]
            pub fn $prefix() -> Self {
                Self::$variant {
                    simd_width: 1,
                    unroll_factor: 1,
                }
            }

            #[doc = "SIMD化ありの" $variant "を作成"]
            pub fn [<$prefix _simd>](simd_width: usize) -> Self {
                Self::$variant {
                    simd_width,
                    unroll_factor: 1,
                }
            }

            #[doc = "アンローリングありの" $variant "を作成"]
            pub fn [<$prefix _unroll>](unroll_factor: usize) -> Self {
                Self::$variant {
                    simd_width: 1,
                    unroll_factor,
                }
            }

            #[doc = "SIMD化とアンローリング両方ありの" $variant "を作成"]
            pub fn [<$prefix _simd_unroll>](simd_width: usize, unroll_factor: usize) -> Self {
                Self::$variant {
                    simd_width,
                    unroll_factor,
                }
            }
        }
    };
}

impl ElementwiseStrategy {
    impl_elementwise_strategy_builders!(Sequential, sequential);
    impl_elementwise_strategy_builders!(Thread, thread);
    impl_elementwise_strategy_builders!(ThreadGroup, thread_group);

    /// SIMD幅を取得
    pub fn simd_width(&self) -> usize {
        match self {
            Self::Sequential { simd_width, .. } => *simd_width,
            Self::Thread { simd_width, .. } => *simd_width,
            Self::ThreadGroup { simd_width, .. } => *simd_width,
        }
    }

    /// アンローリング係数を取得
    pub fn unroll_factor(&self) -> usize {
        match self {
            Self::Sequential { unroll_factor, .. } => *unroll_factor,
            Self::Thread { unroll_factor, .. } => *unroll_factor,
            Self::ThreadGroup { unroll_factor, .. } => *unroll_factor,
        }
    }
}

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

/// ReduceStrategyとCumulativeStrategyのビルダーメソッドを生成するマクロ
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

/// Cumulative演算の並列化戦略
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CumulativeStrategy {
    /// 逐次実行（アンローリング係数: デフォルト1）
    /// 将来的にはParallel Scan（Hillis-Steele、Blelloch等）を追加予定
    Sequential { unroll_factor: usize },
}

impl Default for CumulativeStrategy {
    fn default() -> Self {
        Self::Sequential { unroll_factor: 1 }
    }
}

impl_unroll_only_strategy_builders!(CumulativeStrategy, Sequential, sequential);
