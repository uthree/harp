//! Lowering処理の設定パラメータ
//!
//! このモジュールは、GraphをASTに変換する際の最適化戦略を制御する設定を提供します。

/// Lowering処理の設定パラメータ
///
/// 戦略候補生成に使用される値を一箇所で管理します。
#[derive(Debug, Clone)]
pub struct LoweringConfig {
    /// ベクトル化幅の候補
    ///
    /// 例: `vec![4, 8, 16]` → SIMD幅として試す候補
    /// - 4: SSE相当（128bit、float32 x 4）
    /// - 8: AVX相当（256bit、float32 x 8）
    /// - 16: AVX-512相当（512bit、float32 x 16）
    pub vectorize_widths: Vec<usize>,

    /// タイリングサイズの候補
    ///
    /// 例: `vec![16, 32, 64]` → ループタイリングのサイズ候補
    /// キャッシュフレンドリーなサイズを指定
    pub tile_sizes: Vec<usize>,

    /// アンロール係数の候補
    ///
    /// 例: `vec![2, 4, 8]` → ループアンロールの展開数
    pub unroll_factors: Vec<usize>,

    /// ビームサーチの幅
    ///
    /// 同時に保持する候補数の上限
    /// 大きいほど多くの戦略を試すが、lowering時間が増加
    pub beam_width: usize,

    /// 最適化を有効にするか
    ///
    /// `false` の場合、戦略探索をスキップして従来の動作
    /// デバッグ時や高速なloweringが必要な場合に無効化
    pub enable_optimization: bool,

    /// 並列化を有効にするか
    ///
    /// OpenMP等による並列化戦略を候補に含めるか
    pub enable_parallelization: bool,

    /// ベクトル化を有効にするか
    ///
    /// SIMD命令を使用するベクトル化戦略を候補に含めるか
    pub enable_vectorization: bool,

    /// タイリングを有効にするか
    ///
    /// ループタイリング戦略を候補に含めるか
    pub enable_tiling: bool,
}

impl Default for LoweringConfig {
    fn default() -> Self {
        Self {
            // デフォルトのベクトル幅候補（SSE, AVX, AVX-512相当）
            vectorize_widths: vec![4, 8, 16],

            // デフォルトのタイルサイズ候補（キャッシュフレンドリーなサイズ）
            tile_sizes: vec![16, 32, 64],

            // デフォルトのアンロール係数
            unroll_factors: vec![2, 4, 8],

            // デフォルトのビームサーチ幅
            beam_width: 3,

            // デフォルトで最適化を有効化
            enable_optimization: true,
            enable_parallelization: true,
            enable_vectorization: true,

            // タイリングはデフォルトOFF（実装が複雑なため）
            enable_tiling: false,
        }
    }
}

impl LoweringConfig {
    /// ビルダーパターンで設定を構築
    ///
    /// # Example
    ///
    /// ```
    /// use harp::lowerer::LoweringConfig;
    ///
    /// let config = LoweringConfig::builder()
    ///     .vectorize_widths(vec![8, 16])
    ///     .beam_width(5)
    ///     .build();
    /// ```
    pub fn builder() -> LoweringConfigBuilder {
        LoweringConfigBuilder::new()
    }

    /// 最適化なしの設定を作成
    ///
    /// デバッグや高速なloweringが必要な場合に使用
    pub fn no_optimization() -> Self {
        Self {
            enable_optimization: false,
            ..Default::default()
        }
    }

    /// SSE環境向けの設定を作成
    ///
    /// 128bit SIMD（float32 x 4）のみ
    pub fn for_sse() -> Self {
        Self {
            vectorize_widths: vec![4],
            enable_parallelization: true,
            enable_vectorization: true,
            enable_tiling: false,
            ..Default::default()
        }
    }

    /// AVX環境向けの設定を作成
    ///
    /// 256bit SIMD（float32 x 8）のみ
    pub fn for_avx() -> Self {
        Self {
            vectorize_widths: vec![8],
            enable_parallelization: true,
            enable_vectorization: true,
            enable_tiling: false,
            ..Default::default()
        }
    }

    /// AVX-512環境向けの設定を作成
    ///
    /// 512bit SIMD（float32 x 16）のみ
    pub fn for_avx512() -> Self {
        Self {
            vectorize_widths: vec![16],
            enable_parallelization: true,
            enable_vectorization: true,
            enable_tiling: false,
            ..Default::default()
        }
    }
}

/// ビルダーパターンで設定を構築
pub struct LoweringConfigBuilder {
    config: LoweringConfig,
}

impl LoweringConfigBuilder {
    /// 新しいビルダーを作成（デフォルト設定から開始）
    pub fn new() -> Self {
        Self {
            config: LoweringConfig::default(),
        }
    }

    /// ベクトル化幅の候補を設定
    pub fn vectorize_widths(mut self, widths: Vec<usize>) -> Self {
        self.config.vectorize_widths = widths;
        self
    }

    /// タイリングサイズの候補を設定
    pub fn tile_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.config.tile_sizes = sizes;
        self
    }

    /// アンロール係数の候補を設定
    pub fn unroll_factors(mut self, factors: Vec<usize>) -> Self {
        self.config.unroll_factors = factors;
        self
    }

    /// ビームサーチの幅を設定
    pub fn beam_width(mut self, width: usize) -> Self {
        self.config.beam_width = width;
        self
    }

    /// 最適化を有効/無効にする
    pub fn enable_optimization(mut self, enable: bool) -> Self {
        self.config.enable_optimization = enable;
        self
    }

    /// 並列化を有効/無効にする
    pub fn enable_parallelization(mut self, enable: bool) -> Self {
        self.config.enable_parallelization = enable;
        self
    }

    /// ベクトル化を有効/無効にする
    pub fn enable_vectorization(mut self, enable: bool) -> Self {
        self.config.enable_vectorization = enable;
        self
    }

    /// タイリングを有効/無効にする
    pub fn enable_tiling(mut self, enable: bool) -> Self {
        self.config.enable_tiling = enable;
        self
    }

    /// 設定を構築
    pub fn build(self) -> LoweringConfig {
        self.config
    }
}

impl Default for LoweringConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LoweringConfig::default();
        assert_eq!(config.vectorize_widths, vec![4, 8, 16]);
        assert_eq!(config.tile_sizes, vec![16, 32, 64]);
        assert_eq!(config.beam_width, 3);
        assert!(config.enable_optimization);
        assert!(config.enable_parallelization);
        assert!(config.enable_vectorization);
        assert!(!config.enable_tiling);
    }

    #[test]
    fn test_builder() {
        let config = LoweringConfig::builder()
            .vectorize_widths(vec![8, 16])
            .beam_width(5)
            .enable_tiling(true)
            .build();

        assert_eq!(config.vectorize_widths, vec![8, 16]);
        assert_eq!(config.beam_width, 5);
        assert!(config.enable_tiling);
    }

    #[test]
    fn test_no_optimization() {
        let config = LoweringConfig::no_optimization();
        assert!(!config.enable_optimization);
    }

    #[test]
    fn test_sse_config() {
        let config = LoweringConfig::for_sse();
        assert_eq!(config.vectorize_widths, vec![4]);
        assert!(config.enable_vectorization);
    }

    #[test]
    fn test_avx_config() {
        let config = LoweringConfig::for_avx();
        assert_eq!(config.vectorize_widths, vec![8]);
    }

    #[test]
    fn test_avx512_config() {
        let config = LoweringConfig::for_avx512();
        assert_eq!(config.vectorize_widths, vec![16]);
    }

    #[test]
    fn test_builder_chain() {
        let config = LoweringConfig::builder()
            .vectorize_widths(vec![4])
            .enable_parallelization(false)
            .enable_vectorization(true)
            .enable_tiling(false)
            .beam_width(1)
            .build();

        assert_eq!(config.vectorize_widths, vec![4]);
        assert!(!config.enable_parallelization);
        assert!(config.enable_vectorization);
        assert!(!config.enable_tiling);
        assert_eq!(config.beam_width, 1);
    }
}
