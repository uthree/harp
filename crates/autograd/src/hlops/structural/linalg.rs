//! 線形代数の高級演算（Matmul フォールバック）
//!
//! primops の組み合わせで実装される線形代数演算。
//! primops::Matmul は効率的な特殊化実装（BLAS など）用で、
//! こちらは汎用的なフォールバック実装を提供する。
//!
//! ## Matmul フォールバック設計
//!
//! 行列積 C = A @ B（A: [M, K], B: [K, N] → C: [M, N]）は
//! 以下の primops の組み合わせで表現できる：
//!
//! ```text
//! 1. Reshape: A [M, K] → [M, K, 1]
//! 2. Reshape: B [K, N] → [1, K, N]
//! 3. Expand:  A [M, K, 1] → [M, K, N]
//! 4. Expand:  B [1, K, N] → [M, K, N]
//! 5. Mul:     A * B → [M, K, N]  (要素ごとの積)
//! 6. Sum:     axis=1 で縮約 → [M, 1, N]
//! 7. Reshape: [M, 1, N] → [M, N]
//! ```
//!
//! ## 実装状況
//!
//! primops は以下が実装済み：
//! - [x] Expand (primops::structural::reduce)
//! - [x] Sum (primops::structural::reduce)
//! - [x] Reshape (primops::structural::reshape)
//! - [x] Permute (primops::structural::permute)
//!
//! ## 現状の制限
//!
//! 現在の Reshape トレイトは `fn reshape(&self, new_shape: &[usize]) -> Self` という
//! シグネチャで、戻り値が `Self` のため **次元数を変えることができない**。
//!
//! 上記のフォールバック実装には 2D → 3D → 2D の変換が必要だが、
//! 静的型システム（`Array2<T>` など）ではこれを表現できない。
//!
//! 完全な実装には以下のいずれかが必要：
//! 1. 動的次元テンソル（`ArrayD<T>` / `IxDyn`）専用の実装
//! 2. Unsqueeze/Squeeze トレイトの追加（次元の追加・削除）
//!
//! ## 代替案
//!
//! - **特殊化実装を使用**: `primops::Matmul` トレイトを実装したバックエンド
//!   （ndarray など）では、効率的な特殊化実装が使用される
//! - **動的次元版**: 将来的に `ArrayD<T>` 向けのフォールバックを追加可能
