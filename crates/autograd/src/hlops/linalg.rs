//! 線形代数演算（Matmul など）
//!
//! これらはprimopsの組み合わせで実装されます：
//! - Matmul = Expand + Mul + Sum
//!
//! 現在は未実装（shape管理が複雑なため後回し）

// TODO: Matmul の実装
// 行列積 C = A @ B は以下のprimopsで表現：
// 1. Aを(M, 1, K)に reshape
// 2. Bを(1, K, N)に reshape
// 3. ExpandでAを(M, K, N)に、Bを(M, K, N)に拡張
// 4. 要素ごとのMul
// 5. axis=1 (K次元)でSum → (M, N)
