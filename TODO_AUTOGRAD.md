# Autograd実装の問題点・改善タスク

## 概要
2層パーセプトロンのデモ作成時に発覚した、autogradシステムの問題点と改善タスク。

---

## 1. View操作のバックワード未実装 ✅ **修正済み**

### 問題
`unsqueeze`, `expand`, `reshape` などのview操作で新しい `TensorInner` を作成する際に `autograd: None` が設定され、勾配追跡が途切れる。

### 修正内容
各view操作に対応するバックワード関数を実装：
- `SqueezeBackward` → unsqueezeで勾配を戻す
- `UnsqueezeBackward` → squeeze で勾配を戻す
- `ReshapeBackward` → 元の形状にreshapeして勾配を戻す
- `ExpandBackward` → ブロードキャスト次元に沿ってsumで勾配を集約
- `PermuteBackward` → 逆順のpermutationで勾配を戻す
- `TransposeBackward` → 逆転置で勾配を戻す

### 状態
**修正済み**: `src/tensor/primops/movement.rs` に各バックワード関数を実装し、FloatDType用のview操作に勾配追跡を追加。

---

## 2. TensorInner::new の autograd デフォルト値

### 問題
`TensorInner::new` が常に `autograd: None` を設定している。

### 該当箇所
- `src/tensor/mod.rs:476-487`

```rust
pub fn new(op: TensorOp, view: View, shape: Vec<usize>, dtype: DType) -> Self {
    Self {
        op,
        view,
        shape,
        dtype,
        name: None,
        autograd: None,  // ← 常にNone
        buffer: RwLock::new(None),
    }
}
```

### 考慮事項
autogradを引き継ぐ新しいコンストラクタを追加するか、または各演算で明示的にautogradを設定する方針を検討。

---

## 3. matmul2 の autograd 情報破棄（修正済み）

### 問題
`matmul2` の結果を `Dim2` に変換する際に autograd 情報を破棄していた。

### 該当箇所
- `src/tensor/hlops/linalg.rs:107-120`

### 状態
**修正済み**: `result_dyn.inner.clone()` を使用してautograd情報を保持するよう変更。

---

## 4. outer1/dot1 の autograd 情報破棄 ✅ **修正済み**

### 問題
`matmul2` と同様に、`outer1` と `dot1` も結果を変換する際にautograd情報を破棄している可能性。

### 該当箇所
- `src/tensor/hlops/linalg.rs` の `outer1` (53-66行目)
- `src/tensor/hlops/linalg.rs` の `dot1` は DimDyn を返すため問題なさそう

### 状態
**修正済み**: `outer1` も `result_dyn.inner.clone()` を使用してautograd情報を保持するよう変更。

---

## 5. Tensor::rand の OpenCL 非対応 ✅ **修正済み**

### 問題
`Tensor::rand()` が生成するカーネルコードで C言語の `rand()` と `RAND_MAX` を使用しているが、OpenCL にはこれらの関数がない。

### 該当エラー
```
error: implicit declaration of function 'rand' is invalid in OpenCL
error: use of undeclared identifier 'RAND_MAX'
```

### 修正内容
CPU側で乱数を事前に生成してVecBufferに格納するように変更：
- `generate_random_f32()` / `generate_random_f64()` ヘルパー関数を追加（Xorshift64 PRNG使用）
- `Tensor<f32, Dim<N>>::rand()` を修正: `TensorOp::Rand` の代わりに `TensorOp::Executed` + VecBuffer
- `Tensor<f64, Dim<N>>::rand()` を修正: 同上
- `Tensor<T: FloatDType, DimDyn>::rand_dyn()` を修正: DTypeに応じて適切な乱数生成関数を呼び出し

### 状態
**修正済み**: 乱数はCPU側で生成されるため、全てのバックエンド（OpenCL, Metal等）で動作する。

---

## 6. VecBuffer から OpenCL への転送問題 ✅ **修正済み**

### 問題
`from_ndarray` で作成したテンソル（VecBuffer使用）をOpenCLで実行する際に、カーネル引数数のミスマッチが発生。

### 該当エラー
```
The wrong number of kernel arguments have been specified (required: 3, specified: 2)
```

### 原因
`TensorLowerer` の `build_input_expr` 関連メソッドで、同じテンソルが複数回参照される場合（例: `a + a`）に毎回新しいバッファインデックスを割り当てていた。これにより、カーネルが期待する引数の数と実際に渡されるバッファの数にミスマッチが生じた。

### 修正内容
`src/tensor/lowerer/mod.rs`:
- `TensorLowerer` に `buffer_index_map: HashMap<*const TensorInner, usize>` を追加
- `build_input_expr`, `build_input_expr_linear`, `build_input_expr_with_view` を修正し、同じテンソルに対しては同じバッファインデックスを再利用するよう変更
- テストケース追加: `test_lower_same_tensor_multiple_references`, `test_lower_different_tensors`, `test_lower_mixed_tensor_references`

### 状態
**修正済み**: 同じテンソルが複数回参照される場合でも正しい引数数でカーネルを生成するようになった。

---

## 7. デバイス未設定時のエラーメッセージ

### 問題
デバイスが設定されていない状態で `realize()` を呼ぶとエラーになる。これ自体は正しい動作だが、純粋なCPU実行バックエンドがない。

### 現状
HarpはGPU/アクセラレータ向けに設計されており、純粋なCPU実行（VecBufferのみ）での計算実行は未サポート。

### 検討事項
- CPUバックエンド（CDevice + VecBuffer）での計算実行サポートを追加するか
- または、これはHarpの設計意図通りであり、GPU/OpenCL/Metalが必須という方針を明確化するか

---

## 8. スカラー演算の勾配追跡未対応 ✅ **修正済み**

### 問題
`Tensor * f32`、`Tensor + f32`、`Tensor - f32` などのスカラー演算は勾配追跡を設定していない。
これにより、`tanh()` や `sigmoid()` などの活性化関数内でスカラー演算を使用すると勾配追跡が途切れる。

### 修正内容
- `ScalarAddBackward` 構造体を追加: スカラー加算の勾配（勾配をそのまま通す）
- `ScalarMulBackward` 構造体を追加: スカラー乗算の勾配（スカラー値で勾配をスケール）
- `impl_scalar_ops!` マクロを廃止し、f32/f64 個別の実装に置き換え
- 各演算で `requires_grad()` チェックを行い、勾配追跡を設定

### 状態
**修正済み**: `src/tensor/primops/binary.rs` に `ScalarAddBackward` と `ScalarMulBackward` を実装。
`tanh()` や `sigmoid()` などのスカラー演算を使用するhlopsで勾配が正しく流れるようになった。

---

## 優先度

1. ~~**高**: View操作のバックワード実装 (autograd全体に影響)~~ ✅ **完了**
2. ~~**高**: VecBuffer → OpenCL転送の修正 (基本的な実行に影響)~~ ✅ **完了**
3. ~~**中**: Tensor::rand のOpenCL対応~~ ✅ **完了**
4. ~~**低**: outer1 の autograd 修正~~ ✅ **完了**
5. ~~**中**: スカラー演算の勾配追跡対応（tanh/sigmoid等に必要）~~ ✅ **完了**
6. **低**: CPUバックエンドの検討
