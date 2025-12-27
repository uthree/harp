# Autograd実装の問題点・改善タスク

## 概要
2層パーセプトロンのデモ作成時に発覚した、autogradシステムの問題点と改善タスク。

---

## 1. View操作のバックワード未実装

### 問題
`unsqueeze`, `expand`, `reshape` などのview操作で新しい `TensorInner` を作成する際に `autograd: None` が設定され、勾配追跡が途切れる。

### 該当箇所
- `src/tensor/primops/movement.rs` の `unsqueeze` (225行目付近)
- `src/tensor/primops/movement.rs` の他のview操作

### 必要な修正
各view操作に対応するバックワード関数を実装：
- `UnsqueezeBackward` → squeeze で勾配を戻す
- `ExpandBackward` → ブロードキャスト次元に沿ってsumで勾配を集約
- `ReshapeBackward` → 元の形状にreshapeして勾配を戻す
- `SqueezeBackward` → unsqueezeで勾配を戻す
- `TransposeBackward` → 逆転置で勾配を戻す

### 影響範囲
`matmul`, `matmul2`, `dot`, `outer` などの高レベル演算がview操作を内部で使用しているため、これらすべての勾配計算が機能しない。

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

## 4. outer1/dot1 の autograd 情報破棄

### 問題
`matmul2` と同様に、`outer1` と `dot1` も結果を変換する際にautograd情報を破棄している可能性。

### 該当箇所
- `src/tensor/hlops/linalg.rs` の `outer1` (53-66行目)
- `src/tensor/hlops/linalg.rs` の `dot1` は DimDyn を返すため問題なさそう

### 必要な確認
`outer1` の実装を確認し、必要であれば修正。

---

## 5. Tensor::rand の OpenCL 非対応

### 問題
`Tensor::rand()` が生成するカーネルコードで C言語の `rand()` と `RAND_MAX` を使用しているが、OpenCL にはこれらの関数がない。

### 該当エラー
```
error: implicit declaration of function 'rand' is invalid in OpenCL
error: use of undeclared identifier 'RAND_MAX'
```

### 必要な修正
- OpenCL向けに適切な乱数生成方法を実装
- または、rand操作はCPU側で事前に実行してからGPUに転送する設計に変更

---

## 6. VecBuffer から OpenCL への転送問題

### 問題
`from_ndarray` で作成したテンソル（VecBuffer使用）をOpenCLで実行する際に、カーネル引数数のミスマッチが発生。

### 該当エラー
```
The wrong number of kernel arguments have been specified (required: 3, specified: 2)
```

### 該当箇所
- `src/tensor/forward.rs` のOpenCL実行ロジック
- `src/backend/opencl/` のカーネル実行処理

### 必要な調査
VecBufferから OpenCLBuffer への変換・転送ロジックの確認。

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

## 優先度

1. **高**: View操作のバックワード実装 (autograd全体に影響)
2. **高**: VecBuffer → OpenCL転送の修正 (基本的な実行に影響)
3. **中**: Tensor::rand のOpenCL対応
4. **低**: outer1 の autograd 修正
5. **低**: CPUバックエンドの検討
