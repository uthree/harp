# 既知の問題

## 未修正の問題

### 3つ以上の定数配列を組み合わせると誤った結果が返される

**発見日**: 2024-12

**状態**: 調査中

**症状**:
lazy-arrayで3つ以上の独立した定数配列（`zeros()`, `ones()`, `full()`等で作成）を組み合わせると、誤った計算結果が返される。

```rust
// 動作する: 2つの配列
let a = Array2::<f32>::zeros([4, 4]);
let b = Array2::<f32>::ones([4, 4]);
let c = &a + &b;  // 正しく 1.0 が返される

// 動作する: 配列 + スカラー
let a = Array2::<f32>::full([4, 4], 2.0);
let c = &a * 3.0;  // 正しく 6.0 が返される

// バグ: 3つの配列
let a = Array2::<f32>::full([4, 4], 2.0);
let b = Array2::<f32>::full([4, 4], 3.0);
let c = Array2::<f32>::ones([4, 4]);
let result = &(&a + &b) * &c;  // 期待: 5.0, 実際: 8.0
```

**原因（推測）**:
core側のグラフ最適化、Lowering、またはコード生成のどこかで、複数の独立した定数ソース（Contiguous操作を含む）を持つグラフが正しく処理されていない。`8 = 5 + 3 = (2+3) + 3`となっていることから、乗算の代わりに加算が実行されているか、変数の参照が誤っている可能性がある。

**ワークアラウンド**:
- 2つの配列の組み合わせまでは正しく動作する
- スカラー演算は正しく動作する

**関連テスト**:
- `crates/lazy-array/src/dyn_backend.rs`: `test_eval_three_arrays_add_then_mul`（`#[ignore]`で保留中）

---

## 修正済みの問題

### GraphRuntimeSelector計測時のバッファ追跡バグ

**発見日**: 2024-12
**修正日**: 2024-12

**状態**: 修正済み

**症状**:
Phase 1でGraphRuntimeSelectorを使用した実測値ベース最適化を行う際、計測用のLoweringで2つのエラーが発生していた：
1. `use of undeclared identifier 'input1'` - 入力バッファが正しく追跡されない
2. `use of undeclared identifier 'shape2'` - 形状変数が正しく置換されない

**原因**:
`collect_input_buffers`関数でView操作を経由してInputノードを追跡する際に：
1. 入力バッファの追跡が行われていなかった
2. 追跡時に元のBufferのviewを使用していたため、View操作による形状変換が失われていた

**修正内容**:
`src/opt/graph/suggesters/lowering/helpers.rs`に以下の修正を実施：

1. `collect_input_buffers`関数を追加し、Viewノードを透過してBufferノードを追跡
2. トレース開始点（Kernelの直接のsrc）のviewを"entry view"として保持し、Bufferノード作成時にそのviewを使用

```rust
// 修正のポイント
for src in src_nodes {
    // 各srcノードのviewを"entry view"として保持
    // これがKernelが期待する入力形状
    let entry_view = src.view.clone();
    collect_inputs_recursive(src, &mut input_names, &mut seen, &entry_view);
}
```

これにより、Kernelが期待する形状（View変換後の形状）が正しくBufferノードに伝達され、
ProgramRootAbsorptionでのshapeプレースホルダー置換が正しく動作するようになった。

### ループタイリング時の変数未宣言バグ

**発見日**: 2024-12
**修正日**: 2024-12

**状態**: 修正済み

**症状**:
AST最適化でループタイリング(`LoopTilingSuggester`)が適用された際、タイル化されたループ変数（例: `ridx2`）が宣言される前に使用されるコードが生成される。

**原因**:
`src/opt/ast/transforms.rs`の`tile_loop`関数で、タイル化後の内側ループ本体のBlockを作成する際に、元のループ変数をスコープに宣言していなかった。

```rust
// 修正前
let inner_body = AstNode::Block {
    statements: inner_body_statements,
    scope: Box::new(Scope::new()),  // 空のスコープ
};

// 修正後
let mut inner_scope = Scope::new();
inner_scope
    .declare(var.clone(), DType::Int, Mutability::Mutable)
    .expect("Failed to declare loop variable in inner scope");

let inner_body = AstNode::Block {
    statements: inner_body_statements,
    scope: Box::new(inner_scope),  // 変数が宣言されたスコープ
};
```

**修正内容**:
1. 内側ループ本体のスコープに元のループ変数を宣言
2. 端数処理ループ本体のスコープにも同様に宣言を追加
3. 修正を検証するテストケース`test_tile_loop_declares_original_variable`を追加
