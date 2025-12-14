# 既知の問題

（現時点で未修正の問題はありません）

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
