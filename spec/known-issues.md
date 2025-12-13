# 既知の問題

(現時点で既知の問題はありません)

## 修正済みの問題

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
