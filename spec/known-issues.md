# 既知の問題

## 未修正の問題

（現在なし）

---

## 修正済みの問題

### 3つ以上の定数配列を組み合わせると誤った結果が返される

**発見日**: 2024-12
**修正日**: 2024-12

**状態**: 修正済み

**症状**:
lazy-arrayで3つ以上の独立した定数配列（`zeros()`, `ones()`, `full()`等で作成）を組み合わせると、誤った計算結果が返されていた。

**原因**:
- Kernelノードに一意なIDがなく、複数のKernelがマージされた際にデータフロー解析が誤って動作していた
- `establish_data_flow`で入力ノードの追跡が正しく行われていなかった

**修正内容**:
- Kernelノードに一意なIDを追加
- `establish_data_flow`のデータフロー追跡を修正

**関連テスト**:
- 関連テストは統合後のコードベースに移行済み

---

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
