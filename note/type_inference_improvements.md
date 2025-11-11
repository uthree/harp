# 型推論の改善

## 概要

Cレンダラーが`DType::Unknown`に対して`auto`を出力していた問題を解決するため、ASTノードに明示的な型情報を追加する作業を実施。

## 日付

2025-11-11

## 背景

### 問題の発見

- Cレンダラーが`DType::Unknown`を`"auto"`文字列に変換していた
- C言語では`auto`は型推論に使えない（C++11以降でのみ可能）
- 型推論機能（`infer_type()`）があるにも関わらず、実際には推論が失敗していた

### 根本原因

1. `Var`ノードが型情報を持たず、常に`DType::Unknown`を返す
2. `Load { ptr: Var("input0"), ... }`のように、`Var`をポインタとして使うと、ポインタの型も推論できない
3. その結果、`Load`の戻り値の型も`Unknown`になる

## 実施した対応（Phase 1: Load ノード）

### 1. `AstNode::Load`に`dtype`フィールドを追加

#### 変更前
```rust
Load {
    ptr: Box<AstNode>,    // ポインタ（Ptr<T>型の式）
    offset: Box<AstNode>, // オフセット（Usize型の式）
    count: usize,         // 読み込む要素数（コンパイル時定数、1ならスカラー）
}
```

#### 変更後
```rust
Load {
    ptr: Box<AstNode>,    // ポインタ（Ptr<T>型の式）
    offset: Box<AstNode>, // オフセット（Usize型の式）
    count: usize,         // 読み込む要素数（コンパイル時定数、1ならスカラー）
    dtype: DType,         // 読み込む要素の型
}
```

### 2. `infer_type()`の簡素化

#### 変更前
```rust
AstNode::Load { ptr, count, .. } => {
    let ptr_type = ptr.infer_type();
    let pointee_type = ptr_type.deref_type().clone();
    if *count == 1 {
        pointee_type // スカラー
    } else {
        pointee_type.to_vec(*count) // Vec型
    }
}
```

#### 変更後
```rust
AstNode::Load { dtype, .. } => dtype.clone(),
```

### 3. ヘルパー関数のシグネチャ変更

```rust
// 変更前
pub fn load(ptr: AstNode, offset: AstNode) -> AstNode
pub fn load_vec(ptr: AstNode, offset: AstNode, count: usize) -> AstNode

// 変更後
pub fn load(ptr: AstNode, offset: AstNode, dtype: DType) -> AstNode
pub fn load_vec(ptr: AstNode, offset: AstNode, count: usize, dtype: DType) -> AstNode
```

### 4. Lowererでの使用例

```rust
// Lowererでは型情報を持っているので、明示的に渡す
let input_ptr_dtype = self.graph_dtype_to_ast_ptr(&input.dtype)?;
let input_dtype = input_ptr_dtype.deref_type().clone();
statements.push(assign(&alu_var, load(input_ptr, input_offset, input_dtype)));
```

### 5. 修正したファイル

#### コア実装
- `src/ast/mod.rs` - Load定義、map_children、infer_type
- `src/ast/helper.rs` - load/load_vecヘルパー関数
- `src/ast/pat.rs` - パターンマッチング
- `src/backend/c_like.rs` - レンダリング、診断メッセージ追加

#### Lowerer（6ファイル）
- `src/lowerer/contiguous.rs`
- `src/lowerer/elementwise.rs`
- `src/lowerer/fused_elementwise.rs`
- `src/lowerer/fused_elementwise_reduce.rs`
- `src/lowerer/reduce.rs` (2箇所)

#### レンダラー
- `src/backend/openmp/renderer.rs` - `DType::Unknown`でパニックするように変更
- `src/backend/metal/renderer.rs` - `DType::Unknown`でパニックするように変更

#### テストコード（自動修正）
- `src/ast/tests/mod.rs` (4箇所)
- `src/ast/tests/node_tests.rs` (3箇所)
- `src/ast/tests/scope_tests.rs` (1箇所)
- `src/ast/tests/control_flow_tests.rs` (1箇所)
- `src/ast/helper.rs` (テスト: 4箇所)
- `src/backend/metal/renderer.rs` (テスト: 3箇所)

## 結果

### 成功
- ✅ コンパイルエラー: 0
- ✅ 警告: 0
- ✅ 成功したテスト: 305個
- ✅ `Load`ノードの型推論が正しく動作

### 新たに発見された問題

10個のテストが失敗（既存の問題が顕在化）:
- `lowerer::tests::contiguous_tests::test_lower_contiguous_2d`
- `lowerer::tests::elementwise_tests::test_lower_simple_add`
- `lowerer::tests::elementwise_tests::test_lower_with_flipped_view`
- `lowerer::tests::fusion_tests::test_lower_fused_elementwise`
- `lowerer::tests::fusion_tests::test_lower_fused_elementwise_reduce`
- `lowerer::tests::reduce_tests::test_lower_reduce_3d`
- `lowerer::tests::reduce_tests::test_lower_reduce_max`
- `lowerer::tests::reduce_tests::test_lower_reduce_mul`
- `lowerer::tests::reduce_tests::test_lower_reduce_sum_1d`
- `lowerer::tests::reduce_tests::test_lower_reduce_sum_2d`
- `lowerer::tests::reduce_tests::test_lower_reduce_sum_axis0`

#### エラー例
```
Type inference failed for variable 'alu2': inferred type is Unknown.
Value: Add(Var("alu0"), Var("alu1"))
```

## 残っている問題と解決策

### 問題: `Assign`ノードでの型推論失敗

#### 原因
1. `Var`ノードは型情報を持たない（常に`DType::Unknown`を返す）
2. `Add(Var("alu0"), Var("alu1"))`のような演算で、両オペランドが`Unknown`
3. 結果も`Unknown`になり、`Assign`で変数宣言時の型が決定できない

#### 現在のコード
```rust
// src/backend/c_like.rs:151
AstNode::Assign { var, value } => {
    let inferred_type = value.infer_type();
    if matches!(inferred_type, DType::Unknown) {
        panic!(
            "Type inference failed for variable '{}': inferred type is Unknown. Value: {:?}",
            var, value
        );
    }
    let type_str = self.render_dtype_backend(&inferred_type);
    format!(
        "{}{} {} = {};",
        self.indent(),
        type_str,
        var,
        self.render_expr(value)
    )
}
```

### 解決策の選択肢

#### Option 1: `Assign`に型情報を追加（推奨）

**長所:**
- `Load`と同じアプローチで一貫性がある
- Lowererは既に型情報を持っているので実装が簡単
- 最も直接的で影響範囲が小さい

**短所:**
- ノードサイズが大きくなる

**実装例:**
```rust
// AstNode定義
Assign {
    var: String,         // 変数名
    value: Box<AstNode>, // 代入する値
    dtype: DType,        // 変数の型（NEW）
}

// ヘルパー関数
pub fn assign(var: impl Into<String>, value: AstNode, dtype: DType) -> AstNode

// infer_type
AstNode::Assign { dtype, .. } => dtype.clone(),

// レンダリング
AstNode::Assign { var, value, dtype } => {
    let type_str = self.render_dtype_backend(dtype);
    format!("{}{} {} = {};", self.indent(), type_str, var, self.render_expr(value))
}
```

#### Option 2: 型付き`Var`を導入

**長所:**
- より汎用的な解決策
- 他の型推論問題も解決できる可能性

**短所:**
- 大規模な変更が必要
- すべての`Var`生成箇所を修正する必要がある
- 既存のコードへの影響が大きい

**実装例:**
```rust
// 2つのバリアントを用意
Var(String),              // 型なし変数（パターンマッチング用）
TypedVar(String, DType),  // 型付き変数（実際のコード生成用）
```

#### Option 3: 型アノテーションパスを追加

**長所:**
- 最も汎用的
- AST構造を変更しなくても良い

**短所:**
- 最も複雂
- 新しいパスの実装が必要
- パフォーマンスへの影響

**実装例:**
```rust
// レンダリング前に実行
fn annotate_types(node: &mut AstNode, scope: &Scope) {
    // Scopeを使って全ての変数の型を解決
    // Assignで宣言された変数をScopeに登録
    // Varを見つけたらScopeから型を取得
}
```

## 推奨アプローチ

**Option 1（`Assign`に`dtype`追加）を推奨**

理由:
1. 今回の`Load`修正と同じパターンで一貫性がある
2. 実装が最も簡単で影響範囲が小さい
3. Lowererは既に型情報を持っているので追加コストがない
4. 段階的な改善が可能（必要に応じて他のノードも同様に修正）

## 次のステップ

1. `Assign`ノードに`dtype`フィールドを追加
2. Lowererの`assign()`呼び出し箇所を修正
3. テストを実行して動作確認
4. 他に型推論が失敗しているノードがないか確認

## 実施した対応（Phase 2: VarDeclとScope-based変数管理）

### アプローチの選択

Phase 1での`Load`ノード対応後、`Assign`ノードの型推論失敗に対して、当初は`Assign`に`dtype`フィールドを追加する案（Option 1）を検討したが、より設計として綺麗な**Scope-based変数管理**（改良版のOption 3）を採用することに決定。

### 1. `VarDecl`に`initial_value`フィールドを追加

#### 変更内容
```rust
pub struct VarDecl {
    pub name: String,
    pub dtype: DType,
    pub mutability: Mutability,
    pub region: AccessRegion,
    pub kind: VarKind,
    pub initial_value: Option<AstNode>, // 新規追加: 初期値（パラメータ等はNone）
}
```

- 変数宣言時に初期値を一緒に保持できるようになった
- パラメータなど初期値がない変数は`None`を設定

### 2. `Scope::declare`のシグネチャを更新

#### 変更前
```rust
pub fn declare(
    &mut self,
    name: String,
    dtype: DType,
    mutability: Mutability,
    region: AccessRegion,
) -> Result<(), String>
```

#### 変更後
```rust
pub fn declare(
    &mut self,
    name: String,
    dtype: DType,
    mutability: Mutability,
    region: AccessRegion,
    initial_value: Option<AstNode>, // 新規追加
) -> Result<(), String>
```

### 3. `Scope`に`local_variables()`メソッドを追加

親スコープを除いた、このスコープで宣言されたローカル変数のみを取得するメソッドを追加：

```rust
pub fn local_variables(&self) -> impl Iterator<Item = &VarDecl> {
    self.variables.values()
}
```

### 4. `Block`のレンダリングを変更（C89スタイル）

#### 変更後の動作
```rust
fn render_block(&mut self, statements: &[AstNode], scope: &crate::ast::Scope) -> String {
    let mut result = String::new();

    // ブロック先頭で変数宣言を出力
    for var_decl in scope.local_variables() {
        if let Some(initial_value) = &var_decl.initial_value {
            let type_str = self.render_dtype_backend(&var_decl.dtype);
            result.push_str(&format!(
                "{}{} {} = {};\n",
                self.indent(),
                type_str,
                var_decl.name,
                self.render_expr(initial_value)
            ));
        }
    }

    // 文を描画
    for stmt in statements {
        result.push_str(&self.render_statement(stmt));
        result.push('\n');
    }
    result
}
```

- C89スタイル：ブロック先頭で全ての変数を一括宣言
- 初期値が設定されている変数のみを出力
- パラメータなど初期値がない変数は宣言されない（関数パラメータとして既に宣言済み）

### 5. `Assign`のレンダリングを簡素化

#### 変更前
```rust
AstNode::Assign { var, value } => {
    let inferred_type = value.infer_type();
    if matches!(inferred_type, DType::Unknown) {
        panic!("Type inference failed...");
    }
    let type_str = self.render_dtype_backend(&inferred_type);
    format!("{}{} {} = {};", self.indent(), type_str, var, self.render_expr(value))
}
```

#### 変更後
```rust
AstNode::Assign { var, value } => {
    // 単なる代入として扱う（変数宣言はBlockで行われる）
    format!("{}{} = {};", self.indent(), var, self.render_expr(value))
}
```

- 型推論のロジックを削除
- `Assign`は純粋な代入のみを表現
- 変数宣言は`Block`のレンダリング時に行われる

### 6. Lowererの変更

全てのLowererファイル（`contiguous.rs`, `elementwise.rs`, `reduce.rs`, `fused_elementwise.rs`, `fused_elementwise_reduce.rs`）で以下の変更を実施：

#### パターン: 変更前
```rust
let alu_var = self.fresh_alu();
statements.push(assign(&alu_var, load(input_ptr, input_offset, input_dtype)));
```

#### パターン: 変更後
```rust
let alu_var = self.fresh_alu();
scope.declare(
    alu_var.clone(),
    input_dtype.clone(),
    Mutability::Mutable,
    AccessRegion::ThreadLocal,
    Some(load(input_ptr, input_offset, input_dtype)), // 初期値
)?;
```

- `assign()`ヘルパー関数の使用を廃止
- `scope.declare()`で変数宣言と初期値設定を同時に行う
- 関数シグネチャに`scope: &mut Scope`パラメータを追加
- ループごとに適切にスコープを管理

### 7. 修正したファイル一覧

#### コア実装
- `src/ast/mod.rs` - VarDecl定義、Scope::declare、local_variables
- `src/backend/c_like.rs` - render_block、render_statement（Assign）

#### Lowerer（5ファイル）
- `src/lowerer/contiguous.rs`
- `src/lowerer/elementwise.rs`
- `src/lowerer/reduce.rs`
- `src/lowerer/fused_elementwise.rs`
- `src/lowerer/fused_elementwise_reduce.rs`

#### テストコード
- `src/ast/tests/call_return_tests.rs`
- `src/ast/tests/control_flow_tests.rs`
- `src/ast/tests/mod.rs`
- `src/ast/tests/scope_tests.rs`
- `src/ast/helper.rs`（テスト内のVarDecl初期化）
- `src/backend/metal/renderer.rs`（テスト内のVarDecl初期化）
- その他のテストファイル

### 修正の影響範囲
- VarDeclを使用する全ての箇所に`initial_value: None`を追加
- `scope.declare()`を呼び出す全ての箇所に5番目の引数を追加

## 結果

### 成功
- ✅ コンパイルエラー: 0
- ✅ 警告: 0（cargo clippy）
- ✅ 成功したテスト: 315個
- ✅ 型推論が正しく動作
- ✅ C89スタイルの変数宣言が出力される

### Phase 1との比較

| 項目 | Phase 1（Load対応） | Phase 2（VarDecl + Scope） |
|------|---------------------|---------------------------|
| アプローチ | ノードに型情報追加 | Scope-based管理 |
| 変更規模 | 小（Loadのみ） | 中（全Lowerer） |
| 設計の一貫性 | 部分的 | 高い |
| コード生成スタイル | - | C89スタイル |

## 設計上の利点

1. **型情報の一元管理**: 変数の型情報が`Scope`に集約され、管理が容易
2. **C89互換**: ブロック先頭での変数宣言により、古いCコンパイラとも互換性を保つ
3. **可読性の向上**: 生成されるCコードの可読性が向上（変数宣言がブロック先頭にまとまる）
4. **拡張性**: 将来的に変数の追加情報（アライメント、attributeなど）を`VarDecl`に追加しやすい

## 参考

- `Load`ノードの修正（Phase 1）から、Scope-based変数管理（Phase 2）への段階的な改善
- 今後も必要に応じて他のノードに型情報を追加可能
- パフォーマンスへの影響は最小限（型情報のサイズは小さく、クローンのコストも低い）
