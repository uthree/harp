# AST統一化設計: Program/FunctionをAstNodeに統合

## 背景と動機

### 現在の問題点

現在、ASTは3つの別々の型で構成されている：

```rust
pub enum AstNode { /* 式と文 */ }
pub struct Function { params, return_type, body, kind }
pub struct Program { functions: HashMap<String, Function>, entry_point }
```

**問題:**
1. **最適化の制限**: AstOptimizerは`AstNode`しか扱えず、関数を跨いだ最適化（インライン展開、関数融合）ができない
2. **コードの重複**: Function/Programに対する操作（検証、変換、レンダリング）が別途必要
3. **複雑なLowerer**: 異なる型を行き来する必要があり、コードが煩雑

### 提案: 統一化

`Function`と`Program`を`AstNode`のバリアントとして統合：

```rust
pub enum AstNode {
    // 既存の式と文...

    // 新規追加
    Function {
        name: Option<String>,         // 関数名（匿名関数はNone）
        params: Vec<VarDecl>,
        return_type: DType,
        body: Box<AstNode>,           // Block等
        kind: FunctionKind,
    },

    Program {
        functions: Vec<AstNode>,      // AstNode::Function のリスト
        entry_point: String,
    },
}
```

## メリット

### 1. 関数を跨いだ最適化が可能に

**インライン展開:**
```rust
// Before
Program {
    functions: [
        "add" -> Function { body: a + b },
        "main" -> Function { body: Call("add", [x, y]) }
    ]
}

// After optimization
Program {
    functions: [
        "main" -> Function { body: x + y }  // add関数がインライン化
    ]
}
```

**関数融合:**
```rust
// Before
f(x) = x * 2
g(x) = x + 1
h(x) = g(f(x))  // 2つの関数呼び出し

// After optimization
h(x) = (x * 2) + 1  // 1つの式に融合
```

### 2. AstOptimizerの強化

現在の`AstRewriteRule`がそのまま`Function`と`Program`にも適用可能に：

```rust
// 関数のインライン化ルール
astpat!(|name, args, body| {
    AstNode::Call(name, args)  // where function is pure & small
} => {
    // 引数を代入してbodyを展開
    inline_function(body, args)
})
```

### 3. Lowererの簡略化

現在は`Program`と`Function`を別々に構築しているが、すべて`AstNode`の操作に統一：

```rust
// Before
let function = Function::new(kind, params, return_type, statements)?;
program.add_function(name, function)?;

// After
let function = AstNode::Function { name, params, return_type, body, kind };
let program = AstNode::Program { functions: vec![function], entry_point };
```

### 4. パターンマッチングの統一

`ast/pat.rs`の`AstPattern`がそのまま関数にも適用可能：

```rust
// 未使用関数の削除
astpat!(|functions, entry| {
    AstNode::Program { functions, entry_point }
} => {
    AstNode::Program {
        functions: remove_unused_functions(functions, entry_point),
        entry_point
    }
})
```

## 設計の詳細

### 新しいAstNode定義

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum AstNode {
    // ========================================
    // 既存のバリアント（変更なし）
    // ========================================
    Wildcard(String),
    Const(Literal),
    Add(Box<AstNode>, Box<AstNode>),
    Mul(Box<AstNode>, Box<AstNode>),
    // ... 他の演算子

    Var(String),
    Load { ptr: Box<AstNode>, offset: Box<AstNode>, count: usize },
    Store { ptr: Box<AstNode>, offset: Box<AstNode>, value: Box<AstNode> },
    Assign { var: String, value: Box<AstNode> },

    Block { statements: Vec<AstNode>, scope: Box<Scope> },
    Range { var: String, start: Box<AstNode>, step: Box<AstNode>, stop: Box<AstNode>, body: Box<AstNode> },

    Call { name: String, args: Vec<AstNode> },
    Return { value: Box<AstNode> },
    Barrier,

    // ========================================
    // 新規追加
    // ========================================

    /// 関数定義
    Function {
        name: Option<String>,         // 関数名（Program内では不要、匿名関数も可能）
        params: Vec<VarDecl>,         // 引数リスト
        return_type: DType,           // 返り値の型
        body: Box<AstNode>,           // 関数本体（通常はBlock）
        kind: FunctionKind,           // Normal or Kernel
    },

    /// プログラム全体
    Program {
        functions: Vec<AstNode>,      // AstNode::Function のリスト
        entry_point: String,          // エントリーポイントの関数名
    },
}
```

### ヘルパー関数

後方互換性とユーザビリティのため、ヘルパー関数を提供：

```rust
// ast/helper.rs に追加
pub fn function(
    name: impl Into<String>,
    kind: FunctionKind,
    params: Vec<VarDecl>,
    return_type: DType,
    body: AstNode,
) -> AstNode {
    AstNode::Function {
        name: Some(name.into()),
        params,
        return_type,
        body: Box::new(body),
        kind,
    }
}

pub fn program(functions: Vec<AstNode>, entry_point: impl Into<String>) -> AstNode {
    AstNode::Program {
        functions,
        entry_point: entry_point.into(),
    }
}

// 関数名でアクセス
impl AstNode {
    pub fn get_function(&self, name: &str) -> Option<&AstNode> {
        match self {
            AstNode::Program { functions, .. } => {
                functions.iter().find(|f| {
                    matches!(f, AstNode::Function { name: Some(n), .. } if n == name)
                })
            }
            _ => None,
        }
    }

    pub fn get_entry(&self) -> Option<&AstNode> {
        match self {
            AstNode::Program { entry_point, .. } => self.get_function(entry_point),
            _ => None,
        }
    }
}
```

### 既存APIの型シグネチャ変更

```rust
// lowerer/mod.rs
pub(crate) fn lower(graph: Graph) -> AstNode {  // was: Program
    // ...
    AstNode::Program { functions, entry_point: "main".to_string() }
}

// backend/mod.rs
pub trait Renderer {
    fn render_program(&self, program: &AstNode) -> String;  // was: &Program
    fn render_function(&self, func: &AstNode) -> String;    // was: &Function
}
```

## 実装計画

### Phase 1: 基盤変更（1-2日）

1. **型定義の変更**
   - [ ] `AstNode`に`Function`と`Program`バリアントを追加
   - [ ] `FunctionKind`は維持（Normal/Kernel）
   - [ ] 既存の`Function`/`Program`構造体は一旦残す（段階的移行）

2. **ヘルパー関数の追加**
   - [ ] `ast/helper.rs`に`function()`、`program()`を追加
   - [ ] `AstNode`に`get_function()`、`get_entry()`メソッドを追加

3. **基本機能の実装**
   - [ ] `infer_type()`の拡張（Function/Programの型推論）
   - [ ] `check_scope()`の拡張（関数呼び出しの検証）
   - [ ] `children()`/`map_children()`の拡張

### Phase 2: Lowerer/Rendererの更新（2-3日）

4. **Lowererの更新**
   - [ ] `lowerer/mod.rs`の`lower()`を`AstNode`返却に変更
   - [ ] 各lowerer（elementwise, reduce等）をAstNode::Functionを返すように変更
   - [ ] テストの更新

5. **Rendererの更新**
   - [ ] `backend/metal/renderer.rs`の更新
   - [ ] `backend/openmp/renderer.rs`の更新
   - [ ] `AstNode::Function`/`AstNode::Program`のレンダリング対応

6. **パターンマッチングの拡張**
   - [ ] `ast/pat.rs`にFunction/Programのパターン対応を追加
   - [ ] `astpat!`マクロの拡張

### Phase 3: 最適化機能の追加（3-5日）

7. **関数インライン化**
   - [ ] 小さな関数を自動的にインライン展開するルール
   - [ ] `opt/ast/rules.rs`に追加

8. **未使用関数の削除**
   - [ ] エントリーポイントから到達不可能な関数を削除
   - [ ] デッドコード除去

9. **関数融合**
   - [ ] 連続した関数呼び出しを1つに融合
   - [ ] パイプライン最適化

### Phase 4: 移行完了とクリーンアップ（1日）

10. **旧型の削除**
    - [ ] 元の`Function`/`Program`構造体を削除
    - [ ] すべてのコードが`AstNode`版を使用していることを確認

11. **テストの追加**
    - [ ] 関数インライン化のテスト
    - [ ] 関数融合のテスト
    - [ ] 未使用関数削除のテスト

12. **ドキュメント更新**
    - [ ] `spec/ast.md`の更新
    - [ ] コード内のドキュメントコメント更新

## 影響範囲

### 変更が必要なファイル（推定）

**Core AST:**
- `src/ast/mod.rs` (型定義) - 大幅変更
- `src/ast/helper.rs` (ヘルパー関数) - 追加
- `src/ast/pat.rs` (パターンマッチング) - 拡張
- `src/ast/renderer.rs` - 拡張

**Lowerer:**
- `src/lowerer/mod.rs` - 大幅変更
- `src/lowerer/elementwise.rs` - 中程度
- `src/lowerer/reduce.rs` - 中程度
- `src/lowerer/contiguous.rs` - 中程度
- `src/lowerer/fused_*.rs` - 中程度
- `src/lowerer/recursive.rs` - 中程度

**Backend:**
- `src/backend/generic.rs` - 中程度
- `src/backend/metal/renderer.rs` - 中程度
- `src/backend/openmp/renderer.rs` - 中程度

**Optimizer:**
- `src/opt/ast/rules.rs` - 拡張（新ルール追加）
- `src/opt/ast/optimizer.rs` - 軽微な変更

**Tests:**
- `src/ast/tests/function_program_tests.rs` - 大幅変更
- `src/lowerer/tests/*.rs` - 中程度
- `tests/*.rs` - 軽微な変更

**推定総行数:** 約4,000行の変更/追加

## リスクと対策

### リスク1: 大規模な変更による不具合

**対策:**
- 段階的に進める（Phase 1-4に分割）
- 各Phaseでテストを実行し、動作を確認
- 既存のテストが全てパスすることを確認

### リスク2: 型システムの複雑化

**対策:**
- パターンマッチングを多用し、型安全性を確保
- ヘルパー関数で使いやすいAPIを提供
- 適切なドキュメント・コメント

### リスク3: パフォーマンスの劣化

**対策:**
- `Box`の使用を最小限に（既存と同等）
- ベンチマークで確認
- 必要に応じて`Rc`等を検討

## 互換性

### 後方互換性

**破壊的変更:**
- `lowerer::lower()`の戻り値が`Program` → `AstNode`に変更
- `Renderer`のメソッドシグネチャが変更

**移行パス:**
- Phase 1で旧型を残し、段階的に移行
- Phase 4で完全削除

### 将来への拡張性

統一化により、以下の機能が追加しやすくなる：

1. **ラムダ関数/クロージャ**
   ```rust
   AstNode::Function {
       name: None,  // 匿名
       params: vec![VarDecl { name: "x", ... }],
       body: ...
   }
   ```

2. **関数ポインタ/高階関数**
   ```rust
   AstNode::Call {
       name: "map",
       args: vec![array, function_node]
   }
   ```

3. **ジェネリクス/テンプレート**
   ```rust
   AstNode::Function {
       type_params: vec!["T"],
       ...
   }
   ```

## 参考: 他のコンパイラの実装

### LLVM IR
- すべてが`Value`という統一型
- `Function`も`Value`の一種

### Cranelift IR
- 関数は独立した型だが、IR全体が統一的に扱える

### TinyGrad
- LazyBufferで全てを統一的に扱う

## 結論

この変更は大規模だが、以下の理由から実行すべき：

1. **明確なメリット**: 関数を跨いだ最適化が可能に
2. **長期的な保守性向上**: コードの統一性が向上
3. **実装可能**: 段階的に進めれば安全に実施可能

**推奨実装スケジュール:** 1-2週間（フルタイムの場合）

## 次のステップ

1. ✅ この設計書をレビュー
2. [ ] Phase 1の実装開始
3. [ ] 各Phaseごとにレビュー・テスト
4. [ ] 完了後、仕様書を更新
