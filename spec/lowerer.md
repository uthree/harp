# Lowerer モジュール仕様書

計算グラフ（GraphNode）をAST（AstNode）に変換する`src/lowerer/`モジュールの仕様。

## 概要

Lowererモジュールは`GraphNode`計算グラフを実行可能な`AstNode`に変換します。
変換過程で以下の処理を行います：

1. **融合（Fusion）**: 連続する互換演算を1つにまとめる
2. **インデックス生成**: ViewからLoad/Store用のインデックス式を生成
3. **ループ生成**: テンソル形状からネストループ構造を生成
4. **Lowering**: 各ノードをカーネルASTに変換

## ディレクトリ構成

```
src/lowerer/
├── mod.rs          # モジュール定義、re-exports
├── fusion.rs       # ノード融合パス
├── index_gen.rs    # インデックス式生成
├── loop_gen.rs     # ループ構造生成
├── lower.rs        # メインLowering処理
└── tests.rs        # 統合テスト
```

## 戦略: 1 Node = 1 Kernel

融合パス適用後、残りの各GraphNodeは1つのKernelに変換されます。
これにより：

- バッファ管理がシンプルになる
- 実行境界が明確になる
- デバッグが容易になる

## 主要な型

### Lowerer

```rust
pub struct Lowerer {
    buffer_counter: usize,
    buffer_map: HashMap<*const GraphInner, String>,
    dtype_map: HashMap<*const GraphInner, DType>,
    kernels: Vec<AstNode>,
    loop_gen: LoopGenerator,
}

impl Lowerer {
    pub fn new() -> Self;
    pub fn lower(&mut self, roots: &[GraphNode]) -> AstNode;
}
```

生成されるKernelは入力バッファと出力バッファをポインタ引数として持ちます。

### FusionPass

```rust
pub trait FusionPass {
    fn apply(&self, roots: &[GraphNode]) -> Vec<GraphNode>;
}
```

## 融合ルール

### View融合

連続するView演算を1つにまとめます。

```
Before: GraphNode → View(v1) → View(v2)
After:  GraphNode → View(compose(v2, v1))
```

### Elementwise + Reduce融合

Identity mapの縮約と先行するElementwise演算を1つにまとめます。

```
Before: x → MapReduce{map=f, reduce=None} → MapReduce{map=id, reduce=Sum}
After:  x → MapReduce{map=f, reduce=Sum}
```

## インデックス生成

`IndexGenerator`がViewからインデックス式を生成します。

### Linearビュー

```rust
// View::Linear { shape, strides, offset }
// → offset + Σ(ridx[i] * strides[i])
```

### IndexExprビュー

```rust
// View::IndexExpr { shape, index_expr }
// → index_expr をそのまま使用
```

### Maskedビュー

```rust
// View::Masked { inner, condition, default_value }
// → Select { cond: condition, then: inner_idx, else: default }
```

## ループ生成

`LoopGenerator`がテンソル形状からネストループを生成します。

### Elementwise演算

```text
for ridx0 in 0..shape[0]:
    for ridx1 in 0..shape[1]:
        output[idx] = f(input[idx])
```

### Reduce演算

```text
for ridx0 in 0..shape[0]:
    acc = identity
    for ridx1 in 0..shape[1]:
        acc = combine(acc, input[idx])
    output[ridx0] = acc
```

## 生成されるAST

### Elementwise: `y = x * 2.0`

```rust
Kernel {
    name: "kernel_y",
    params: [
        VarDecl { name: "x", dtype: Ptr(F32), mutability: Immutable },
        VarDecl { name: "y", dtype: Ptr(F32), mutability: Mutable },
    ],
    body: Range { var: "ridx0", stop: shape[0],
        body: Range { var: "ridx1", stop: shape[1],
            body: Store(output, idx, Load(input, idx) * 2.0)
        }
    }
}
```

### Reduce: `y = sum(x, axis=1)`

```rust
Kernel {
    name: "kernel_y",
    params: [
        VarDecl { name: "x", dtype: Ptr(F32), mutability: Immutable },
        VarDecl { name: "y", dtype: Ptr(F32), mutability: Mutable },
    ],
    body: Range { var: "ridx0", stop: shape[0],
        body: Block [
            Assign(acc, 0.0),
            Range { var: "ridx1", stop: shape[1],
                body: Assign(acc, acc + Load(input, idx))
            },
            Store(output, ridx0, acc)
        ]
    }
}
```

## Wildcard置換

`MapReduce.map`内の`Wildcard`は`Load`演算に置き換えられます：

```
Wildcard("0") → Load { ptr: src[0], offset: index_expr, count: 1, dtype }
Wildcard("1") → Load { ptr: src[1], offset: index_expr, count: 1, dtype }
...
```

## 使用例

```rust
use harp::graph::{input, Expr, DType};
use harp::lowerer::Lowerer;

// 計算グラフ構築
let a = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("a");
let b = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("b");
let c = (&a + &b).sum(1).with_name("c");

// ASTに変換
let mut lowerer = Lowerer::new();
let program = lowerer.lower(&[c]);

// program は AstNode::Program { functions: [...], execution_waves: [...] }
```

## 制限事項

現在のLowererは基本的な演算のみをサポートしています。
サポートされるAstNode演算子：

- 算術: `Add`, `Mul`, `Max`, `Recip`, `Sqrt`, `Log2`, `Exp2`, `Sin`, `Floor`
- 比較: `Lt`
- 論理: `And`, `Not`
- 型変換: `Cast`
- 条件: `Select`

サポートされないAST演算子（将来対応予定）：
- `Sub`, `Div`, `Neg` (プリミティブとして存在しない)
- `Cos`, `Ln`, `Exp`, `Abs` (プリミティブとして存在しない)
- `Gt`, `Le`, `Ge`, `Eq`, `Ne`, `Or` (Ltから派生)

## 関連仕様書

- [ast.md](ast.md) - AstNode定義
- [graph.md](graph.md) - GraphNode定義
- [backend.md](backend.md) - Pipelineでの使用
