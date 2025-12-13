# Harp プロジェクト概要

Harpは高性能なテンソル計算ライブラリです。計算グラフを構築し、様々なバックエンド（Metal等）で実行します。

## ワークスペース構成

```
harp/
├── Cargo.toml          # ワークスペースルート
├── src/                # harp (コアクレート)
├── crates/
│   ├── harp-derive/    # proc-macro (Module derive)
│   └── viz/            # 可視化ツール (harp-viz)
├── examples/           # 使用例
└── spec/               # 仕様書
```

## クレート依存関係

```
┌──────────┐      ┌───────────────┐
│ harp-viz │─────▶│     harp      │
└──────────┘      └───────────────┘
                          ▲
                          │
                  ┌───────────────┐
                  │  harp-derive  │
                  └───────────────┘
```

### 依存関係の詳細

| クレート | 依存先 | 説明 |
|---------|--------|------|
| `harp` | - | コアクレート（Graph, AST, Backend, Lowerer, Optimizer） |
| `harp-derive` | - | proc-macro（`#[derive(DeriveModule)]`） |
| `harp-viz` | harp | 計算グラフ可視化 |

## クレート詳細

### harp（コアクレート）

テンソル計算の基盤となるコアライブラリ。

**主要モジュール:**
- `graph`: 計算グラフ表現（GraphNode, Graph, DType）
- `ast`: 抽象構文木（AstNode, Function, Literal）
- `backend`: バックエンド抽象化（Device, Compiler, Kernel）
- `lowerer`: Graph→AST変換
- `opt`: 最適化パイプライン（graph最適化、ast最適化）

**仕様書:** [graph.md](graph.md), [ast.md](ast.md), [backend.md](backend.md), [lowerer.md](lowerer.md), [opt.md](opt.md)

### harp-derive

proc-macroクレート。

**提供マクロ:**
- `#[derive(DeriveModule)]`: Module traitの自動実装

### harp-viz

計算グラフの可視化ツール。

## 使用例

### 基本的な計算グラフ

```rust
use harp::prelude::*;

let mut graph = Graph::new();
let a = graph.input("a", DType::F32, vec![10, 20]);
let b = graph.input("b", DType::F32, vec![10, 20]);
let result = a + b;
graph.output("result", result);
```

### 畳み込み演算

```rust
use harp::prelude::*;

let mut graph = Graph::new();
let x = graph.input("x", DType::F32, vec![3, 32, 32]);
let kernel = graph.input("kernel", DType::F32, vec![16, 3, 3, 3]);

// 2D conv: (3, 32, 32) conv (16, 3, 3, 3) -> (16, 30, 30)
let output = x.conv(kernel, (1, 1), (1, 1), (0, 0));
```

