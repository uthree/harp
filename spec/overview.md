# Harp プロジェクト概要

Harpは高性能なテンソル計算ライブラリです。計算グラフを構築し、様々なバックエンド（Metal等）で実行します。

## ワークスペース構成

```
harp/
├── Cargo.toml          # ワークスペースルート
├── src/                # harp (コアクレート)
├── crates/
│   ├── harp-autograd/  # 自動微分機能
│   ├── harp-nn/        # ニューラルネットワークモジュール
│   ├── harp-derive/    # proc-macro (Module derive)
│   ├── viz/            # 可視化ツール (harp-viz)
│   └── unified_ir/     # 統一中間表現
├── examples/           # 使用例
└── spec/               # 仕様書
```

## クレート依存関係

```
                    ┌─────────────┐
                    │   harp-nn   │
                    └──────┬──────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  harp-autograd │
                  └───────┬────────┘
                          │
                          ▼
┌──────────┐      ┌───────────────┐      ┌──────────────┐
│ harp-viz │─────▶│     harp      │◀─────│  unified_ir  │
└──────────┘      └───────────────┘      └──────────────┘
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
| `harp-autograd` | harp | 自動微分（Tensor, GradFn, backward） |
| `harp-nn` | harp-autograd, harp-derive | ニューラルネットワーク（Module, Parameter, Optimizer） |
| `harp-derive` | - | proc-macro（`#[derive(DeriveModule)]`） |
| `harp-viz` | harp | 計算グラフ可視化 |
| `unified_ir` | harp | 統一中間表現 |

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

### harp-autograd

PyTorchライクな自動微分機能を提供。

**主要型:**
- `Tensor`: 勾配追跡機能付きテンソル
- `GradFn`: 勾配計算関数trait

**仕様書:** [autograd.md](autograd.md)

### harp-nn

PyTorchの`torch.nn`に相当するニューラルネットワーク機能。

**主要型:**
- `Module`: ニューラルネットワークモジュールtrait
- `Parameter`: 学習可能パラメータ（`Tensor`のnewtype）
- `Optimizer`: 最適化アルゴリズムtrait
- `SGD`: 確率的勾配降下法

**仕様書:** [nn.md](nn.md)

### harp-derive

`harp-nn`用のproc-macro。

**提供マクロ:**
- `#[derive(DeriveModule)]`: Module traitの自動実装

### harp-viz

計算グラフの可視化ツール。

### unified_ir

異なるバックエンド間で共有可能な統一中間表現。

**仕様書:** [unified-ir.md](unified-ir.md)

## 使用例

### 基本的な計算グラフ

```rust
use harp::prelude::*;

let mut graph = Graph::new();
let a = graph.input("a").with_dtype(DType::F32).with_shape([10, 20]).build();
let b = graph.input("b").with_dtype(DType::F32).with_shape([10, 20]).build();
let result = a + b;
graph.output("result", result);
```

### 自動微分

```rust
use harp::prelude::*;
use harp_autograd::Tensor;

let x = Tensor::ones(vec![10, 20]);
let y = &x * 2.0 + 1.0;
let loss = y.sum(0).sum(0);
loss.backward();

let grad = x.grad().unwrap();
```

### ニューラルネットワーク

```rust
use harp_nn::{Module, Parameter, impl_module};
use harp_autograd::Tensor;

struct Linear {
    weight: Parameter,
    bias: Parameter,
}

impl_module! {
    for Linear {
        parameters: [weight, bias]
    }
}
```

## 変更履歴

### 2025-11-26: autograd サブクレート化

`autograd`モジュールを`harp`本体から分離し、独立したサブクレート`harp-autograd`として再構成。

**変更点:**
- `harp::autograd::Tensor` → `harp_autograd::Tensor`
- `harp-nn`は`harp-autograd`に依存するよう変更
- `Graph::realize()`メソッドを`harp`本体に移動

**目的:**
- モジュール独立性の向上
- コンパイル時間の最適化
- 依存関係の明確化

### 2025-11-25: nn サブクレート化

`nn`モジュールを`harp`本体から分離し、独立したサブクレート`harp-nn`として再構成。
