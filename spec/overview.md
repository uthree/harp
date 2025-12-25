# Harp プロジェクト概要

Harpは高性能なテンソル計算ライブラリです。計算グラフを構築し、様々なバックエンド（OpenCL, Metal等）で実行します。

## ディレクトリ構成

```
harp/
├── Cargo.toml            # メインクレート
├── src/
│   ├── lib.rs            # エントリポイント
│   ├── tensor/           # 統合Tensor型（autograd + lazy evaluation）
│   │   ├── mod.rs        # Tensor<D> 定義
│   │   ├── dimension.rs  # Dimension trait, Dim<N>, DimDyn
│   │   ├── ops.rs        # 演算子オーバーロード
│   │   ├── forward.rs    # forward() 実装
│   │   └── grad.rs       # GradFn, backward() 実装
│   ├── graph/            # 計算グラフ表現
│   ├── ast/              # 抽象構文木
│   ├── renderer/         # コードレンダラー（常に利用可能）
│   │   ├── mod.rs
│   │   ├── traits.rs     # Renderer trait
│   │   ├── c_like.rs     # CLikeRenderer
│   │   ├── opencl.rs     # OpenCLRenderer
│   │   └── metal.rs      # MetalRenderer
│   ├── backend/          # 実行バックエンド（feature gated）
│   │   ├── mod.rs
│   │   ├── traits.rs     # Device, Buffer, Kernel, Compiler
│   │   ├── pipeline.rs   # Pipeline, CompiledKernel
│   │   ├── global.rs     # グローバルデバイス管理
│   │   ├── opencl/       # OpenCLバックエンド (feature: opencl)
│   │   └── metal/        # Metalバックエンド (feature: metal)
│   ├── lowerer/          # Graph→AST変換
│   └── opt/              # 最適化パイプライン
├── tests/                # 統合テスト
└── spec/                 # 仕様書
```

## Feature Flags

| Feature | 説明 |
|---------|------|
| `opencl` | OpenCLバックエンドを有効化 |
| `metal` | Metalバックエンドを有効化（macOSのみ） |

```toml
# Cargo.toml での使用例
[dependencies]
harp = { version = "0.1", features = ["opencl"] }
# macOSの場合
harp = { version = "0.1", features = ["metal"] }
```

## 主要モジュール

### tensor（統合Tensor型）

PyTorchライクなTensor APIを提供。遅延評価と自動微分をサポート。

**主要機能:**
- 静的次元管理（`Dim<N>`: `Dim0`-`Dim6`）と動的次元（`DimDyn`）
- 遅延評価による計算グラフ構築
- `forward()`: デフォルトデバイスでの計算実行
- `backward()`: 自動微分（勾配計算）
- `requires_grad()`: 勾配追跡の有効化

**仕様書:** [tensor.md](tensor.md)（新規作成予定）

### graph

計算グラフ表現（GraphNode, Graph, DType）

**仕様書:** [graph.md](graph.md)

### ast

抽象構文木（AstNode, Function, Literal）

**仕様書:** [ast.md](ast.md)

### renderer

コードレンダラー。バックエンド無しでもソースコード生成が可能。

- `GenericRenderer`: C言語風コード生成
- `OpenCLRenderer`: OpenCL Cコード生成
- `MetalRenderer`: Metal Shading Language生成

### backend

GPU実行バックエンド。Renderer traitとDevice/Buffer/Kernel traits。

**仕様書:** [backend.md](backend.md)

### lowerer

Graph→AST変換

**仕様書:** [lowerer.md](lowerer.md)

### opt

最適化パイプライン（graph最適化、ast最適化）

**仕様書:** [opt.md](opt.md), [opt-graph.md](opt-graph.md), [opt-ast.md](opt-ast.md)

## 使用例

### 基本的な計算（遅延評価）

```rust
use harp::tensor::{Tensor, Dim2};

// テンソル作成（まだ計算は実行されない）
let x = Tensor::<Dim2>::full([3, 4], 2.0);
let y = Tensor::<Dim2>::full([3, 4], 3.0);

// 演算（計算グラフを構築）
let z = &x + &y;  // z = x + y
let w = &z * &z;  // w = z^2

// forward()で計算実行
w.forward().unwrap();
let result = w.data().unwrap();
```

### 自動微分（backward）

```rust
use harp::tensor::{Tensor, Dim2};

// 勾配追跡を有効化
let x = Tensor::<Dim2>::full([2, 2], 2.0).set_requires_grad(true);

// y = x * x = x^2
let y = &x * &x;

// 逆伝播
y.backward();

// 勾配取得: dy/dx = 2x = 4.0
let grad = x.grad().unwrap();
```

### グローバルデバイス管理

```rust
use harp::backend::{set_default_device, DeviceKind};

#[cfg(feature = "metal")]
{
    use harp::backend::metal::MetalDevice;
    let device = MetalDevice::new().unwrap();
    set_default_device(device, DeviceKind::Metal);
}

// デバイス設定後はforward()が使用可能
let x = Tensor::<Dim2>::ones([3, 4]);
let y = &x + &x;
y.forward().unwrap();
```

### 低レベルAPI（計算グラフ直接操作）

```rust
use harp::prelude::*;

let mut graph = Graph::new();
let a = graph.input("a", DType::F32, vec![10, 20]);
let b = graph.input("b", DType::F32, vec![10, 20]);
let result = a + b;
graph.output("result", result);
```
