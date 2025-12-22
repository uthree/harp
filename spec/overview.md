# Harp プロジェクト概要

Harpは高性能なテンソル計算ライブラリです。計算グラフを構築し、様々なバックエンド（OpenCL, Metal等）で実行します。

## ワークスペース構成

```
harp/
├── Cargo.toml            # ワークスペースルート + re-exportクレート
├── src/lib.rs            # harp (harp-coreのre-export + backend modules)
├── tests/                # 統合テスト（バックエンド機能を使用）
├── crates/
│   ├── core/             # harp-core（コア機能）
│   ├── lazy-array/       # harp-lazy-array（ndarray風配列API）
│   ├── backend-opencl/   # harp-backend-opencl
│   ├── backend-metal/    # harp-backend-metal (macOS only)
│   ├── dsl/              # harp-dsl
│   ├── cli/              # harp-cli (harpc コマンド)
│   ├── viz/              # harp-viz
│   └── nn/               # harp-nn
└── spec/                 # 仕様書
```

## クレート依存関係

```
     ┌───────────┐      ┌───────────┐
     │ harp-viz  │──┬──▶│   harp    │
     ├───────────┤  │   └─────┬─────┘
     │ harp-cli  │──┤         │ re-exports
     └───────────┘  │   ┌─────┴─────────────────────┐
                    │   ▼                           ▼
               ┌────────────┐  ┌───────────────────────┐
               │ harp-core  │◀─┤ harp-backend-{opencl,metal} │
               └────────────┘  └───────────────────────┘
                    ▲
          ┌─────────┴─────────┐
          ▼                   ▼
   ┌───────────┐       ┌───────────┐
   │ harp-dsl  │       │  harp-nn  │
   └───────────┘       └───────────┘
```

### 依存関係の詳細

| クレート | 依存先 | 説明 |
|---------|--------|------|
| `harp` | harp-core, harp-backend-* (optional) | re-exportクレート、後方互換性を維持 |
| `harp-core` | - | コアクレート（Graph, AST, Backend共通, Lowerer, Optimizer） |
| `harp-array` | harp-core | ndarray/PyTorchライクな配列API（遅延評価、キャッシュ） |
| `harp-backend-opencl` | harp-core | OpenCLバックエンド実装 |
| `harp-backend-metal` | harp-core | Metalバックエンド実装（macOSのみ） |
| `harp-dsl` | harp-core | DSLパーサー・コンパイラ |
| `harp-cli` | harp, harp-dsl | コマンドラインツール |
| `harp-viz` | harp, harp-dsl | 計算グラフ可視化 |
| `harp-nn` | harp-core | ニューラルネットワーク用ユーティリティ |

## クレート詳細

### harp（re-exportクレート）

`harp-core`の内容をすべてre-exportし、オプションでバックエンドも提供。

```toml
# Cargo.toml での使用例
[dependencies]
harp = { version = "0.1", features = ["opencl", "metal"] }
```

### harp-core（コアクレート）

テンソル計算の基盤となるコアライブラリ。

**主要モジュール:**
- `graph`: 計算グラフ表現（GraphNode, Graph, DType）
- `ast`: 抽象構文木（AstNode, Function, Literal）
- `backend`: バックエンド共通抽象化（Renderer trait, Pipeline, Device/Buffer/Kernel traits）
- `lowerer`: Graph→AST変換
- `opt`: 最適化パイプライン（graph最適化、ast最適化）

**仕様書:** [graph.md](graph.md), [ast.md](ast.md), [backend.md](backend.md), [lowerer.md](lowerer.md), [opt.md](opt.md)

### harp-array（配列API）

ndarray/PyTorchライクなAPIで配列計算を行うクレート。

**主要機能:**
- 遅延評価による計算グラフの構築
- コンパイル済みカーネルのキャッシュ
- 型レベル次元管理（`Dim0`-`Dim6`, `DimDyn`）
- バックエンド抽象化（ジェネリクスで注入）

**仕様書:** [array.md](array.md)

### harp-backend-opencl

OpenCL GPU向けバックエンド。CLikeRendererを実装。

### harp-backend-metal

Metal GPU向けバックエンド（macOS専用）。CLikeRendererを実装。

### harp-dsl

`.harp`ファイルのパースとグラフへのコンパイル。

### harp-cli

`harpc`コマンド。`.harp`ファイルからカーネルコードを生成。

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
