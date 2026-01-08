# Harp プロジェクト概要

Harpは計算グラフから効率的なGPU/CPUカーネルを生成するトランスコンパイラです。

**注意**: 現在リファクタリング中のため、上位レイヤー（Tensor, nn, data）は削除されています。
再設計後に新しい高レベルAPIが追加される予定です。

## ディレクトリ構成

```
harp/
├── Cargo.toml              # パッケージ設定
├── src/
│   ├── lib.rs              # ルートモジュール
│   ├── ast/                # 抽象構文木
│   ├── backend/            # バックエンドトレイト・Pipeline
│   ├── opt/                # AST最適化
│   ├── shape/              # 形状式（Expr, View）
│   ├── viz/                # 可視化TUI (feature: viz)
│   └── backends/           # バックエンド実装
│       ├── c/              # Cコード生成 (feature: c)
│       ├── opencl/         # OpenCL (feature: opencl)
│       └── metal/          # Metal (feature: metal, macOSのみ)
├── examples/               # サンプルコード
├── tests/                  # 統合テスト
└── spec/                   # 仕様書
```

## Feature Flags

| Feature | 説明 |
|---------|------|
| `c` | Cバックエンドを有効化 |
| `opencl` | OpenCLバックエンドを有効化 |
| `metal` | Metalバックエンドを有効化（macOSのみ） |
| `viz` | 最適化履歴可視化TUIを有効化 |

```toml
# Cargo.toml での使用例
[dependencies]
harp = { version = "0.1", features = ["opencl"] }
# macOSの場合
harp = { version = "0.1", features = ["metal"] }
```

## 主要モジュール

### ast

抽象構文木（AstNode）の定義。計算グラフの中間表現として使用されます。

**主要な型:**
- `AstNode`: 計算ノード（演算、定数、変数、ループ、関数など）
- `DType`: データ型（F32, F64, I32, etc.）
- `Literal`: 定数値

**仕様書:** [ast.md](ast.md)

### shape

形状式とビューの定義。動的・静的な形状表現をサポートします。

**主要な型:**
- `Expr`: 形状式（定数、算術演算、ループインデックス）
- `View`: メモリレイアウト（Linear, IndexExpr, Masked）
- `PadValue`: パディング値（Zero, One, NegInf）

### backend

バックエンドトレイトとPipeline。

**主要な型:**
- `Device`: デバイストレイト
- `Buffer`: バッファトレイト
- `Kernel`: カーネルトレイト
- `Compiler`: コンパイラトレイト
- `Pipeline`: AST→カーネルのパイプライン
- `Renderer`, `CLikeRenderer`: コードレンダラートレイト

**仕様書:** [backend.md](backend.md)

### opt

AST最適化パイプライン。ループ融合、タイリング、ベクトル化などを提供。

**仕様書:** [opt.md](opt.md), [opt-ast.md](opt-ast.md)

### viz (feature: viz)

最適化履歴可視化TUI。ratatuiを使用してターミナルで最適化ステップを対話的に確認できます。

**機能:**
- 左右キー（←/→ または h/l）: ステップ間を移動
- 上下キー（↑/↓ または j/k）: 候補を選択
- 2ペインレイアウト: ソースコード（syntectでハイライト）+ 候補リスト
- ステータスバー: ステップ番号、コスト、Suggester名

## バックエンド実装

### backends::c (feature: c)

Cコードを生成するバックエンド。実行機能は持たず、コード生成のみ。

### backends::opencl (feature: opencl)

OpenCLを使用したGPU実行バックエンド。

### backends::metal (feature: metal, macOSのみ)

Apple Metal APIを使用したGPU実行バックエンド。

## 使用例

### AST構築とレンダリング

```rust
use harp::ast::{AstNode, DType, Literal};
use harp::ast::helper::*;

// AST構築
let a = var("a");
let b = var("b");
let expr = add(a, b);

// レンダリング
use harp::renderer::GenericRenderer;
let renderer = GenericRenderer::default();
let code = renderer.render(&expr);
```

### Pipeline によるカーネルコンパイル

```rust
use harp::backend::{Pipeline, KernelSignature, BufferSignature};
use harp::shape::Expr;

// Pipeline作成（要: バックエンド有効化）
#[cfg(feature = "metal")]
{
    use harp::backends::metal::{MetalDevice, MetalCompiler, MetalRenderer};

    let device = MetalDevice::new().unwrap();
    let renderer = MetalRenderer::default();
    let compiler = MetalCompiler::new();
    let mut pipeline = Pipeline::new(renderer, compiler, device);

    // ASTをコンパイル
    let signature = KernelSignature::new(
        vec![BufferSignature::new("input".to_string(), vec![Expr::Const(32)])],
        vec![BufferSignature::new("output".to_string(), vec![Expr::Const(32)])],
    );
    let compiled = pipeline.compile_ast(ast, signature).unwrap();
}
```

## 今後の予定

- 新しい高レベルTensor API
- 自動微分のサポート
- ニューラルネットワーク層
