# バックエンド

バックエンドはAST→実行可能コードへの変換と実行を担当します。

## モジュール構成

バックエンド関連のコードは以下のように構成されています：

### コアモジュール
- `mod.rs`: Renderer trait、KernelSignature、BufferSignatureの定義
- `pipeline.rs`: 多フェーズ最適化パイプライン（`create_multi_phase_optimizer`, `MultiPhaseConfig`）
- `c_like.rs`: C言語系構文の共通レンダリングロジック（CLikeRenderer trait）、OptimizationLevel

### レンダラー実装
- `metal/`: Metalバックエンド（macOS GPU）
  - `mod.rs`, `renderer.rs`
  - `MetalCode`: Metal Shading Languageソースコードを表す型
- `opencl/`: OpenCLバックエンド（クロスプラットフォームGPU）
  - `mod.rs`, `renderer.rs`
  - `OpenCLCode`: OpenCL Cコードを表す型

### Nativeバックエンド
- `native/`: Rustから直接GPU APIを呼び出すバックエンド
  - `mod.rs`: モジュール定義とtrait再エクスポート
  - `traits.rs`: 共通trait定義
  - `opencl/`: OpenCLネイティブ実装（`ocl`クレート使用）
  - `metal/`: Metalネイティブ実装（`metal`クレート使用）

## 主要コンポーネント

### Renderer trait
ASTをターゲット言語のソースコードに変換するtrait。

```rust
pub trait Renderer {
    type CodeRepr: Into<String>;
    type Option;
    fn render(&self, program: &AstNode) -> Self::CodeRepr;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, option: Self::Option) {}
}
```

C言語系の構文を持つ言語（OpenCL、Metal）は`CLikeRenderer` traitで共通ロジックを共有。

### KernelSignature / BufferSignature
カーネルの入出力バッファーの形状情報を表す構造体。

```rust
pub struct KernelSignature {
    pub inputs: Vec<BufferSignature>,
    pub outputs: Vec<BufferSignature>,
    pub shape_vars: HashMap<String, isize>,
}

pub struct BufferSignature {
    pub name: String,
    pub shape: Vec<Expr>,
}
```

## パイプライン（最適化フロー）

### MultiPhaseConfig / create_multi_phase_optimizer

Graph最適化を3フェーズで行うためのパイプライン。

**処理フロー:**
1. **Preparation** (グラフ準備): View挿入、融合、タイリングなどのグラフ構造最適化
2. **Lowering**: 全てのGraphOpノードをKernelノードに変換（並列化戦略の選択を含む）
3. **Fusion**: 全てのKernelノードをProgramRootに融合（決定論的変換）

**設定オプション:**
- `beam_width`: ビームサーチの幅（デフォルト: 4）
- `max_steps`: 最大ステップ数（デフォルト: 10000）
- `show_progress`: プログレスバー表示（デフォルト: false）
- `collect_logs`: 最適化ログの収集（デフォルト: false）

**使用例:**
```rust
use harp::backend::{create_multi_phase_optimizer, MultiPhaseConfig};
use harp::opt::graph::GraphOptimizer;

let config = MultiPhaseConfig::new()
    .with_beam_width(4)
    .with_max_steps(1000)
    .with_progress(false);

let optimizer = create_multi_phase_optimizer(config);
let (optimized_graph, history) = optimizer.optimize_with_history(graph);
```

## Nativeバックエンド

`native`モジュールは、Rustから直接GPU APIを呼び出すバックエンド実装を提供します。

### 特徴

- **Rust型安全性**: GPU操作がRustの型システムで保護される
- **デバッグ容易性**: Rustコードなのでデバッグが容易
- **統一API**: OpenCLとMetalで共通のtraitインターフェース

### 主要trait

#### NativeContext
GPU実行コンテキスト。デバイスの初期化とリソース管理を担当。

#### NativeBuffer
GPUメモリバッファ。ホスト⇔デバイス間のデータ転送を提供。

#### NativeKernel
コンパイル済みカーネル。バッファを受け取って実行。

#### NativeCompiler
カーネルソースをコンパイルしてNativeKernelを生成。

### 使用例（OpenCL）

```rust
use harp::backend::native::{KernelConfig, NativeBuffer, NativeCompiler, NativeContext};
use harp::backend::native::opencl::{OpenCLNativeBuffer, OpenCLNativeCompiler, OpenCLNativeContext};

// コンテキスト作成
let context = OpenCLNativeContext::new()?;

// カーネルソース
let source = r#"
    __kernel void add(__global float* a, __global float* b, __global float* c) {
        int i = get_global_id(0);
        c[i] = a[i] + b[i];
    }
"#;

// コンパイル
let compiler = OpenCLNativeCompiler::new();
let config = KernelConfig::new("add").with_global_work_size([4, 1, 1]);
let kernel = compiler.compile(&context, source, config)?;

// バッファ作成
let a = OpenCLNativeBuffer::from_vec(&context, vec![4], DType::F32, &[1.0, 2.0, 3.0, 4.0])?;
let b = OpenCLNativeBuffer::from_vec(&context, vec![4], DType::F32, &[5.0, 6.0, 7.0, 8.0])?;
let c = OpenCLNativeBuffer::allocate(&context, vec![4], DType::F32)?;

// 実行
kernel.execute_with_buffers(&[&a, &b, &c])?;

// 結果読み出し
let result: Vec<f32> = c.read_vec()?;  // [6.0, 8.0, 10.0, 12.0]
```

### NativePipeline

`NativePipeline`は、Graphから直接GPUカーネルを生成・実行するためのパイプラインです。

**処理フロー:**
1. Graphの最適化（多フェーズグラフ最適化）
2. AST抽出（lowerer）
3. ASTの最適化（ルールベース＋ビームサーチ）
4. カーネルソースのみをレンダリング（`KernelSourceRenderer` trait）
5. ネイティブコンパイル（`NativeCompiler`）
6. GPUカーネル実行（`NativeKernel`）

**使用例:**
```rust
use harp::backend::native::{NativePipeline, NativeBuffer, NativeCompiler, NativeContext};
use harp::backend::native::opencl::{OpenCLNativeBuffer, OpenCLNativeCompiler, OpenCLNativeContext};
use harp::backend::opencl::OpenCLRenderer;
use harp::graph::{Graph, DType};

// パイプライン作成
let context = OpenCLNativeContext::new()?;
let renderer = OpenCLRenderer::new();
let compiler = OpenCLNativeCompiler::new();
let mut pipeline = NativePipeline::new(renderer, compiler, context);

// グラフ作成
let mut graph = Graph::new();
let a = graph.input("a", DType::F32, vec![1024]);
let b = graph.input("b", DType::F32, vec![1024]);
let c = a + b;
graph.output("out", c);

// コンパイル
let compiled = pipeline.compile_graph(graph)?;

// バッファ作成・実行
let mut input_a = OpenCLNativeBuffer::allocate(&pipeline.context(), vec![1024], AstDType::F32)?;
let mut input_b = OpenCLNativeBuffer::allocate(&pipeline.context(), vec![1024], AstDType::F32)?;
let mut output = OpenCLNativeBuffer::allocate(&pipeline.context(), vec![1024], AstDType::F32)?;

// データ書き込み・カーネル実行・結果読み出し
input_a.write_vec(&vec![1.0f32; 1024])?;
input_b.write_vec(&vec![2.0f32; 1024])?;
compiled.execute(&[&input_a, &input_b], &mut [&mut output])?;
let result: Vec<f32> = output.read_vec()?;
```

**設定:**
```rust
{
    let config = pipeline.config_mut();
    config.graph_beam_width = 2;    // グラフ最適化のビーム幅
    config.ast_beam_width = 2;      // AST最適化のビーム幅
    config.max_steps = 1000;        // 最大最適化ステップ数
    config.show_progress = true;    // プログレス表示
    config.collect_history = true;  // 最適化履歴の収集
}
```

### Feature flags

- `native-opencl`: OpenCLネイティブバックエンドを有効化
- `native-metal`: Metalネイティブバックエンドを有効化（macOSのみ）

## CLI (harpc)

CLIツール`harpc`を使用して.harpファイルをコンパイルできます。

```bash
# OpenCLカーネルを生成
harpc input.harp --target opencl

# Metal Shading Languageを生成
harpc input.harp --target metal

# 標準入力から読み込み
cat input.harp | harpc - --target opencl

# ソースコードを埋め込んでコメントとして出力
harpc input.harp --embed-source
```

**オプション:**
- `--target <opencl|metal>`: 出力ターゲット
- `--output <FILE>`: 出力ファイル
- `--emit <code|graph|ast>`: 出力形式
- `--embed-source`: 元のDSLソースをコメントとして埋め込む

**サブコマンド:**
- `compile`: ファイルをコンパイル（デフォルト）
- `check`: 構文チェックのみ
- `ast`: ASTを出力
- `fmt`: ソースコードをフォーマット
