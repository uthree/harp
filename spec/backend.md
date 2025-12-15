# バックエンド

バックエンドはAST→実行可能コードへの変換と実行を担当します。

## モジュール構成

バックエンド関連のコードは以下のように構成されています：

### コアモジュール
- `mod.rs`: Renderer trait、KernelSignature、BufferSignatureの定義
- `traits.rs`: GPU実行用の共通trait定義（NativeContext, NativeBuffer, NativeKernel, NativeCompiler, KernelConfig）
- `sequence.rs`: 複数カーネル順次実行（CompiledProgram, KernelCallInfo, IntermediateBufferSpec）
- `execution.rs`: Pipeline、CompiledKernel、AST式評価関数
- `pipeline.rs`: 多フェーズ最適化パイプライン（`create_multi_phase_optimizer`, `MultiPhaseConfig`）
- `c_like.rs`: C言語系構文の共通レンダリングロジック（CLikeRenderer trait）、OptimizationLevel

### バックエンド実装
- `metal/`: Metalバックエンド（macOS GPU）
  - `mod.rs`: モジュール定義とre-export
  - `renderer.rs`: Metal Shading Languageレンダラー
  - `MetalCode`: Metal Shading Languageソースコードを表す型
  - `buffer.rs`, `context.rs`, `kernel.rs`, `compiler.rs`: GPU実行用の実装（native-metal feature）
- `opencl/`: OpenCLバックエンド（クロスプラットフォームGPU）
  - `mod.rs`: モジュール定義とre-export
  - `renderer.rs`: OpenCL Cレンダラー
  - `OpenCLCode`: OpenCL Cコードを表す型
  - `buffer.rs`, `context.rs`, `kernel.rs`, `compiler.rs`: GPU実行用の実装（native-opencl feature）

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

## GPU実行バックエンド

各バックエンド（Metal、OpenCL）はレンダラーとGPU実行の両方の機能を提供します。

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

主要メソッド：
- `execute()`: 設定済みサイズで実行
- `execute_with_sizes()`: 動的なグリッド/ローカルサイズで実行
- `config()`: カーネル設定を取得

#### NativeCompiler
カーネルソースをコンパイルしてNativeKernelを生成。

### 使用例（OpenCL）

```rust
use harp::backend::traits::{KernelConfig, NativeBuffer, NativeCompiler, NativeContext};
use harp::backend::opencl::{OpenCLNativeBuffer, OpenCLNativeCompiler, OpenCLNativeContext};

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

### Pipeline

`Pipeline`は、Graphから直接GPUカーネルを生成・実行するためのパイプラインです。

**処理フロー:**
1. Graphの最適化（多フェーズグラフ最適化）
2. AST抽出（lowerer）
3. ASTの最適化（ルールベース＋ビームサーチ）
4. カーネルソースのみをレンダリング（`KernelSourceRenderer` trait）
5. ネイティブコンパイル（`NativeCompiler`）
6. GPUカーネル実行（`NativeKernel`）

**使用例:**
```rust
use harp::backend::{Pipeline, NativeBuffer, NativeCompiler, NativeContext};
use harp::backend::opencl::{OpenCLNativeBuffer, OpenCLNativeCompiler, OpenCLNativeContext, OpenCLRenderer};
use harp::graph::{Graph, DType};

// パイプライン作成
let context = OpenCLNativeContext::new()?;
let renderer = OpenCLRenderer::new();
let compiler = OpenCLNativeCompiler::new();
let mut pipeline = Pipeline::new(renderer, compiler, context);

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

### スレッド数・グループ数の決定フロー

カーネル実行時のスレッド数（grid size）とグループサイズ（local/threadgroup size）は以下のフローで決定されます：

1. **Lowering段階**: `AstNode::Kernel`ノードの`default_grid_size`と`default_thread_group_size`に設定
2. **Pipeline**: AST Kernelから`KernelConfig`に情報を伝播
3. **実行時**: `KernelConfig`の値を使用してディスパッチ

**情報フロー:**
```
Graph → Lowerer → AST(Kernel) → Pipeline → KernelConfig → execute
                   ↓                  ↓
            default_grid_size    evaluate_dispatch_size()
            default_thread_group_size   ↓
                               [usize; 3] (実際のサイズ)
```

サイズ式（`[Box<AstNode>; 3]`）は`evaluate_ast_expr`関数で評価され、`shape_vars`（シェイプ変数）を参照して具体値に解決されます。

### 複数カーネルの順次実行

最適化の結果、1つのGraphが複数のカーネルに分割されることがあります。`CompiledProgram`はこれらのカーネルを正しい順序で実行する機能を提供します。

**主要型:**

- `KernelCallInfo`: カーネル呼び出し情報（名前、入出力バッファ、グリッドサイズ）
- `IntermediateBufferSpec`: 中間バッファ仕様（カーネル間で受け渡されるバッファ）
- `CompiledProgram`: コンパイル済みプログラム（複数カーネル対応）

**使用例:**
```rust
// compile_program()は複数カーネルに対応
let program = pipeline.compile_program(graph)?;

// 名前付きバッファで実行
let mut inputs = HashMap::new();
inputs.insert("a".to_string(), &input_a);
let mut outputs = HashMap::new();
outputs.insert("out".to_string(), &mut output);

program.execute(&context, &inputs, &mut outputs)?;

// または位置引数で実行（単一カーネルの場合に便利）
program.execute_positional(&context, &[&input_a], &mut [&mut output])?;
```

**動的サイズでの実行:**

`NativeKernel`トレイトの`execute_with_sizes`メソッドを使用して、実行時にグリッドサイズとローカルサイズを指定できます：

```rust
kernel.execute_with_sizes(
    &[&input],
    &mut [&mut output],
    [1024, 1, 1],  // grid_size
    [64, 1, 1],    // local_size
)?;
```

### Feature flags

- `native-opencl`: OpenCL GPU実行機能を有効化
- `native-metal`: Metal GPU実行機能を有効化（macOSのみ）

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
