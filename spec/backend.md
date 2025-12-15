# バックエンド

バックエンドはAST→実行可能コードへの変換と実行を担当します。

## モジュール構成

バックエンド関連のコードは以下のように構成されています：

### コアモジュール
- `mod.rs`: Renderer、Compiler、Kernel、Buffer、Device等の基本trait定義
- `device.rs`: デバイス抽象化（Metal、OpenCL等の実行環境）
- `pipeline.rs`: Pipeline trait（グラフからカーネルまでの処理フロー）
- `generic.rs`: GenericPipeline（任意のRendererとCompilerを組み合わせた汎用実装）
- `c_like.rs`: C言語系構文の共通レンダリングロジック（CLikeRenderer trait）、OptimizationLevel

### バックエンド実装
各バックエンドは独立したモジュールとして実装されています：

- `metal/`: Metalバックエンド（macOS GPU）
  - `mod.rs`, `renderer.rs`, `compiler.rs`, `kernel.rs`, `buffer.rs`
- `opencl/`: OpenCLバックエンド（クロスプラットフォームGPU）
  - `mod.rs`, `renderer.rs`, `compiler.rs`, `kernel.rs`, `buffer.rs`

各バックエンドは共通のtrait（Renderer、Compiler、Kernel、Buffer）を実装しており、バックエンドの切り替えが容易です。

## 主要コンポーネント

### Renderer
ASTをターゲット言語のソースコードに変換。C言語系の構文を持つ言語（C、Metal、OpenCL）は`CLikeRenderer` traitで共通ロジックを共有。

### Compiler
ソースコードをコンパイルしてKernel（実行可能バイナリ）を生成。

### Kernel
実行可能なカーネル。`KernelSignature`で入出力バッファーの形状情報を保持。

### Buffer
デバイス上のデータバッファー。型情報（dtype）を保持し、バイト列との相互変換、型付きベクタアクセスのデフォルト実装を提供。

### Query
カーネル実行時の入出力バッファーとshape変数をまとめた構造体。

### Pipeline
Graphを最適化、lower、AST最適化などの一通りの処理をまとめて行うためのtrait。

処理フロー（3フェーズグラフ最適化）:
1. **Preparation** (グラフ準備): View挿入、融合、タイリングなどのグラフ構造最適化
2. **Lowering**: 全てのGraphOpノードをKernelノードに変換（並列化戦略の選択を含む）
3. **Fusion**: 全てのKernelノードをProgramRootに融合（決定論的変換）
4. **AST最適化**:
   - ルールベース最適化（代数的簡約、定数畳み込み）
   - ビームサーチ最適化（ループ変換など）
5. **レンダリング**: AST → ソースコード
6. **コンパイル**: ソースコード → 実行可能カーネル

実測値ベース最適化（RuntimeSelector）はPhase 1とPhase 2で使用されます。Phase 3（Fusion）は決定論的な変換のため、RuntimeSelectorは使用されず、ビーム幅=1で高速に処理されます。

#### GenericPipeline
任意のRendererとCompilerを組み合わせて使用できる汎用Pipeline実装。

**主要な機能:**
- コンパイル済みKernelのキャッシュ機能
- 最適化履歴の収集（可視化ツールとの統合用）
- グラフ最適化とAST最適化（両方とも常に有効）

**設定フィールド:**
- `graph_config`: グラフ最適化の設定（OptimizationConfig）
- `ast_config`: AST最適化の設定（OptimizationConfig）
- `collect_histories`: 最適化履歴を収集するか（DEBUGビルドでデフォルトtrue、RELEASEビルドでfalse）

**OptimizationConfig:**
- `beam_width`: ビームサーチの幅（デフォルト: 4）
- `max_steps`: 最大ステップ数（デフォルト: 10000）
- `show_progress`: プログレスバー表示（デフォルト: false）
- `early_termination_threshold`: 早期終了閾値（デフォルト: Some(2)）
- `enable_runtime_selector`: RuntimeSelector（実測値ベース最適化）を有効化（デフォルト: false）
- `pre_filter_count`: RuntimeSelector使用時の静的コスト足切り候補数（デフォルト: 4）
- `measurement_count`: RuntimeSelector使用時の計測回数（デフォルト: 10）

**RuntimeSelector（実測値ベース最適化）:**
`enable_runtime_selector()`を呼び出すと、グラフ最適化とAST最適化の両方で実測値ベースの候補選択が有効になる。バッファは`Buffer::allocate`を使用して自動的に生成される。

**使用例:**
```rust
let mut pipeline = GenericPipeline::new(renderer, compiler);
pipeline.graph_config.beam_width = 8;

// 実測値ベース最適化を有効化
pipeline.enable_runtime_selector();
```

**最適化履歴:**
最適化の各ステップは`OptimizationHistories`に記録され、`pipeline.histories.graph`および`pipeline.histories.ast`からアクセス可能。可視化ツール（harp-viz）で最適化過程を確認できる。

## Nativeバックエンド（新アーキテクチャ）

`native`モジュールは、libloadingを使わずにRustから直接GPU APIを呼び出す新しいバックエンド実装を提供します。

### 特徴

- **libloading不要**: C言語のホストコード生成が不要
- **Rust型安全性**: GPU操作がRustの型システムで保護される
- **デバッグ容易性**: Rustコードなのでデバッグが容易
- **統一API**: OpenCLとMetalで共通のtraitインターフェース

### モジュール構成

- `native/mod.rs`: モジュール定義とtrait再エクスポート
- `native/traits.rs`: 共通trait定義（NativeContext, NativeBuffer, NativeKernel, NativeCompiler）
- `native/opencl/`: OpenCLネイティブ実装（`ocl`クレート使用）
- `native/metal/`: Metalネイティブ実装（`metal`クレート使用）

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

`NativePipeline`は、Graphから直接GPUカーネルを生成・実行するためのパイプラインです。GenericPipelineと同様の最適化機能を持ちつつ、libloadingを必要としません。

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

## 従来のバックエンド（libloading方式）

以下は従来のlibloadingベースのバックエンド実装です。

## 実装状況

### Metal Backend（macOS）
MetalのComputePipelineStateを使用してGPUで直接実行。

**実装方式:**
- Objective-C++でMetal API呼び出しコードを生成
- clang++でコンパイルし、動的ライブラリとしてロード

**コンパイラフラグ:**
- macOS: `-framework Metal -framework Foundation`

### OpenCL Backend（クロスプラットフォーム）
OpenCLを使ったクロスプラットフォームGPU実行バックエンド。ASTをOpenCLカーネルソース + ホストコードに変換し、動的ライブラリとしてコンパイルして実行。

**実装方式:**
- OpenCLカーネルソースを文字列リテラルとしてC言語コードに埋め込み
- ホストコードでOpenCL APIを直接呼び出し（プラットフォーム/デバイス取得、バッファ作成、カーネルビルド・実行、結果読み出し）
- libloadingでラッパー関数を呼び出す

**libloading対応:**
libloadingはRust側からバッファ情報を受け取るシグネチャを使用：
```c
// libloading用ラッパー（自動生成）
void __harp_entry(void** buffers, size_t* sizes, int* is_outputs, int num_buffers) {
    // OpenCLの初期化
    // カーネルソースのビルド
    // OpenCLバッファの作成（Host→Device転送）
    // カーネル引数設定・実行
    // 結果の読み出し（Device→Host転送）
    // リソース解放
}
```

**引数:**
- `buffers`: ホスト側のバッファポインタの配列
- `sizes`: 各バッファのバイトサイズ
- `is_outputs`: 出力バッファかどうかのフラグ（0: 入力、1: 出力）
- `num_buffers`: バッファの総数

**コンパイラフラグ:**
- macOS: `-framework OpenCL`
- Linux/Windows: `-lOpenCL`
