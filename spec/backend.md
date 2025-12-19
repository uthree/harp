# バックエンド

バックエンドはAST→実行可能コードへの変換と実行を担当します。

## モジュール構成

バックエンド関連のコードは以下のように構成されています：

### コアモジュール
- `mod.rs`: Renderer trait、KernelSignature、BufferSignatureの定義
- `traits.rs`: GPU実行用の共通trait定義（Device, Buffer, Kernel, Compiler, KernelConfig）
- `sequence.rs`: 複数カーネル順次実行（CompiledProgram, KernelCallInfo, IntermediateBufferSpec, ExecutionQuery）
- `pipeline.rs`: Pipeline、PipelineConfig、CompiledKernel、KernelExecutionError、DispatchSizeConfig、DispatchSizeExpr、AST式評価関数、KernelSourceRenderer trait
- `c_like.rs`: C言語系構文の共通レンダリングロジック（CLikeRenderer trait）、OptimizationLevel、extract_buffer_placeholders関数

**注意**: グラフ最適化のファクトリ関数（`create_multi_phase_optimizer`, `MultiPhaseConfig`等）は
`opt/graph/factory.rs` に移動しました。後方互換性のため `backend` モジュールから re-export されています。

### バックエンド実装
- `metal/`: Metalバックエンド（macOS GPU）
  - `mod.rs`: モジュール定義とre-export
  - `renderer.rs`: Metal Shading Languageレンダラー
  - `MetalCode`: Metal Shading Languageソースコードを表す型
  - `buffer.rs`, `device.rs`, `kernel.rs`, `compiler.rs`: GPU実行用の実装（metal feature）
- `opencl/`: OpenCLバックエンド（クロスプラットフォームGPU）
  - `mod.rs`: モジュール定義とre-export
  - `renderer.rs`: OpenCL Cレンダラー
  - `OpenCLCode`: OpenCL Cコードを表す型
  - `buffer.rs`, `device.rs`, `kernel.rs`, `compiler.rs`: GPU実行用の実装（opencl feature）

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

### KernelSourceRenderer trait
GPU APIに直接渡せるカーネルソースコードのみを生成するためのtrait。`CLikeRenderer`を拡張します。

```rust
pub trait KernelSourceRenderer: CLikeRenderer {
    fn render_kernel_source(&mut self, program: &AstNode) -> String;
}
```

`Renderer::render()`がホストコード等を含む完全なコードを生成するのに対し、`render_kernel_source()`はGPU APIに直接渡せるカーネル関数のみを返します。PipelineがGPUコンパイラにソースを渡す際に使用されます。

Metal/OpenCLレンダラーの実装では、`AstNode::Function`ノードを`render_sequential_function_as_kernel()`メソッドでカーネル形式に変換します。

### extract_buffer_placeholders関数
AST本体から入出力バッファのプレースホルダー変数を自動抽出します（`c_like.rs`）。

```rust
pub fn extract_buffer_placeholders(body: &AstNode) -> (Vec<String>, bool)
```

- 戻り値: `(入力バッファ名のリスト, outputが存在するか)`
- `inputN`パターンの変数名と`output`変数を検出
- Loweringで空のパラメータリストを持つFunctionノードに対してバッファパラメータを自動生成するために使用

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

#### Device
GPUデバイスを表すマーカートレイト。バックエンドの利用可能性チェックのみを提供。

```rust
pub trait Device {
    fn is_available() -> bool;
}
```

デバイス初期化（`new()`, `with_device()`）やデバイス名取得（`device_name()`）は各具体型（`OpenCLDevice`, `MetalDevice`）の固有メソッドとして実装されており、トレイトでは規定しない。

#### Buffer
GPUメモリバッファ。ホスト⇔デバイス間のデータ転送を提供。

#### Kernel
コンパイル済みカーネル。バッファを受け取って実行。

主要メソッド：
- `execute()`: 設定済みサイズで実行
- `execute_with_sizes()`: 動的なグリッド/ローカルサイズで実行
- `config()`: カーネル設定を取得

#### Compiler
カーネルソースをコンパイルしてKernelを生成。

### 使用例（OpenCL）

```rust
use harp::backend::traits::{Buffer, Compiler, Device, KernelConfig};
use harp::backend::opencl::{OpenCLBuffer, OpenCLCompiler, OpenCLDevice};

// デバイス作成
let device = OpenCLDevice::new()?;

// カーネルソース
let source = r#"
    __kernel void add(__global float* a, __global float* b, __global float* c) {
        int i = get_global_id(0);
        c[i] = a[i] + b[i];
    }
"#;

// コンパイル
let compiler = OpenCLCompiler::new();
let config = KernelConfig::new("add").with_global_work_size([4, 1, 1]);
let kernel = compiler.compile(&device, source, config)?;

// バッファ作成
let a = OpenCLBuffer::from_vec(&device, vec![4], DType::F32, &[1.0, 2.0, 3.0, 4.0])?;
let b = OpenCLBuffer::from_vec(&device, vec![4], DType::F32, &[5.0, 6.0, 7.0, 8.0])?;
let c = OpenCLBuffer::allocate(&device, vec![4], DType::F32)?;

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
5. ネイティブコンパイル（`Compiler`）
6. GPUカーネル実行（`Kernel`）

**使用例:**
```rust
use harp::backend::{Buffer, Compiler, Device, Pipeline, ExecutionQuery};
use harp::backend::opencl::{OpenCLBuffer, OpenCLCompiler, OpenCLDevice, OpenCLRenderer};
use harp::graph::{Graph, DType};

// パイプライン作成
let device = OpenCLDevice::new()?;
let renderer = OpenCLRenderer::new();
let compiler = OpenCLCompiler::new();
let mut pipeline = Pipeline::new(renderer, compiler, device);

// グラフ作成
let mut graph = Graph::new();
let a = graph.input("a", DType::F32, vec![1024]);
let b = graph.input("b", DType::F32, vec![1024]);
let c = a + b;
graph.output("out", c);

// コンパイル
let compiled = pipeline.compile_graph(graph)?;

// バッファ作成
let mut input_a = OpenCLBuffer::allocate(&pipeline.context(), vec![1024], AstDType::F32)?;
let mut input_b = OpenCLBuffer::allocate(&pipeline.context(), vec![1024], AstDType::F32)?;
let mut output = OpenCLBuffer::allocate(&pipeline.context(), vec![1024], AstDType::F32)?;

// データ書き込み
input_a.write_vec(&vec![1.0f32; 1024])?;
input_b.write_vec(&vec![2.0f32; 1024])?;

// 実行（名前ベース - ExecutionQuery使用）
let query = ExecutionQuery::new()
    .input("a", &input_a)
    .input("b", &input_b)
    .output("out", &mut output);
compiled.execute_with(query)?;

// 結果読み出し
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

### ExecutionQuery

`ExecutionQuery`は、バッファを名前ベースで指定するためのビルダー構造体です。動的shape変数もサポートしています。

**利点:**
- バッファの順序を気にせず名前で指定可能
- 必要なバッファの欠落を実行前にチェック
- 型安全なフルエントAPI
- 動的shape変数による実行時のグリッドサイズ計算

**使用例:**
```rust
use harp::backend::ExecutionQuery;

// ビルダーパターンでバッファと動的shapeを指定
let query = ExecutionQuery::new()
    .input("a", &buf_a)
    .input("b", &buf_b)
    .output("result", &mut buf_out)
    .shape_var("batch_size", 32)
    .shape_var("seq_len", 128);

// CompiledKernel での実行（shape_varsに基づいてグリッドサイズを動的計算）
compiled_kernel.execute_with(query)?;

// CompiledProgram での実行
compiled_program.execute_with(context, query)?;
```

**動的shape変数:**

`shape_var()`メソッドで指定した変数は、実行時にグリッドサイズの計算に使用されます。これにより、コンパイル時に固定されないサイズでもカーネルを実行できます。

```rust
// バッチサイズが実行時に決まる場合
for batch_size in [16, 32, 64, 128] {
    let query = ExecutionQuery::new()
        .input("x", &input_buf)
        .output("y", &mut output_buf)
        .shape_var("batch_size", batch_size as isize);

    compiled_kernel.execute_with(query)?;
}
```

**実行メソッドの比較:**

| メソッド | バッファ指定 | 動的shape | 用途 |
|----------|-------------|-----------|------|
| `execute(&[&B], &mut [&mut B])` | 位置引数 | ✗ | シンプルなケース |
| `execute_with(ExecutionQuery)` | 名前ベース | ✓ | 複雑なグラフ、動的サイズ |
| `query().input().output().execute()` | 名前ベース | ✓ | フルエントAPI（推奨） |
| `execute_positional(...)` | 位置引数 | ✗ | CompiledProgramの便利メソッド |

### BoundExecutionQuery（フルエントAPI）

`BoundExecutionQuery`は`CompiledKernel`にバインドされた`ExecutionQuery`で、よりフルエントなAPIを提供します。

**使用例:**
```rust
// query()でデフォルトshape変数が初期化済みのクエリを取得
compiled_kernel.query()
    .input("x", &input_buf)
    .output("y", &mut output_buf)
    .shape_var("batch_size", batch_size as isize)
    .execute()?;
```

**特徴:**
- `query()`メソッドが`KernelSignature`のデフォルト`shape_vars`で初期化された`ExecutionQuery`を返す
- 指定しないshape変数はデフォルト値が使用される
- チェーン呼び出しで`execute()`まで一気に実行可能

**フロー:**
```
CompiledKernel::query()
    → BoundExecutionQuery (デフォルトshape_vars設定済み)
    → .input() / .output() / .shape_var() で設定を追加
    → .execute() で実行
```

### DispatchSizeConfig / DispatchSizeExpr

`DispatchSizeConfig`はグリッドサイズとローカルサイズを式として保持し、実行時にshape変数を使って評価します。

```rust
// 内部的にCompiledKernelがDispatchSizeConfigを保持
pub struct CompiledKernel<K, B> {
    pub kernel: K,
    pub signature: KernelSignature,
    pub dispatch_config: DispatchSizeConfig,  // グリッドサイズ式
}

// DispatchSizeExprで式を表現
pub enum DispatchSizeExpr {
    Const(isize),           // 定数
    Var(String),            // 変数参照
    Add(Box<...>, Box<...>), // 加算
    Mul(Box<...>, Box<...>), // 乗算
    Div(Box<...>, Box<...>), // 除算
    // ...
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

### エントリポイント名の解決

Pipelineは`extract_entry_point_name`内部メソッドでASTからカーネル/関数名を自動解決します：

1. `AstNode::Kernel`ノードの`name`フィールドを優先検索
2. 見つからなければ`AstNode::Function`ノードの`name`フィールドを検索
3. どちらも見つからなければ`"main"`をデフォルト値として使用

これにより、Loweringで生成された任意の名前を持つカーネルが正しくコンパイル・実行されます。

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
let device = pipeline.context();  // デバイスを取得

// 名前付きバッファで実行
let mut inputs = HashMap::new();
inputs.insert("a".to_string(), &input_a);
let mut outputs = HashMap::new();
outputs.insert("out".to_string(), &mut output);

program.execute(device, &inputs, &mut outputs)?;

// または位置引数で実行（単一カーネルの場合に便利）
program.execute_positional(device, &[&input_a], &mut [&mut output])?;
```

**動的サイズでの実行:**

`Kernel`トレイトの`execute_with_sizes`メソッドを使用して、実行時にグリッドサイズとローカルサイズを指定できます：

```rust
kernel.execute_with_sizes(
    &[&input],
    &mut [&mut output],
    [1024, 1, 1],  // grid_size
    [64, 1, 1],    // local_size
)?;
```

### Feature flags

- `opencl`: OpenCL GPU実行機能を有効化
- `metal`: Metal GPU実行機能を有効化（macOSのみ）

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
