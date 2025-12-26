# バックエンド

バックエンドはAST→実行可能コードへの変換と実行を担当します。

## モジュール構成

バックエンド関連のコードは以下のように構成されています：

### コアモジュール
- `mod.rs`: Renderer trait、KernelSignature、BufferSignatureの定義
- `traits.rs`: GPU実行用の共通trait定義（Device, Buffer, Kernel, Compiler, KernelConfig）
- `global.rs`: グローバルデバイス管理（DeviceKind, set_default_device, get_default_device等）
- `sequence.rs`: 複数カーネル順次実行（CompiledProgram, KernelCallInfo, IntermediateBufferSpec, ExecutionQuery）
- `pipeline.rs`: Pipeline、PipelineConfig、CompiledKernel、KernelExecutionError、DispatchSizeConfig、DispatchSizeExpr、AST式評価関数、KernelSourceRenderer trait
- `c_like.rs`: C言語系構文の共通レンダリングロジック（CLikeRenderer trait）、OptimizationLevel、extract_buffer_placeholders関数

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

### グローバルデバイス管理

PyTorchライクなグローバルデバイス管理。`thread_local!`でスレッドごとにデフォルトデバイスを管理。

```rust
pub enum DeviceKind {
    None,    // デバイス未設定
    Metal,   // Metalバックエンド
    OpenCL,  // OpenCLバックエンド
}

// デバイス設定
pub fn set_default_device<D: Device + Send + Sync + 'static>(device: D, kind: DeviceKind);

// デバイス種類取得
pub fn get_default_device_kind() -> DeviceKind;

// デバイス取得（型指定）
pub fn get_default_device<D: Device + Send + Sync + 'static>() -> Option<Arc<D>>;

// デバイスが設定されているか
pub fn has_default_device() -> bool;

// デバイスクリア
pub fn clear_default_device();

// スコープ付きデバイス切り替え
pub fn with_device<D, F, R>(device: D, kind: DeviceKind, f: F) -> R;
```

**使用例:**

```rust
use harp::backend::{set_default_device, DeviceKind};

#[cfg(feature = "metal")]
{
    use harp::backend::metal::MetalDevice;
    let device = MetalDevice::new().unwrap();
    set_default_device(device, DeviceKind::Metal);
}

// Tensor.forward()でこのデバイスが使用される
```

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

## GPU実行バックエンド

各バックエンド（Metal、OpenCL）はレンダラーとGPU実行の両方の機能を提供します。

### 特徴

- **Rust型安全性**: GPU操作がRustの型システムで保護される
- **デバッグ容易性**: Rustコードなのでデバッグが容易
- **統一API**: OpenCLとMetalで共通のtraitインターフェース

### 主要trait

#### Device
GPUデバイスを表すtrait。バックエンドの利用可能性チェックとハードウェア特性情報を提供。

```rust
pub trait Device {
    fn is_available() -> bool;
    fn profile(&self) -> DeviceProfile;
    fn supports_feature(&self, feature: DeviceFeature) -> bool;
    fn supports_instruction(&self, instruction: DeviceInstruction) -> bool;
    fn simd_width(&self, dtype: &DType, op: OpKind) -> usize;
}
```

デバイス初期化（`new()`, `with_device()`）やデバイス名取得（`device_name()`）は各具体型（`OpenCLDevice`, `MetalDevice`）の固有メソッドとして実装されており、トレイトでは規定しない。

#### DeviceProfile / DeviceFeature / DeviceInstruction

デバイスのハードウェア特性を表す型。最適化時にデバイス固有のパラメータを決定するために使用。

```rust
pub struct DeviceProfile {
    pub device_type: DeviceType,          // CPU, GPU（統合/独立）, アクセラレータ
    pub compute_units: usize,             // 計算ユニット数
    pub max_work_group_size: usize,       // 最大ワークグループサイズ
    pub preferred_work_group_size_range: (usize, usize),  // 推奨範囲
    pub local_memory_size: usize,         // ローカルメモリサイズ
    pub warp_size: usize,                 // ワープ/ウェーブフロントサイズ
    pub preferred_tile_sizes: Vec<usize>, // 推奨タイルサイズ
    pub simd_capabilities: Vec<SimdCapability>,  // SIMD能力（dtype×op別）
}

pub enum DeviceFeature {
    FastMath, HalfPrecision, DoublePrecision,
    LocalMemory, AtomicOperations, SubgroupOperations,
}

pub enum DeviceInstruction {
    Fma, Rsqrt, AtomicAddFloat, NativeDiv, NativeExpLog,
}
```

#### OpKind / SimdCapability

演算子の種類とSIMD能力を表す型。データ型や演算の種類ごとに異なるSIMD幅をサポートするハードウェア特性を表現。

```rust
pub enum OpKind {
    Add, Mul, Div, Recip, Sqrt, Log2, Exp2, Sin,
    Fma, Compare, Load, Store,
}

pub struct SimdCapability {
    pub dtype: DType,    // データ型
    pub op: OpKind,      // 演算の種類
    pub width: usize,    // サポートされるベクトル幅
}
```

**クエリメソッド:**

```rust
impl DeviceProfile {
    // 特定dtype×opの最大SIMD幅を取得
    pub fn simd_width(&self, dtype: &DType, op: OpKind) -> usize;

    // 複数演算で共通のSIMD幅を取得（式全体のベクトル化用）
    pub fn common_simd_width(&self, dtype: &DType, ops: &[OpKind]) -> usize;

    // 利用可能なSIMD幅一覧（1, 2, 4, ...）
    pub fn available_simd_widths(&self, dtype: &DType, op: OpKind) -> Vec<usize>;

    // 全能力から一意の幅リストを取得
    pub fn all_simd_widths(&self) -> Vec<usize>;
}
```

**設計意図:** ハードウェアによっては演算やデータ型ごとにSIMD幅が異なる場合がある（例: F32のMulは4幅だがSqrtは2幅、F64はF32の半分など）。この設計により、そのような特性を正確に表現できる。

#### DeviceCapabilities

デバイス情報を最適化パイプラインに渡すための構造体。デバイスが持つ能力（何ができるか）を表現します。

```rust
pub struct DeviceCapabilities {
    pub profile: DeviceProfile,
    pub features: HashSet<DeviceFeature>,
    pub instructions: HashSet<DeviceInstruction>,
}

impl DeviceCapabilities {
    pub fn from_device<D: Device>(device: &D) -> Self;
    pub fn default_gpu() -> Self;  // デフォルトGPU向け設定
}
```

Pipelineは自動的にデバイスから`DeviceCapabilities`を作成し、Suggesterに渡します。これにより、タイルサイズ、スレッドグループサイズ、SIMD幅などがデバイス特性に基づいて最適化されます。

**条件付き最適化ルール:**
`DeviceInstruction`に基づいて適用される最適化ルールがあります。
- `DeviceInstruction::Fma`: `a * b + c` → `fma(a, b, c)` 変換を適用
- `DeviceInstruction::AtomicAddFloat`: 並列Reduceでatomic add使用可能

```rust
use harp::opt::ast::rules::{rules_for_capabilities, search_rules_for_capabilities};

// デバイスサポートに基づくルールセット取得
let rules = rules_for_capabilities(&caps);
let search_rules = search_rules_for_capabilities(&caps);  // ビームサーチ用
```

#### PipelineConfig

パイプラインの動作設定。

```rust
pub struct PipelineConfig {
    pub ast_beam_width: usize,     // AST最適化のビーム幅
    pub max_steps: usize,          // 最適化の最大ステップ数
    pub show_progress: bool,       // 進行状況表示
    pub collect_history: bool,     // 最適化履歴の収集
    pub fast_math: bool,           // 高速数学最適化の有効化
}
```

**fast_math:**
`fast_math: true`を設定すると、コンパイル時に高速数学オプションが適用されます。
- **OpenCL**: `-cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations`
- **Metal**: `CompileOptions.fastMathEnabled = true`

これによりパフォーマンスが向上しますが、数値精度が低下する可能性があります。

```rust
let config = PipelineConfig::default().with_fast_math(true);
```

#### Buffer
GPUメモリバッファ。ホスト⇔デバイス間のデータ転送を提供。関連型として`Dev: Device`と`Error`を持つ。

#### DynBuffer
`Buffer`トレイトの型消去版。`Tensor`内部でGPU/ホストバッファを抽象化するために使用。

```rust
pub trait DynBuffer: Send + Sync {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> DType;
    fn byte_len(&self) -> usize;
    fn read_to_host(&self) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>>;
    fn write_from_host(&mut self, data: &[u8]) -> Result<(), Box<dyn Error + Send + Sync>>;
    fn clone_buffer(&self) -> Box<dyn DynBuffer>;
}
```

**用途:**
- `TensorInner.buffer`フィールドで`Box<dyn DynBuffer>`として使用
- `realize()`実行後もGPUバッファを保持（ホストへのコピーを遅延）
- `data()`呼び出し時に必要に応じて`read_to_host()`でホストにコピー

**実装:**
- `MetalBuffer`, `OpenCLBuffer`: GPUバックエンドのバッファ
- `VecBuffer`: ホストデータ用の簡易ラッパー（`from_data()`等で使用）

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

`Pipeline`は、ASTからGPUカーネルを生成・実行するためのパイプラインです。

**処理フロー:**
1. ASTの最適化（ルールベース＋ビームサーチ）
2. カーネルソースのみをレンダリング（`KernelSourceRenderer` trait）
3. ネイティブコンパイル（`Compiler`）
4. GPUカーネル実行（`Kernel`）

**設定:**
```rust
{
    let config = pipeline.config_mut();
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

### バッファShape検証

`execute_with`および`BoundExecutionQuery::execute()`は、実行前にバッファのshapeを自動検証します。

**検証内容:**
- 入力/出力バッファの`shape()`と`KernelSignature`の期待されるshapeを比較
- 動的shape（`Expr::Var`を含む）は`shape_vars`で評価してから比較

**エラー型:**
```rust
pub enum KernelExecutionError<KE> {
    KernelError(KE),
    BufferNotFound(String),
    ShapeMismatch {           // バッファのshapeが期待と異なる
        buffer_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    ShapeEvaluationError(String),  // 動的shape評価時のエラー（未定義変数など）
}
```

**使用例:**
```rust
// 正しいshape
let mut buf = OpenCLBuffer::allocate(&device, vec![32, 128], DType::F32)?;
compiled_kernel.query()
    .input("x", &buf)
    .output("y", &mut out_buf)
    .shape_var("batch", 32)
    .shape_var("seq_len", 128)
    .execute()?;  // OK

// shapeが不正な場合
let wrong_buf = OpenCLBuffer::allocate(&device, vec![64, 64], DType::F32)?;
let result = compiled_kernel.query()
    .input("x", &wrong_buf)  // 期待: [32, 128], 実際: [64, 64]
    .output("y", &mut out_buf)
    .execute();  // Err(ShapeMismatch { buffer_name: "x", expected: [32, 128], actual: [64, 64] })
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
Tensor → TensorLowerer → AST(Kernel) → Pipeline → KernelConfig → execute
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

最適化の結果、1つのプログラムが複数のカーネルに分割されることがあります。`CompiledProgram`はこれらのカーネルを正しい順序で実行する機能を提供します。

**主要型:**

- `KernelCallInfo`: カーネル呼び出し情報（名前、入出力バッファ、グリッドサイズ）
- `IntermediateBufferSpec`: 中間バッファ仕様（カーネル間で受け渡されるバッファ）
- `CompiledProgram`: コンパイル済みプログラム（複数カーネル対応）

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

---

## 今後の実装予定

### 並列Reduce

現在のReduceは単一スレッドでの逐次リダクションですが、GPUでの大規模配列には非効率。並列Reduceを実装することで高速化が可能。

**2段階リダクションアルゴリズム:**
1. **Stage 1 (ワークグループ内)**: 各スレッドがストライドアクセスで累積 → ローカルメモリでTree-based reduction
2. **Stage 2 (グローバル集約)**: `AtomicAddFloat`サポート時はatomic_add、非サポート時は中間バッファ経由

**前提条件（実装済み）:**
- `AstNode::AtomicAdd`, `AstNode::AtomicMax`バリアント
- OpenCL/Metalでのatomicレンダリング
- `DeviceInstruction::AtomicAddFloat`検出
