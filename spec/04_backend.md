# Backend モジュール仕様

## 概要

Backendモジュールはプラットフォーム固有の実装を抽象化します。ASTからターゲット言語のソースコードを生成し、コンパイルして実行する機能を提供します。

## トレイト階層

### Buffer

メモリバッファの抽象化。

```rust
pub trait Buffer {
    fn shape(&self) -> Vec<usize>;
    fn allocate(dtype: DType, shape: Vec<usize>) -> Self;
}
```

**要件:**
- ドロップ時にメモリを自動解放すること

### Kernel

コンパイル済みカーネルの抽象化。

```rust
pub trait Kernel {
    type Buffer: Buffer;

    fn signature(&self) -> &GraphSignature;
    fn call(&mut self, buffers: Vec<Self::Buffer>, shape_variables: &[usize]) -> Vec<Self::Buffer>;
}
```

**役割:**
- カーネルの入出力シグネチャを提供
- バッファとシェイプ変数を受け取って実行

### Compiler

コードをカーネルにコンパイルする機能。

```rust
pub trait Compiler {
    type CodeRepr;
    type Buffer: Buffer;
    type KernelType: Kernel<Buffer = Self::Buffer>;
    type Option;

    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, option: Self::Option);
    fn compile(&mut self, code: &Self::CodeRepr, details: GraphSignature) -> Self::KernelType;
}
```

### Renderer

ASTをソースコードに変換。

```rust
pub trait Renderer {
    type CodeRepr;
    type Option;

    fn new() -> Self;
    fn with_option(&mut self, option: Self::Option);
    fn render(&mut self, program: Program) -> Self::CodeRepr;
}
```

### Backend

バックエンド全体の統合。

```rust
pub trait Backend {
    type Buffer: Buffer;
    type Option;
    type Compiler: Compiler;
    type Renderer: Renderer<CodeRepr = <Self::Compiler as Compiler>::CodeRepr>;

    fn new() -> Self;
    fn with_option(&mut self, option: Self::Option);
    fn is_available(&self) -> bool;
    fn execute(&mut self, graph: &Graph, inputs: Vec<Self::Buffer>) -> Vec<Self::Buffer>;
}
```

## C言語系レンダラーの共通化

### CLikeRenderer trait

C、CUDA、Metal、OpenCLなど、C言語に近い構文を持つ言語のレンダラーで共通化できるロジックを抽象化したtrait。

**定義場所:** `src/backend/c_like.rs`

**提供機能:**
- 演算子の優先順位処理
- 二項演算子・単項演算子のレンダリング
- 制御構文（if, for, block）のレンダリング
- 関数定義のレンダリング
- インデント管理
- 型の再帰的レンダリング
- スコープとメモリ管理

**カスタマイズポイント（言語固有の実装）:**
```rust
pub trait CLikeRenderer {
    // インデント管理
    fn indent_level(&self) -> usize;
    fn set_indent_level(&mut self, level: usize);
    fn memory_config(&self) -> &MemoryConfig;

    // 言語固有のカスタマイズ
    fn render_includes(&self) -> String;
    fn render_scalar_dtype(&self, dtype: &DType) -> String;
    fn render_const(&self, c: &ConstLiteral) -> String;
    fn render_math_function(&mut self, name: &str, args: Vec<String>) -> String;
    fn render_barrier(&self) -> String;

    // 共通実装（デフォルト実装を提供）
    fn precedence(&self, node: &AstNode) -> u8 { ... }
    fn render_with_parens(&mut self, node: &AstNode, ...) -> String { ... }
    fn render_node(&mut self, node: &AstNode) -> String { ... }
    // ... その他多数
}
```

**メモリ管理設定:**
```rust
pub struct MemoryConfig {
    pub alloc_fn: String,      // 例: "malloc", "cudaMalloc"
    pub dealloc_fn: String,    // 例: "free", "cudaFree"
    pub needs_cast: bool,      // 割り当て結果をキャストするか
}
```

**利点:**
1. 新しいレンダラー（CUDA、Metal等）の実装が容易
2. コードの重複を削減（約700行の共通ロジック）
3. バグ修正や機能追加が全レンダラーに反映される

## Cバックエンド実装

### CBackend

C言語をターゲットとするバックエンド。

**構成要素:**
- CRenderer: ASTからC言語コードを生成（CLikeRendererを実装）
- CCompiler: C言語コードをコンパイルして動的ライブラリを生成
- CBuffer: libc経由のメモリ管理

### CBuffer

```rust
pub struct CBuffer {
    ptr: *mut c_void,
    dtype: DType,
    shape: Vec<usize>,
}
```

**特徴:**
- `libc::malloc`/`libc::free`でメモリ管理
- Drop時に自動解放

**主要メソッド:**
- `new(dtype, shape)`: メモリを確保
- `from_vec(data, shape)`: Vecから作成
- `to_vec()`: データをVecとして取得

### CRenderer

ASTをC言語コードに変換。`CLikeRenderer` traitを実装。

**実装内容:**
- `CLikeRenderer`の共通ロジックを利用
- C言語固有の部分のみを実装（約380行）

**C言語固有の実装:**
```rust
impl CLikeRenderer for CRenderer {
    fn render_includes(&self) -> String {
        // #include <math.h>, <stddef.h>, etc.
    }

    fn render_scalar_dtype(&self, dtype: &DType) -> String {
        // float, size_t, ssize_t, int, void
    }

    fn render_const(&self, c: &ConstLiteral) -> String {
        // INFINITY, NAN, 整数リテラル
    }
}
```

**生成されるコード:**
- 関数定義
- カーネル定義（将来的にOpenCLカーネルにも対応予定）
- 変数宣言
- ループ構造（for）
- 条件分岐（if-else）
- 算術演算
- メモリアクセス

**最適化:**
- `#pragma unroll`ヒントの挿入
- 型キャストの適切な配置

**レンダリング例:**

ASTノード:
```rust
AstNode::Range {
    counter_name: "i",
    start: 0,
    max: 10,
    step: 1,
    body: ...,
    unroll: Some(0),
}
```

生成されるC言語コード:
```c
#pragma unroll
for (size_t i = 0; i < 10; i += 1) {
    // body
}
```

### CCompiler

C言語コードをコンパイル。

**コンパイルプロセス:**
1. 一時ファイルにソースコードを書き込み
2. システムのCコンパイラ（gcc/clang）を呼び出し
3. 動的ライブラリ（.so/.dylib/.dll）を生成
4. `libloading`で動的ライブラリをロード
5. エントリーポイント関数のシンボルを取得

**コンパイラオプション:**
- `-O3`: 最適化レベル3
- `-fPIC`: 位置独立コード
- `-shared`: 共有ライブラリとして生成
- `-lm`: 数学ライブラリのリンク

### CKernel

コンパイル済みのCカーネル。

```rust
pub struct CKernel {
    _lib: Library,
    entry_point: Symbol<EntryPointFn>,
    signature: GraphSignature,
}
```

**実行時:**
1. 入力バッファのポインタ配列を作成
2. シェイプ変数の配列を作成
3. エントリーポイント関数を呼び出し
4. 出力バッファのポインタを取得

## Genericバックエンド

汎用的なバックエンド実装の基盤。

**提供機能:**
- グラフの最適化
- Lowererの呼び出し
- AST最適化
- Renderer/Compilerの呼び出し

## バックエンドの選択

バックエンドは文字列名で指定：

```rust
let tensor = Tensor::from_vec(data, &[2, 3], "c");  // Cバックエンド
```

**利用可能なバックエンド:**
- `"c"`: Cバックエンド（`backend-c`フィーチャーが有効な場合）

## バックエンド検証

```rust
pub fn validate_backend(backend_name: &str) {
    match backend_name {
        #[cfg(feature = "backend-c")]
        "c" => {
            let backend = CBackend::new();
            if !backend.is_available() {
                panic!("C backend is not available");
            }
        }
        _ => panic!("Unknown backend: {}", backend_name),
    }
}
```
