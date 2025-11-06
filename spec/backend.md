# バックエンド

## ファイル構成

### Metal Backend
- `src/backend/metal/mod.rs` - モジュール定義、MetalCode構造体
- `src/backend/metal/buffer.rs` - MetalBuffer実装
- `src/backend/metal/kernel.rs` - MetalKernel実装
- `src/backend/metal/compiler.rs` - MetalCompiler実装 (159行)
- `src/backend/metal/renderer.rs` - MetalRenderer実装

### C-like Backend
- `src/backend/c_like/mod.rs` - C言語系バックエンド（未実装）

## Renderer
ASTを実際のGPUなどのためのソースコードとして描画する。
`trait CLikeRenderer` を使ってC言語に近い構文の言語のレンダリング処理を共通化したい。

## Compiler
コンパイラを扱うためのもの。
コンパイル処理を実行して、Kernel（バイナリファイル）を得る。

## Kernel
実行可能なバイナリ

### 実装状況
- **Metal Backend**: MetalKernelとして実装済み。ComputePipelineStateを使用してGPUで直接実行。
- **C-like Backend**: 未実装。将来的に`tempfile`などを用いて一時的にファイルとして保持し、必要になったときは`libloading`で動的に呼び出す予定。

### KernelSignature
カーネルが受け取るバッファーの形状とShapeの変数などをまとめた構造体

## Buffer
Kernelに渡すためのデバイス上のデータ。

## Query
Bufferをひとまとめにしたもの。