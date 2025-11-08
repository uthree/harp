# バックエンド

バックエンドはAST→実行可能コードへの変換と実行を担当します。

## 主要コンポーネント

### Renderer
ASTをターゲット言語のソースコードに変換。C言語系の構文を持つ言語は`CLikeRenderer` traitで共通化予定。

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

#### GenericPipeline
任意のRendererとCompilerを組み合わせて使用できる汎用Pipeline実装。コンパイル済みKernelをキーでキャッシュする機能を提供。

## 実装状況

### Metal Backend（macOS/iOS）
MetalのComputePipelineStateを使用してGPUで直接実行。

### OpenMP Backend（クロスプラットフォーム）
C言語とOpenMPを使ったCPUバックエンド。ASTをC言語コードに変換し、OpenMPの`#pragma omp parallel for`でカーネル関数を並列実行することでGPU並列実行を擬似的に再現。動的ライブラリ(.so/.dylib/.dll)としてコンパイルして実行。
