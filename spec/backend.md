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
デバイス上のデータバッファー。

### Query
カーネル実行時の入出力バッファーとshape変数をまとめた構造体。

### Pipeline
Graphを最適化、lower、AST最適化などの一通りの処理をまとめて行うためのtrait

## 実装状況

### Metal Backend（macOS/iOS）
MetalのComputePipelineStateを使用してGPUで直接実行。
