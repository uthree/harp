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

## 実装状況

### Metal Backend（macOS/iOS）
完全実装。MetalのComputePipelineStateを使用してGPUで直接実行。

### C-like Backend
未実装。将来的に`tempfile`で一時ファイル生成→`libloading`で動的ロード予定。