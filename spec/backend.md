# バックエンド
## Renderer
ASTを実際のGPUなどのためのソースコードとして描画する。
`trait CLikeRenderer` を使ってC言語に近い構文の言語のレンダリング処理を共通化したい。

## Compiler
コンパイラを扱うためのもの。
コンパイル処理を実行して、Kernel（バイナリファイル）を得る。

## Kernel
実行可能なバイナリ
`tempfile`などを用いて一時的にファイルとして保持し、必要になったときは`libloading`で動的に呼び出す。

## Buffer
Kernelに渡すためのデバイス上のデータ。

## Query
Bufferをひとまとめにしたもの。