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

処理フロー:
1. **グラフ最適化** (必須): LoweringSuggesterでGraphOpをKernelノードに変換、融合、並列化戦略変更など
2. **Lowering**: Graph → AST変換（Kernelノードの展開）
3. **AST最適化** (オプション):
   - ルールベース最適化（代数的簡約、定数畳み込み）
   - ビームサーチ最適化（ループ変換など）
4. **レンダリング**: AST → ソースコード
5. **コンパイル**: ソースコード → 実行可能カーネル

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
- ホストコードでOpenCLの初期化、コンパイル、実行を行う
- libloadingでラッパー関数を呼び出す

**libloading対応:**
libloadingは固定シグネチャ `fn(*mut *mut u8)` を期待するため、Rendererは自動的にラッパー関数を生成する：
```c
// libloading用ラッパー（自動生成）
void __harp_entry(void** buffers) {
    // カーネル実行ロジック
}
```

**コンパイラフラグ:**
- macOS: `-framework OpenCL`
- Linux/Windows: `-lOpenCL`
