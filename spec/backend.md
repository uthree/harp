# バックエンド

バックエンドはAST→実行可能コードへの変換と実行を担当します。

## モジュール構成

バックエンド関連のコードは以下のように構成されています：

### コアモジュール
- `mod.rs`: Renderer、Compiler、Kernel、Buffer、Device等の基本trait定義
- `device.rs`: デバイス抽象化（CPU、GPU等の実行環境）
- `pipeline.rs`: Pipeline trait（グラフからカーネルまでの処理フロー）
- `generic.rs`: GenericPipeline（任意のRendererとCompilerを組み合わせた汎用実装）
- `c_like.rs`: C言語系構文の共通レンダリングロジック（CLikeRenderer trait）

### バックエンド実装
各バックエンドは独立したモジュールとして実装されています：

- `c/`: Cバックエンド（シングルスレッド）
  - `mod.rs`, `renderer.rs`, `compiler.rs`, `kernel.rs`, `buffer.rs`
- `metal/`: Metalバックエンド（macOS/iOS GPU）
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
1. **グラフ最適化** (必須): LoweringSuggesterでGraphOpをCustomノードに変換、融合、並列化戦略変更など
2. **Lowering**: Graph → AST変換（Customノードの展開）
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
- グラフ最適化（必須）とAST最適化（オプション）

**設定フィールド:**
- `enable_ast_optimization`: AST最適化の有効化
- `graph_config`: グラフ最適化の設定（ビーム幅、最大ステップ数、プログレス表示）
- `ast_config`: AST最適化の設定
- `collect_histories`: 最適化履歴を収集するか（DEBUGビルドでデフォルトtrue、RELEASEビルドでfalse）

**使用例:**
```rust
let mut pipeline = GenericPipeline::new(renderer, compiler);
// グラフ最適化は常に有効（LoweringSuggesterが必須）
pipeline.enable_ast_optimization = true;
pipeline.graph_config.beam_width = 8;
```

**最適化履歴:**
最適化の各ステップは`OptimizationHistories`に記録され、`pipeline.histories.graph`および`pipeline.histories.ast`からアクセス可能。可視化ツール（harp-viz）で最適化過程を確認できる。

## 実装状況

### Metal Backend（macOS/iOS）
MetalのComputePipelineStateを使用してGPUで直接実行。

### C Backend（クロスプラットフォーム、シングルスレッド）
C言語（シングルスレッド）を使ったリファレンス実装バックエンド。ASTをC言語コードに変換し、動的ライブラリ(.so/.dylib/.dll)としてコンパイルして実行。並列化は行わず、単一スレッドでの実行のみをサポート。

**libloading対応:**
libloadingは固定シグネチャ `fn(*mut *mut u8)` を期待するため、CRendererは自動的にラッパー関数を生成する：
```c
// エントリーポイント関数
void harp_main(const float* input0, float* output) { ... }

// libloading用ラッパー（自動生成）
void __harp_entry(void** buffers) {
    harp_main((const float*)(buffers[0]), (float*)(buffers[1]));
}
```

**注意事項:**
- エントリーポイント関数名は "harp_main"（C言語のmain関数との衝突回避）
- Lowererがパラメータを依存関係順序で配置するため、バッファ順序に注意
- 並列化はサポートしない（単一スレッドでの実行のみ）

### OpenCL Backend（クロスプラットフォーム、並列実行）
OpenCLを使ったGPU/並列実行バックエンド。ASTをOpenCLカーネルソース + ホストコードに変換し、動的ライブラリとしてコンパイルして実行。

**実装方式:**
- OpenCLカーネルソースを文字列リテラルとしてC言語コードに埋め込み
- ホストコードでOpenCLの初期化、コンパイル、実行を行う
- libloadingでラッパー関数を呼び出す

**コンパイラフラグ:**
- macOS: `-framework OpenCL`
- Linux/Windows: `-lOpenCL`

**現在の実装状況:**
- 基本的なOpenCL初期化とプログラムビルドのコード生成が実装済み
- カーネルボディとバッファ管理の実装は未完成（TODO）
