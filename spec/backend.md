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

処理フロー:
1. **グラフ最適化** (オプション): 並列化戦略変更、View挿入、ノード融合など
2. **Lowering**: Graph → AST変換
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
- グラフ最適化とAST最適化の個別制御

**設定フィールド:**
- `enable_graph_optimization`: グラフ最適化の有効化
- `enable_ast_optimization`: AST最適化の有効化
- `graph_config`: グラフ最適化の設定（ビーム幅、最大ステップ数、プログレス表示）
- `ast_config`: AST最適化の設定
- `collect_histories`: 最適化履歴を収集するか（DEBUGビルドでデフォルトtrue、RELEASEビルドでfalse）

**使用例:**
```rust
let mut pipeline = GenericPipeline::new(renderer, compiler);
pipeline.enable_graph_optimization = true;
pipeline.enable_ast_optimization = true;
pipeline.graph_config.beam_width = 8;
```

**最適化履歴:**
最適化の各ステップは`OptimizationHistories`に記録され、`pipeline.histories.graph`および`pipeline.histories.ast`からアクセス可能。可視化ツール（harp-viz）で最適化過程を確認できる。

## 実装状況

### Metal Backend（macOS/iOS）
MetalのComputePipelineStateを使用してGPUで直接実行。

### OpenMP Backend（クロスプラットフォーム）
C言語とOpenMPを使ったCPUバックエンド。ASTをC言語コードに変換し、OpenMPの`#pragma omp parallel for`でカーネル関数を並列実行することでGPU並列実行を擬似的に再現。動的ライブラリ(.so/.dylib/.dll)としてコンパイルして実行。

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

**ビルドオプション:**
- `CCompiler::with_openmp(bool)`: OpenMPフラグの有効/無効
- `CRenderer::with_openmp_header(bool)`: omp.hヘッダーの有効/無効（macOS互換性）

**注意事項:**
- エントリーポイント関数名は "harp_main"（C言語のmain関数との衝突回避）
- Lowererがパラメータを依存関係順序で配置するため、バッファ順序に注意
