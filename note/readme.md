# Note
開発時のメモなどを書き残すためのディレクトリです。

## 並列化機能の設計

### parallelize_design_v4.md
GPU並列化機能の最終設計ドキュメント。以下の機能を定義しています：

- **Kernel/CallKernel ASTノード**: GPU並列実行用のカーネル定義と呼び出し
- **ThreadIdDecl/KernelScope**: 型安全なスレッドID管理
- **3次元グリッド対応**: x, y, z軸すべてを利用した並列実行
- **OpenCL/CUDA/OpenMP対応**: 複数のバックエンドに対応したレンダリング

**実装状況**: コア型定義とASTノードは実装済み（v4設計に基づく）

**今後の実装予定**:
- Renderer対応（OpenCL/CUDA、OpenMP）
- KernelizeSuggester（FunctionからKernelへの変換）
- ParallelizeSuggester（ループの並列化）
- メモリアクセス競合解析