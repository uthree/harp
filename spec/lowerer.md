# Lowerer

## 役割
計算グラフをASTに変換する層。マルチフェーズ最適化によりグラフを単一のProgramに収束させ、ASTとして出力します。

## アーキテクチャ

**グラフ融合によるProgram収束**:
`lower()`関数はマルチフェーズ最適化を実行し、グラフ全体を単一の`AstNode::Program`に収束させます。

```
Graph
  ↓ Phase 1 (Preparation)
  │  - ViewInsertionSuggester
  │  - ViewMergeSuggester
  │  - TilingSuggester
  │  - ContiguousInsertionSuggester
  │  - FusionSuggester
  ↓ Phase 2 (Lowering)
  │  - LoweringSuggester: GraphOp → Kernel(Function)
  │  - ViewMergeSuggester: ViewをKernelに吸収
  │  - BufferAbsorptionSuggester: 入力Buffer取り込み
  │  - ProgramRootAbsorptionSuggester: Kernel → ProgramRoot
  │  - ProgramRootBufferAbsorptionSuggester: 入力Buffer除去
  │  - KernelMergeSuggester: 複数Kernelをマージ
  ↓
AstNode::Program (単一ノードに収束)
```

## 処理フロー

1. **グラフ最適化**: マルチフェーズ最適化で単一Programに収束
2. **Program取得**: `ProgramRoot`ノードまたは`Kernel(Program)`ノードからProgramを取得
3. **エラーハンドリング**: 収束しなかった場合はパニック（未対応ノードタイプの可能性）

## ファイル構成
- `mod.rs`: `lower()`関数（マルチフェーズ最適化実行、Program取得）
- `utils.rs`: ユーティリティ関数（シグネチャ生成）

## Lowerer構造体

`Lowerer`構造体は`create_signature()`メソッドのみを提供します：
- `Lowerer::create_signature(graph)`: グラフから`KernelSignature`を生成

## 設計上の決定

### グラフ融合への一本化
以前はトポロジカルソートによる世代別分割をフォールバックとして持っていましたが、現在はマルチフェーズ最適化による単一Program収束に一本化されています。

**理由**:
- 世代別分割はコードの複雑さを増す
- マルチフェーズ最適化は単一ノードに収束するまで繰り返すため、フォールバックは不要
- 並列実行はAST内部のBarrierで表現可能

### 空のProgramの許可
入力をそのまま出力するだけのグラフ（計算なし）では、`Program`のfunctionsリストが空になります。これは正常な状態として許可されています。

## 実装状況

### 実装済み
- マルチフェーズ最適化による単一Program収束
- ProgramRoot/Kernel(Program)からのProgram取得
- 動的shape対応のシグネチャ生成

### 未実装
- FusedReduce演算（タプル出力が必要）
