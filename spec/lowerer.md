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
  │  - ContiguousInsertionSuggester
  │  - FusionSuggester
  ↓ Phase 2 (Lowering)
  │  - LoweringSuggester: GraphOp → Kernel(Function/Kernel)
  │    ※各演算に対して複数の並列化戦略で候補を生成
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
- `subgraph_lowering.rs`: `SubgraphLoweringOptimizer`（サブグラフの個別カーネル化）
- `utils.rs`: ユーティリティ関数（シグネチャ生成）

## サブグラフの処理

### SubgraphLoweringOptimizer
サブグラフを個別のカーネル関数として生成するOptimizer。`GraphOptimizer`トレイトを実装。

### サブグラフ処理モード（SubgraphMode）
パイプライン設定で選択可能な3つのモード：

1. **Inline（デフォルト）**: サブグラフをインライン展開
   - SubgraphInliningSuggesterを使用
   - 単一の大きなカーネルを生成
   - カーネル呼び出しオーバーヘッドなし

2. **SeparateKernels**: サブグラフを個別カーネルとして生成
   - SubgraphLoweringOptimizerを使用
   - 各サブグラフが独立した`__kernel`関数になる
   - `execution_order`で呼び出し順序を管理
   - コードの再利用性が高い

3. **Skip**: サブグラフ処理をスキップ
   - デバッグ用
   - SubgraphCallノードがそのまま残る（警告出力）

### CLIオプション
```bash
harpc --subgraph-mode inline input.harp   # デフォルト: インライン展開
harpc --subgraph-mode separate input.harp # 個別カーネル生成
harpc --subgraph-mode skip input.harp     # スキップ（デバッグ用）
```

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
- Fold演算（col2im）: Sequential/FlatParallel対応、groups > 1対応

### 未実装
- FusedReduce演算（タプル出力が必要）

## 並列化・SIMD化サポート

LoweringSuggesterは各演算に対して複数の並列化戦略で候補を生成し、ビームサーチが最適な戦略を選択します。

**SIMD化はAST最適化フェーズで実行:**
LoweringSuggesterはスカラー版のコードのみを生成します。SIMD（ベクトル）化は`VectorizationSuggester`（AST最適化）で行われます。ループ展開後に連続メモリアクセスパターンを検出し、ベクトル命令に変換します。詳細は`spec/opt-ast.md`のVectorizationSuggesterセクションを参照。

### 戦略一覧
- **Sequential**: 逐次実行（CPU向け、Rangeループ使用）
- **FlatParallel**: 1次元グリッドで全要素を並列処理
- **MultiDimParallel(n)**: n次元グリッドで並列処理（最大3次元）

### 対応演算
- Elementwise, FusedElementwise: 全戦略対応
- Reduce: Sequential, FlatParallel対応
- Fold: Sequential, FlatParallel対応（groups対応済み）
- その他: Sequentialのみ（順次拡張予定）
