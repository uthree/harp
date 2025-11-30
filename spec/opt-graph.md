# Graph最適化

ビームサーチベースの最適化フレームワーク。

## トレイト

- **GraphCostEstimator**: Graphの実行コスト推定
- **GraphOptimizer**: Graphの最適化
- **GraphSuggester**: 書き換え候補を提案（ビームサーチ用）

## コスト推定器

| 推定器 | 用途 |
|--------|------|
| SimpleCostEstimator | ノード数とメモリアクセスベースの簡易推定 |
| AstBasedCostEstimator | ASTに変換して高精度推定（推奨） |
| KernelMergeCostEstimator | カーネルマージ最適化専用 |

SimpleCostEstimatorは複数のKernel(Function)にペナルティを付与し、単一のKernel(Program)への収束を促進する。

## Suggester一覧

### 最適化系
- **FusionSuggester**: 連続するElementwise演算を融合
- **ViewMergeSuggester**: Viewノードを上流ノードにマージ
- **ViewInsertionSuggester**: メモリレイアウト最適化のためのView挿入
- **ContiguousInsertionSuggester**: 非contiguousなViewを実体化
- **TilingSuggester**: ループタイリング適用

### Lowering系
- **LoweringSuggester**: GraphOpをKernel(Function)に変換
- **BufferAbsorptionSuggester**: KernelのsrcにあるBufferを`input_buffers`に取り込む

### ProgramRoot関連
- **ProgramRootAbsorptionSuggester**: Kernel(Function)をProgramRootに吸収
- **ProgramRootBufferAbsorptionSuggester**: ProgramRootのsrcから入力Bufferを除去

### マージ・統合系
- **KernelMergeSuggester**: 依存関係のあるKernelをペアワイズでマージ
- **AstOptimizationSuggester**: KernelのASTにAST最適化を適用
- **CompositeSuggester**: 複数Suggesterを組み合わせ

## Suggesterの連携

```
ViewMergeSuggester → ProgramRootAbsorptionSuggester
```
ProgramRootAbsorptionSuggesterはViewを透過的に扱わないため、ViewMergeSuggesterが先に適用される必要がある。

## 最適化フロー

```
FusionSuggester        : Elementwise演算の融合
       ↓
LoweringSuggester      : GraphOp → Kernel(Function)
       ↓
BufferAbsorptionSuggester : 入力Bufferの取り込み
       ↓
ProgramRootAbsorptionSuggester : Kernel(Function) → ProgramRoot(Program)
       ↓
ProgramRootBufferAbsorptionSuggester : 入力Bufferの除去
```

## 最適化モード

### SuggesterFlags

| フラグ | kernel_merge | ast_optimization | 説明 |
|--------|--------------|------------------|------|
| `new()` | false | false | 基本最適化のみ |
| `single_stage()` | true | false | 単一ステージ最適化 |
| `unified()` | true | true | 統合最適化（推奨） |

### 統合最適化（推奨）

`SuggesterFlags::unified()`でグラフ最適化とAST最適化を単一のビームサーチで探索。

```rust
let flags = SuggesterFlags::unified();
let (graph, history) = optimize_graph_with_history(
    graph, flags, SimpleCostEstimator::new(), 8, 200, true
);
```

## BeamSearchGraphOptimizer

- `beam_width`: ビーム幅（デフォルト: 5）
- `max_steps`: 最大探索ステップ（デフォルト: 10）
- `enable_early_termination`: コスト改善がない場合に早期終了（デフォルト: true）

詳細は`src/opt/graph/`を参照。
