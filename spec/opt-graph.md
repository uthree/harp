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
| LoweringCostEstimator | Loweringフェーズ専用（ProgramRoot集約を促進） |
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
- **LoweringSuggester**: GraphOpをKernel(Function/Kernel)に変換
  - 各演算に対して複数の並列化戦略（`ParallelizationStrategy`）で候補を生成
  - ビームサーチのコスト評価により最適な戦略が選択される
- **BufferAbsorptionSuggester**: KernelのsrcにあるBufferを`input_buffers`に取り込む

#### ParallelizationStrategy

LoweringSuggesterが生成する並列化戦略：

| 戦略 | 説明 | 生成されるAST |
|------|------|---------------|
| Sequential | 逐次実行（CPU向け） | `AstNode::Function` + Rangeループ |
| FlatParallel | 全要素を線形インデックスで並列処理 | `AstNode::Kernel` (1D grid) |
| MultiDimParallel(n) | n次元までをスレッドIDで並列化 | `AstNode::Kernel` (nD grid) |

対応演算:
- **Elementwise/FusedElementwise**: 全戦略対応
- **Reduce**: Sequential, FlatParallel対応
- **その他**: Sequentialのみ

### ProgramRoot関連
- **ProgramRootAbsorptionSuggester**: Kernel(Function)をProgramRootに吸収
- **ProgramRootBufferAbsorptionSuggester**: ProgramRootのsrcから入力Bufferを除去

### マージ・統合系
- **KernelMergeSuggester**: 依存関係のあるKernelをペアワイズでマージ
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

| フラグ | kernel_merge | 説明 |
|--------|--------------|------|
| `new()` | false | 基本最適化のみ |
| `single_stage()` | true | 単一ステージ最適化（推奨） |

### 単一ステージ最適化（推奨）

`SuggesterFlags::single_stage()`でKernelMergeSuggesterを含めた最適化を実行。

```rust
let flags = SuggesterFlags::single_stage();
let (graph, history) = optimize_graph_with_history(
    graph, flags, SimpleCostEstimator::new(), 8, 200, true
);
```

### グラフ最適化とAST最適化の分離

グラフ最適化はグラフ構造の変換（融合、View挿入、Lowering）を担当し、
AST最適化（ループ変換、代数的簡約など）はLowering完了後に独立して実行される。
この分離により、各最適化フェーズが独立して改善可能な設計となっている。

## BeamSearchGraphOptimizer

- `beam_width`: ビーム幅（デフォルト: 5）
- `max_steps`: 最大探索ステップ（デフォルト: 10）
- `enable_early_termination`: コスト改善がない場合に早期終了（デフォルト: true）
- `selector`: 候補選択器（デフォルト: `StaticCostSelector`）

詳細は`src/opt/graph/`を参照。

## Selector

候補選択処理を抽象化するトレイト。ビームサーチにおいて、コスト付き候補から上位n件を選択する処理をカスタマイズ可能にする。

### 設計意図

tinygradのような多段階評価を実現するための抽象化：
1. 静的評価で明らかに悪い候補を足切り
2. 中間的なヒューリスティクスで絞り込み
3. 実行時間の実測値で精密に評価

### 実装

| 選択器 | 説明 |
|--------|------|
| StaticCostSelector | コストでソートして上位n件を選択（デフォルト） |
| MultiStageSelector | メソッドチェーンでn段階の選択パイプラインを構築 |

### 使用例

```rust
// デフォルト（静的コスト選択）
let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator);

// 多段階選択（dagoptスタイル）
let selector = MultiStageSelector::new()
    .then(|c| static_cost(c), 1000)   // 静的コストで1000件に足切り
    .then(|c| memory_cost(c), 100)    // メモリコストで100件に絞り込み
    .then(|c| measure_runtime(c), 10); // 実測で10件を最終選択

let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
    .with_selector(selector);
```

詳細は`src/opt/selector.rs`を参照。設計は[dagopt](https://github.com/uthree/dagopt)を参考にしている。
