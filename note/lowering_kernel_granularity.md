# Lowering時のカーネル粒度の問題

## 現状の実装

### 問題点

現在の`lowerer/mod.rs`の`lower()`関数では、**各GraphNodeを個別のカーネル関数に変換**している：

```rust
// 各世代の各ノードをカーネル関数に変換
let mut kernel_id = 0;
for generation in generations {
    for node in generation {
        if matches!(node.op, GraphOp::Input) {
            continue;
        }
        // 1ノード = 1カーネル
        if let Ok(function) = lowerer.lower_node_to_kernel(&node, kernel_id) {
            let kernel_name = format!("kernel_{}", kernel_id);
            program.add_function(kernel_name, function);
            kernel_id += 1;
        }
    }
}
```

**これは非効率的:**
1. **カーネル起動オーバーヘッド**: 各ノードごとにGPUカーネルを起動するコストが大きい
2. **中間結果のメモリ転送**: 各カーネルの出力を毎回グローバルメモリに書き戻し、次のカーネルが読み込む必要がある
3. **レジスタ活用の機会損失**: 複数の演算を融合すれば、中間結果をレジスタに保持できる

### 例

```
a + b * c
```

という単純な計算でも：
- `kernel_0`: b * c → temp1 （メモリ書き込み）
- `kernel_1`: a + temp1 → result （メモリ読み込み + 書き込み）

となり、2回のカーネル起動と不要なメモリアクセスが発生する。

## 理想的なアプローチ

### オプション1: グラフ全体を1つのカーネルに融合

**メリット:**
- カーネル起動は1回のみ
- 中間結果は全てレジスタで保持
- 最も効率的

**デメリット:**
- 大きなグラフでレジスタ不足の可能性
- 複雑な制御フローの実装が必要
- デバッグが困難

**実装方針:**
- `RecursiveLowerer`を活用
- 出力ノードから再帰的に依存を辿り、1つの大きなカーネル関数を生成
- `recursive.rs`が既に存在するが、現在未使用

### オプション2: 融合可能なサブグラフをカーネル化

**メリット:**
- カーネルサイズのバランスが取れる
- レジスタ使用量を制御しやすい
- 部分的に融合できない演算（Reduce等）に対応可能

**デメリット:**
- 融合境界の判定ロジックが複雑
- 一部カーネル起動オーバーヘッドは残る

**実装方針:**
- グラフ最適化フェーズで融合可能なノードをマージ（`FusedElementwise`等）
- 融合後のノードのみをカーネル化
- 融合前の個別ノードは削除

**現在の中途半端な状態:**
- `FusedElementwise`, `FusedElementwiseReduce`は実装済み
- しかし、融合前のノードも個別にカーネル化されている
- 融合の効果が半減している

### オプション3: 階層的な融合戦略

**戦略:**
1. **基本ブロック**: 依存関係のない連続したelementwise演算を融合
2. **複合演算**: elementwise + reduce、cumulative等のパターンを融合
3. **カーネル境界**: 以下で分割
   - 大きなメモリ転送が必要な箇所（transpose等）
   - 同期が必要な箇所（reduce全体等）
   - レジスタ圧が高くなる箇所

## 既存コードの整理

### 使用されているコード

**`lowerer/mod.rs`:**
- `lower()`: メインエントリーポイント
- `lower_node_to_kernel()`: 1ノード → 1カーネル変換
- `topological_sort()`: グラフのトポロジカルソート

**個別のlowerer:**
- `elementwise.rs`: Elementwise演算
- `reduce.rs`: Reduce演算
- `contiguous.rs`: Contiguous演算
- `fused_elementwise.rs`: 融合elementwise
- `fused_elementwise_reduce.rs`: 融合elementwise+reduce

### 未使用のコード

**`lowerer/recursive.rs`:**
- `RecursiveLowerer`: 再帰的なlowering実装
- `VarMapper`: 変数名マッピング管理
- メモ化機構

**このコードは元々グラフ全体を1つにlowerする想定で作られた可能性が高い**

## 実装の提案

### Phase 1: 融合ノードのみをカーネル化（短期的改善）

現在の実装を修正：

```rust
pub(crate) fn lower(graph: Graph) -> crate::ast::Program {
    // グラフ最適化で融合を適用（opt/graph/で実施済み）

    // 融合後のノードのみをカーネル化
    let generations = Lowerer::topological_sort(&graph);

    for generation in generations {
        for node in generation {
            // Inputと、融合により不要になったノードはスキップ
            if should_skip_node(&node) {
                continue;
            }

            // 融合済みノードまたは融合不可能なノードのみカーネル化
            lowerer.lower_node_to_kernel(&node, kernel_id);
        }
    }
}
```

### Phase 2: RecursiveLowererの活用（長期的改善）

グラフ全体または大きなサブグラフを1カーネルに：

```rust
pub(crate) fn lower(graph: Graph) -> crate::ast::Program {
    let mut lowerer = RecursiveLowerer::new();

    // 入力ノードの変数名を設定
    for (i, input) in graph.inputs().iter().enumerate() {
        lowerer.set_var_name(&input, format!("input_{}", i));
    }

    // 出力ノードから再帰的にlower
    for (i, output) in graph.outputs().iter().enumerate() {
        lowerer.set_var_name(&output, format!("output_{}", i));
        lowerer.lower_node(&output);
    }

    // 1つの大きなカーネル関数を生成
    let kernel = lowerer.build_kernel_function();

    program.add_function("main_kernel", kernel);
}
```

### Phase 3: 適応的融合戦略（最終形）

グラフの特性に応じて最適な粒度を選択：

```rust
// レジスタ圧、メモリ帯域、並列度などを考慮
let fusion_strategy = analyze_graph_characteristics(&graph);

match fusion_strategy {
    FusionStrategy::SingleKernel => {
        // グラフ全体を1カーネルに
    }
    FusionStrategy::Hierarchical(boundaries) => {
        // 適切な境界でカーネルを分割
    }
    FusionStrategy::PerNode => {
        // 現在の実装（デバッグ用）
    }
}
```

## 参考: 他のフレームワークの実装

### TVM/Relay
- 融合パスで大きなサブグラフをまとめる
- カーネル境界は手動アノテーションまたはヒューリスティック

### XLA (TensorFlow)
- HLO (High Level Operations) を融合
- 基本的に1つのHLO計算全体を1カーネルにする

### TinyGrad
- LazyBuffer で演算をバッファリング
- 実行時に融合可能な演算を1カーネルにまとめる

## アクションアイテム

- [ ] Phase 1の実装（融合ノードのみカーネル化）
- [ ] `RecursiveLowerer`の活用方法を検討
- [ ] カーネル分割のヒューリスティックを設計
- [ ] パフォーマンス測定（現在 vs 改善後）
- [ ] 仕様書の更新（`spec/lowerer.md`）

## 関連ファイル

- `src/lowerer/mod.rs`: メインのlowering実装
- `src/lowerer/recursive.rs`: 未使用の再帰的lowerer
- `src/opt/graph/suggesters/fusion.rs`: グラフレベルの融合
- `spec/lowerer.md`: Lowererの仕様書
- `note/lowerer_refactor_design.md`: 過去のリファクタリング設計

## メモ

この問題は実行効率に直結する重要な設計課題。特にGPUバックエンドでは、カーネル起動オーバーヘッドとメモリ転送のコストが支配的になるため、適切な融合戦略が不可欠。

現在の実装は「動作する」が「効率的ではない」状態。将来的には必ず改善が必要。
