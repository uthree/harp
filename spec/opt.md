# 最適化

## ファイル構成

### AST最適化
- `src/opt/mod.rs` - 最適化モジュールの定義
- `src/opt/ast/mod.rs` - AST最適化の公開API（15行）
- `src/opt/ast/estimator.rs` - CostEstimator実装（158行）
- `src/opt/ast/optimizer.rs` - Optimizer実装（257行、RuleBaseOptimizer、BeamSearchOptimizer）
- `src/opt/ast/suggester.rs` - Suggester実装（314行）
- `src/opt/ast/rules.rs` - 代数的書き換えルール集（899行、定数畳み込み含む）

### グラフ最適化
- `src/opt/graph/mod.rs` - グラフ最適化の公開API・トレイト定義
- `src/opt/graph/estimator.rs` - SimpleCostEstimator実装
- `src/opt/graph/optimizer.rs` - BeamSearchGraphOptimizer実装
- `src/opt/graph/suggesters/mod.rs` - Suggester関連の公開API
- `src/opt/graph/suggesters/composite.rs` - CompositeSuggester実装
- `src/opt/graph/suggesters/parallel.rs` - ParallelStrategyChanger実装
- `src/opt/graph/suggesters/view.rs` - ViewInsertionSuggester実装
- `src/opt/graph/suggesters/tiling.rs` - TilingSuggester実装（将来の拡張のためのスケルトン）
- `src/opt/graph/suggesters/fusion.rs` - FusionSuggester実装

## 実装状況

### グラフ最適化
`src/opt/graph/mod.rs`に実装済み。ビームサーチベースの最適化フレームワーク。

### AST最適化
`src/opt/ast/mod.rs`に実装済み。

## AST最適化

### トレイト

#### CostEstimator
ASTの実行コストを推定するトレイト。

```rust
pub trait CostEstimator {
    /// ASTノードのコストを推定
    fn estimate(&self, ast: &AstNode) -> f64;
}
```

#### Optimizer
ASTを最適化するトレイト。

```rust
pub trait Optimizer {
    /// ASTを最適化して返す
    fn optimize(&self, ast: AstNode) -> AstNode;
}
```

#### Suggester
複数の書き換え候補を提案するトレイト（ビームサーチ用）。

```rust
pub trait Suggester {
    /// 現在のASTから書き換え可能な候補をすべて提案
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode>;
}
```

### 実装

#### SimpleCostEstimator
ノード数ベースの簡単なコスト推定器。

**特徴:**
- 各ノードタイプに固定のコストを割り当て
- 子ノードのコストを再帰的に計算

**デフォルトコスト:**
- 基本演算（Add, Mul, Max）: 1.0
- 除算系（Rem, Idiv）: 2.0
- 単項演算（Recip, Sqrt）: 3.0～5.0
- メモリアクセス（Load, Store）: 5.0
- 定数・変数: 0.0
- ループ（Range）: 本体コスト × 10

**使用例:**
```rust
use harp::opt::ast::{CostEstimator, SimpleCostEstimator};

let estimator = SimpleCostEstimator::new();
let cost = estimator.estimate(&ast_node);
```

#### RuleBaseOptimizer
ルールベースの最適化器。`AstRewriter`をラップし、`Optimizer`トレイトを実装。

**特徴:**
- 複数の書き換えルールを適用
- 変化がなくなるまで繰り返し適用
- 最大反復回数を設定可能（デフォルト: 100）

**使用例:**
```rust
use harp::opt::ast::{Optimizer, RuleBaseOptimizer};
use harp::astpat;

// ルールを定義
let rule1 = astpat!(|a| {
    AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::Isize(0))))
} => {
    a  // a + 0 -> a
});

let rule2 = astpat!(|a| {
    AstNode::Mul(Box::new(a), Box::new(AstNode::Const(Literal::Isize(1))))
} => {
    a  // a * 1 -> a
});

// 最適化器を作成
let optimizer = RuleBaseOptimizer::new(vec![rule1, rule2])
    .with_max_iterations(50);

// 最適化を実行
let optimized = optimizer.optimize(ast);
```

#### BeamSearchOptimizer
ビームサーチアルゴリズムを使用した最適化器。`Suggester`と`CostEstimator`を組み合わせて、複数の候補から最良の書き換えを探索。

**特徴:**
- ビームサーチによる探索的最適化
- Cargoスタイルのプログレスバー表示（indicatif使用）
- ビーム幅と探索深さを設定可能
- デフォルト: ビーム幅10、最大深さ10

**プログレスバー表示例:**
```
  Optimizing [=========================================>                  ] 12/20 depth 12
```

**使用例:**
```rust
use harp::opt::ast::{
    Optimizer, BeamSearchOptimizer,
    RuleBaseSuggester, SimpleCostEstimator,
};
use harp::opt::ast::rules::all_algebraic_rules;

// SuggesterとEstimatorを作成
let suggester = RuleBaseSuggester::new(all_algebraic_rules());
let estimator = SimpleCostEstimator::new();

// ビームサーチ最適化器を作成
let optimizer = BeamSearchOptimizer::new(suggester, estimator)
    .with_beam_width(20)      // ビーム幅を設定
    .with_max_depth(15)        // 探索深さを設定
    .with_progress(true);      // プログレスバー表示を有効化

// 最適化を実行
let optimized = optimizer.optimize(ast);
```

**注意:**
- プログレスバーはデフォルトで有効です。テスト環境などで無効化したい場合は`.with_progress(false)`を使用してください。

#### RuleBaseSuggester
ルールベースの候補提案器。各ルールを全ての位置に適用して候補を生成。

**特徴:**
- 各ルールをASTの全ての位置に適用
- 重複する候補を除外
- ビームサーチなどの探索アルゴリズムで使用

**使用例:**
```rust
use harp::opt::ast::{Suggester, RuleBaseSuggester};

// ルールを定義
let rule = astpat!(|a, b| {
    AstNode::Add(Box::new(a), Box::new(b))
} => {
    AstNode::Add(Box::new(b), Box::new(a))  // 交換則
});

// 提案器を作成
let suggester = RuleBaseSuggester::new(vec![rule])
    .with_max_depth(10);

// 候補を生成
let candidates = suggester.suggest(&ast);

// コスト推定器と組み合わせて最良の候補を選択
let estimator = SimpleCostEstimator::new();
let best = candidates.iter()
    .min_by(|a, b| {
        estimator.estimate(a)
            .partial_cmp(&estimator.estimate(b))
            .unwrap()
    });
```

### 組み合わせ例：ビームサーチ

ビームサーチは`BeamSearchOptimizer`として実装されています（上記参照）。以下は基本的な使用例です：

```rust
use harp::opt::ast::{
    Optimizer, BeamSearchOptimizer,
    RuleBaseSuggester, SimpleCostEstimator,
};
use harp::opt::ast::rules::all_algebraic_rules;

// ルール、Suggester、Estimatorを準備
let rules = all_algebraic_rules();
let suggester = RuleBaseSuggester::new(rules);
let estimator = SimpleCostEstimator::new();

// ビームサーチで最適化
let optimizer = BeamSearchOptimizer::new(suggester, estimator)
    .with_beam_width(10)
    .with_max_depth(10);

let optimized_ast = optimizer.optimize(input_ast);
```

**カスタム実装の例:**

独自のビームサーチを実装する場合の例：

```rust
use harp::opt::ast::{Suggester, CostEstimator};

fn custom_beam_search<S: Suggester, E: CostEstimator>(
    initial_ast: AstNode,
    suggester: &S,
    estimator: &E,
    beam_width: usize,
    max_depth: usize,
) -> AstNode {
    let mut beam = vec![initial_ast];

    for _ in 0..max_depth {
        let mut candidates = Vec::new();

        // 現在のビーム内の各候補から新しい候補を生成
        for ast in &beam {
            let new_candidates = suggester.suggest(ast);
            candidates.extend(new_candidates);
        }

        if candidates.is_empty() {
            break;
        }

        // コストでソートして上位beam_width個を残す
        candidates.sort_by(|a, b| {
            estimator.estimate(a)
                .partial_cmp(&estimator.estimate(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        beam = candidates.into_iter().take(beam_width).collect();
    }

    // 最良の候補を返す
    beam.into_iter()
        .min_by(|a, b| {
            estimator.estimate(a)
                .partial_cmp(&estimator.estimate(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
}
```

## 代数的書き換えルール集

`src/opt/ast/rules.rs`には、ASTノードに対する標準的な代数的変形ルールが実装されています。

### 提供されるルール

#### 単位元ルール (Identity Rules)
- `add_zero_right()`: a + 0 = a
- `add_zero_left()`: 0 + a = a
- `mul_one_right()`: a * 1 = a
- `mul_one_left()`: 1 * a = a

#### 零元ルール (Zero Rules)
- `mul_zero_right()`: a * 0 = 0
- `mul_zero_left()`: 0 * a = 0

#### 冪等則 (Idempotent Rules)
- `max_idempotent()`: max(a, a) = a

#### 逆演算ルール (Inverse Operation Rules)
- `recip_recip()`: recip(recip(a)) = a
- `sqrt_squared()`: sqrt(a) * sqrt(a) = a

#### 交換則 (Commutative Rules)
- `add_commutative()`: a + b = b + a
- `mul_commutative()`: a * b = b * a
- `max_commutative()`: max(a, b) = max(b, a)

#### 結合則 (Associative Rules)
- `add_associate_left_to_right()`: (a + b) + c = a + (b + c)
- `add_associate_right_to_left()`: a + (b + c) = (a + b) + c
- `mul_associate_left_to_right()`: (a * b) * c = a * (b * c)
- `mul_associate_right_to_left()`: a * (b * c) = (a * b) * c

#### 分配則 (Distributive Rules)
- `distributive_left()`: a * (b + c) = a * b + a * c
- `distributive_right()`: (a + b) * c = a * c + b * c
- `factor_left()`: a * b + a * c = a * (b + c)（因数分解）
- `factor_right()`: a * c + b * c = (a + b) * c（因数分解）

#### 定数畳み込み (Constant Folding)
- `const_fold_add()`: Const(a) + Const(b) = Const(a + b)
- `const_fold_mul()`: Const(a) * Const(b) = Const(a * b)
- `const_fold_max()`: max(Const(a), Const(b)) = Const(max(a, b))
- `const_fold_rem()`: Const(a) % Const(b) = Const(a % b)
- `const_fold_idiv()`: Const(a) / Const(b) = Const(a / b)
- `const_fold_recip()`: recip(Const(a)) = Const(1 / a)
- `const_fold_sqrt()`: sqrt(Const(a)) = Const(sqrt(a))
- `const_fold_log2()`: log2(Const(a)) = Const(log2(a))
- `const_fold_exp2()`: exp2(Const(a)) = Const(2^a)
- `const_fold_sin()`: sin(Const(a)) = Const(sin(a))

### ルール集の生成関数

#### constant_folding_rules()
定数畳み込みルール集。コンパイル時に計算可能な定数式を評価する。

**使用例:**
```rust
use harp::opt::ast::{RuleBaseOptimizer, Optimizer};
use harp::opt::ast::rules::constant_folding_rules;

let optimizer = RuleBaseOptimizer::new(constant_folding_rules());

// (2 + 3) * 4 を最適化
let input = AstNode::Mul(
    Box::new(AstNode::Add(
        Box::new(AstNode::Const(Literal::Isize(2))),
        Box::new(AstNode::Const(Literal::Isize(3))),
    )),
    Box::new(AstNode::Const(Literal::Isize(4))),
);

let result = optimizer.optimize(input);
// 結果: 20 (5 * 4 = 20)
```

#### simplification_rules()
式を簡単にするルール集。単位元、零元、冪等則、逆演算のルールを含む。

**使用例:**
```rust
use harp::opt::ast::{RuleBaseOptimizer, Optimizer};
use harp::opt::ast::rules::simplification_rules;

let optimizer = RuleBaseOptimizer::new(simplification_rules());

// (42 + 0) * 1 を最適化
let input = AstNode::Mul(
    Box::new(AstNode::Add(
        Box::new(AstNode::Const(Literal::Isize(42))),
        Box::new(AstNode::Const(Literal::Isize(0))),
    )),
    Box::new(AstNode::Const(Literal::Isize(1))),
);

let result = optimizer.optimize(input);
// 結果: 42
```

#### normalization_rules()
式を標準形に変換するルール集。結合則を使って右結合に統一する。

#### all_algebraic_rules()
定数畳み込み、簡約、正規化ルールを含む全ての代数的ルール。

**使用例:**
```rust
use harp::opt::ast::{RuleBaseOptimizer, Optimizer};
use harp::opt::ast::rules::all_algebraic_rules;

let optimizer = RuleBaseOptimizer::new(all_algebraic_rules());

// ((2 + 3) * 1) + 0 を最適化
let input = AstNode::Add(
    Box::new(AstNode::Mul(
        Box::new(AstNode::Add(
            Box::new(AstNode::Const(Literal::Isize(2))),
            Box::new(AstNode::Const(Literal::Isize(3))),
        )),
        Box::new(AstNode::Const(Literal::Isize(1))),
    )),
    Box::new(AstNode::Const(Literal::Isize(0))),
);

let result = optimizer.optimize(input);
// 結果: 5 (定数畳み込みと簡約の組み合わせ)
```

**注意:** 交換則はビームサーチなどの探索で使用することを想定しているため、`all_algebraic_rules()`には含まれていません（無限ループの可能性があるため）。

### カスタムルール集の作成

個別のルール関数を組み合わせて、カスタムなルール集を作成できます：

```rust
use harp::opt::ast::rules::*;

// 簡約のみのルール集
let my_rules = vec![
    add_zero_right(),
    mul_one_right(),
    mul_zero_right(),
];

// 定数畳み込みのみ
let const_fold_only = vec![
    const_fold_add(),
    const_fold_mul(),
    const_fold_sqrt(),
];

// 分配則を使った展開
let expand_rules = vec![
    distributive_left(),
    distributive_right(),
];

// 因数分解
let factor_rules = vec![
    factor_left(),
    factor_right(),
];
```

### 注記
AST最適化の基礎機能として`src/ast/pat.rs`にAstRewriteRuleとAstRewriterが実装されており、`opt`モジュールはこれらをラップして体系的な最適化フレームワークを提供しています。

## グラフ最適化

グラフ最適化は計算グラフ（Graph）を対象とした最適化で、並列化戦略の変更、メモリレイアウトの最適化、ノード融合などを行います。AST最適化と同じパラダイム（Optimizer、Suggester、CostEstimator）を採用しています。

### トレイト

#### GraphCostEstimator
グラフの実行コストを推定するトレイト。

```rust
pub trait GraphCostEstimator {
    /// グラフの実行コストを推定
    fn estimate(&self, graph: &Graph) -> f32;
}
```

#### GraphOptimizer
グラフを最適化するトレイト。

```rust
pub trait GraphOptimizer {
    /// グラフを最適化して返す
    fn optimize(&self, graph: Graph) -> Graph;
}
```

#### GraphSuggester
複数の書き換え候補を提案するトレイト（ビームサーチ用）。

```rust
pub trait GraphSuggester {
    /// 現在のグラフから書き換え可能な候補をすべて提案
    fn suggest(&self, graph: &Graph) -> Vec<Graph>;
}
```

### 実装

#### SimpleCostEstimator
ノード数とメモリアクセスベースの簡単なコスト推定器。

**特徴:**
- 各ノードに固定コストを割り当て
- メモリアクセス量を推定（dtype × 要素数）
- 並列化戦略を考慮（Sequential < Thread < ThreadGroup）
- カーネル起動オーバーヘッドを含む

**使用例:**
```rust
use harp::opt::graph::{GraphCostEstimator, SimpleCostEstimator};

let estimator = SimpleCostEstimator::new();
let cost = estimator.estimate(&graph);
```

#### BeamSearchGraphOptimizer
ビームサーチアルゴリズムを使用したグラフ最適化器。AST版と同じインターフェース。

**特徴:**
- ビームサーチによる探索的最適化
- Cargoスタイルのプログレスバー表示（indicatif使用）
- ビーム幅と探索深さを設定可能
- デフォルト: ビーム幅10、最大深さ10

**使用例:**
```rust
use harp::opt::graph::{
    BeamSearchGraphOptimizer, SimpleCostEstimator,
    ParallelStrategyChanger,
};

let suggester = ParallelStrategyChanger::with_default_strategies();
let estimator = SimpleCostEstimator::new();

let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
    .with_beam_width(20)
    .with_max_depth(15)
    .with_progress(true);

let optimized_graph = optimizer.optimize(graph);
```

#### ParallelStrategyChanger
並列化戦略を変更するSuggester。

**特徴:**
- 各ノードの各軸について、異なる並列化戦略を試す
- Sequential → Thread、Thread → ThreadGroup などの変更
- SIMD幅とアンローリング係数の変更
- 探索空間の爆発を防ぐため、一度に1つのノードの1つの軸のみを変更

**使用例:**
```rust
use harp::opt::graph::ParallelStrategyChanger;
use harp::graph::ElementwiseStrategy;

// カスタム戦略候補
let suggester = ParallelStrategyChanger::new(vec![
    ElementwiseStrategy::sequential(),
    ElementwiseStrategy::thread(),
    ElementwiseStrategy::thread_simd(4),
]);

// デフォルト戦略候補
let suggester = ParallelStrategyChanger::with_default_strategies();

let suggestions = suggester.suggest(&graph);
```

#### ViewInsertionSuggester
View変更ノード（転置など）を挿入するSuggester。

**特徴:**
- ノード間にView変更（permute）を挿入
- その後にContiguousノードを挿入してメモリレイアウトを実体化
- 主にメモリアクセスパターンの改善を目的とする

**使用例:**
```rust
use harp::opt::graph::ViewInsertionSuggester;

let suggester = ViewInsertionSuggester::new()
    .with_transpose(true);

let suggestions = suggester.suggest(&graph);
```

#### TilingSuggester
ループのタイル化に相当するView変更を提案するSuggester（将来の拡張）。

**注意:**
現在のViewはreshape操作をサポートしていないため、完全なタイル化は未実装です。この実装は将来の拡張のためのスケルトンとなっています。

将来的な実装方針:
1. shape [N, M] を [N/tile, tile, M/tile, tile] に変換（reshape）
2. permuteで [N/tile, M/tile, tile, tile] に並べ替え
3. これにより内側ループがタイルサイズになり、時間的局所性が向上

**使用例:**
```rust
use harp::opt::graph::TilingSuggester;

let suggester = TilingSuggester::with_default_tile_sizes();
// 現在は候補を生成しない（reshape未実装のため）
let suggestions = suggester.suggest(&graph);
```

#### FusionSuggester
ノード融合を提案するSuggester。

**特徴:**
- elementwise演算の連鎖を検出してFusedElementwiseに変換
- elementwise → reduceパターンをFusedElementwiseReduceに変換
- 中間バッファを削減し、メモリアクセスを削減

**使用例:**
```rust
use harp::opt::graph::FusionSuggester;

let suggester = FusionSuggester::new();
let suggestions = suggester.suggest(&graph);
```

#### CompositeSuggester
複数のSuggesterを組み合わせるSuggester。

**特徴:**
- 低レベルなSuggesterを組み合わせて使用
- 各Suggesterから候補を収集して統合

**使用例:**
```rust
use harp::opt::graph::{
    CompositeSuggester, ParallelStrategyChanger,
    ViewInsertionSuggester, FusionSuggester,
};

let composite = CompositeSuggester::new(vec![
    Box::new(ParallelStrategyChanger::with_default_strategies()),
    Box::new(ViewInsertionSuggester::new()),
    Box::new(FusionSuggester::new()),
]);

let suggestions = composite.suggest(&graph);
```

### 組み合わせ例：グラフ最適化パイプライン

```rust
use harp::opt::graph::*;
use harp::graph::Graph;

// グラフを作成
let mut graph = Graph::new();
// ... グラフ構築 ...

// 複数のSuggesterを組み合わせ
let suggester = CompositeSuggester::new(vec![
    Box::new(ParallelStrategyChanger::with_default_strategies()),
    Box::new(ViewInsertionSuggester::new()),
    Box::new(FusionSuggester::new()),
]);

let estimator = SimpleCostEstimator::new();

// ビームサーチで最適化
let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
    .with_beam_width(20)
    .with_max_depth(10)
    .with_progress(true);

let optimized_graph = optimizer.optimize(graph);
```
