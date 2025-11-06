# 最適化

## ファイル構成
- `src/opt/mod.rs` - 最適化モジュールの定義
- `src/opt/ast/mod.rs` - AST最適化の公開API（14行）
- `src/opt/ast/estimator.rs` - CostEstimator実装（235行）
- `src/opt/ast/optimizer.rs` - Optimizer実装（66行）
- `src/opt/ast/suggester.rs` - Suggester実装（314行）
- `src/opt/graph/mod.rs` - グラフ最適化（未実装）

## 実装状況

### グラフ最適化
未実装。将来的に実装予定。

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
- カスタムコストの設定が可能

**デフォルトコスト:**
- 基本演算（Add, Mul, Max）: 1.0
- 除算系（Rem, Idiv）: 2.0
- 単項演算（Recip, Sqrt）: 3.0～5.0
- メモリアクセス（Load, Store）: 5.0
- 定数・変数: 0.0
- ループ（Range）: 本体コスト × 10

**使用例:**
```rust
use harp::opt::ast::{CostEstimator, SimpleCostEstimator, NodeType};

let estimator = SimpleCostEstimator::new()
    .with_cost(NodeType::Add, 1.5)  // カスタムコストを設定
    .with_cost(NodeType::Mul, 2.0);

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

```rust
use harp::opt::ast::{Suggester, CostEstimator, RuleBaseSuggester, SimpleCostEstimator};

fn beam_search(
    initial_ast: AstNode,
    suggester: &RuleBaseSuggester,
    estimator: &SimpleCostEstimator,
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
                .unwrap()
        });

        beam = candidates.into_iter().take(beam_width).collect();
    }

    // 最良の候補を返す
    beam.into_iter()
        .min_by(|a, b| {
            estimator.estimate(a)
                .partial_cmp(&estimator.estimate(b))
                .unwrap()
        })
        .unwrap()
}
```

### 注記
AST最適化の基礎機能として`src/ast/pat.rs`にAstRewriteRuleとAstRewriterが実装されており、`opt`モジュールはこれらをラップして体系的な最適化フレームワークを提供しています。
