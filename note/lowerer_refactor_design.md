# Lowererリファクタリング設計

## 目標

部分的なGraphノードのLoweringとASTコスト探索を可能にするLowererの再設計。

## 要件

### 1. 部分的なLowering
- 特定のGraphノードやサブグラフのみをASTに変換できる
- ノード単位、サブグラフ単位で独立してloweringできる

### 2. 複数戦略の探索
- 同じGraphノードに対して複数のLowering戦略を試せる
- 各戦略で生成されたASTのコストを評価
- 最適なASTを選択

### 3. 既存機能の維持
- 現在のGraph最適化（fusion等）は維持
- 既存のテストが通る互換性

## 現状分析

### 現在のLowererのフロー

```
Graph::outputs (Vec<GraphNode>)
    ↓
Lowerer::lower(graph: &Graph)
    ↓
topological_sort_by_generation() → Vec<Vec<GraphNode>>
    ↓
for each generation:
    for each node:
        lower_node(node) → Option<AstNode>
    insert Barrier
    ↓
create_kernel_function() → AstNode (Function)
    ↓
create_entry_function() → AstNode (Function)
    ↓
AstNode::Program
```

**課題**：
- グラフ全体が処理単位
- ノード単位のloweringはできるが、コンテキストが必要
- 変数名のマッピング（node_to_var）がLowererの状態として保持される

### GraphNodeの構造

```rust
pub struct GraphNode(Rc<GraphNodeData>);

pub struct GraphNodeData {
    pub op: GraphOp,           // 演算の種類
    pub dtype: DType,          // データ型
    pub view: View,            // メモリレイアウト
    pub strategy: Option<LoopStrategy>,  // 最適化戦略
}

pub struct LoopStrategy {
    pub vectorize: Option<(usize, usize)>,
    pub unroll: Option<(usize, usize)>,
    pub parallelize: Vec<usize>,
    pub tile: Vec<(usize, usize)>,
    pub use_shared_memory: bool,
}
```

## 設計案

### 案A: Partial Lowerer（推奨）

部分的なloweringを可能にする新しいLowerer設計。

#### 概念

```
Graph
    ↓
SubGraph (部分グラフの抽象化)
    ↓ PartialLowerer
PartialAST (部分的なAST + メタデータ)
    ↓ ASTコスト評価
コスト値
    ↓ 探索・選択
最適なPartialAST
    ↓ 結合
完全なAST
```

#### データ構造

```rust
/// 部分的なAST表現
/// 単一のGraphノードまたはサブグラフに対応
pub struct PartialAST {
    /// 生成されたASTステートメント
    pub statements: Vec<AstNode>,

    /// 必要な変数宣言
    pub declarations: Vec<VariableDecl>,

    /// 入力変数名（このPartialASTが依存する変数）
    pub input_vars: Vec<String>,

    /// 出力変数名（このPartialASTが生成する変数）
    pub output_var: String,

    /// 対応するGraphノード
    pub source_node: GraphNode,

    /// 使用されたLowering戦略
    pub strategy: Option<LoopStrategy>,

    /// 推定コスト
    pub estimated_cost: Option<usize>,
}

/// サブグラフの表現
/// 複数のGraphノードのまとまり
pub struct SubGraph {
    /// サブグラフに含まれるノード（トポロジカル順）
    pub nodes: Vec<GraphNode>,

    /// サブグラフへの入力（外部からの依存）
    pub inputs: Vec<GraphNode>,

    /// サブグラフからの出力
    pub outputs: Vec<GraphNode>,
}

impl SubGraph {
    /// 単一ノードからサブグラフを作成
    pub fn from_node(node: &GraphNode) -> Self {
        let inputs = node.input_nodes();
        SubGraph {
            nodes: vec![node.clone()],
            inputs,
            outputs: vec![node.clone()],
        }
    }

    /// 複数ノードをマージしてサブグラフを作成
    pub fn merge(subgraphs: Vec<SubGraph>) -> Self {
        // トポロジカルソートしてノードを統合
        // 入力・出力を再計算
        todo!()
    }
}
```

#### PartialLowerer

```rust
/// 部分的なLoweringを実行
pub struct PartialLowerer {
    /// 変数名のマッピング（既知のノード）
    node_to_var: HashMap<GraphNode, String>,

    /// 次の一時変数ID
    next_temp_id: usize,
}

impl PartialLowerer {
    pub fn new() -> Self {
        Self {
            node_to_var: HashMap::new(),
            next_temp_id: 0,
        }
    }

    /// 既知のノードに変数名を設定
    pub fn set_var_name(&mut self, node: &GraphNode, var_name: String) {
        self.node_to_var.insert(node.clone(), var_name);
    }

    /// サブグラフを部分的にlower
    pub fn lower_subgraph(&mut self, subgraph: &SubGraph) -> PartialAST {
        let mut statements = Vec::new();
        let mut declarations = Vec::new();
        let mut input_vars = Vec::new();

        // 入力ノードの変数名を収集
        for input in &subgraph.inputs {
            let var_name = self.get_or_create_var_name(input);
            input_vars.push(var_name);
        }

        // サブグラフ内の各ノードをlower
        for node in &subgraph.nodes {
            if let Some(stmt) = self.lower_node_internal(node, &mut declarations) {
                statements.push(stmt);
            }
        }

        // 出力変数名を取得（単一出力を想定）
        let output_var = self.get_or_create_var_name(&subgraph.outputs[0]);

        PartialAST {
            statements,
            declarations,
            input_vars,
            output_var,
            source_node: subgraph.outputs[0].clone(),
            strategy: subgraph.outputs[0].strategy.clone(),
            estimated_cost: None,
        }
    }

    /// 複数の戦略でloweringを試す
    pub fn lower_with_strategies(
        &mut self,
        node: &GraphNode,
        strategies: Vec<LoopStrategy>,
    ) -> Vec<PartialAST> {
        let mut results = Vec::new();

        for strategy in strategies {
            // 戦略を適用した新しいノードを作成
            let node_with_strategy = node.clone().with_strategy(strategy);

            // サブグラフを作成
            let subgraph = SubGraph::from_node(&node_with_strategy);

            // Lowering実行
            let partial_ast = self.lower_subgraph(&subgraph);

            results.push(partial_ast);
        }

        results
    }

    fn lower_node_internal(
        &mut self,
        node: &GraphNode,
        declarations: &mut Vec<VariableDecl>,
    ) -> Option<AstNode> {
        // 既存のlower_nodeロジックを流用
        todo!()
    }

    fn get_or_create_var_name(&mut self, node: &GraphNode) -> String {
        if let Some(name) = self.node_to_var.get(node) {
            name.clone()
        } else {
            let name = format!("temp{}", self.next_temp_id);
            self.next_temp_id += 1;
            self.node_to_var.insert(node.clone(), name.clone());
            name
        }
    }
}
```

#### ASTコスト評価

```rust
/// PartialASTのコストを推定
pub fn estimate_partial_ast_cost(partial_ast: &PartialAST) -> usize {
    use crate::opt::ast::cost_estimator;

    // 既存のAST cost_estimatorを使用
    let mut total_cost = 0;

    for stmt in &partial_ast.statements {
        total_cost += cost_estimator::estimate_cost(stmt);
    }

    total_cost
}
```

#### 最適化探索エンジン

```rust
/// Lowering戦略の探索と最適化
pub struct LoweringOptimizer {
    partial_lowerer: PartialLowerer,
}

impl LoweringOptimizer {
    pub fn new() -> Self {
        Self {
            partial_lowerer: PartialLowerer::new(),
        }
    }

    /// ノードに対する最適なLowering戦略を探索
    pub fn optimize_node_lowering(&mut self, node: &GraphNode) -> PartialAST {
        // 候補となる戦略を生成
        let strategies = self.generate_candidate_strategies(node);

        // 各戦略でlowering
        let mut candidates = self.partial_lowerer.lower_with_strategies(node, strategies);

        // コストを評価
        for candidate in &mut candidates {
            candidate.estimated_cost = Some(estimate_partial_ast_cost(candidate));
        }

        // 最小コストの戦略を選択
        candidates.sort_by_key(|c| c.estimated_cost.unwrap_or(usize::MAX));

        candidates.into_iter().next().unwrap()
    }

    /// 候補戦略を生成
    fn generate_candidate_strategies(&self, node: &GraphNode) -> Vec<LoopStrategy> {
        let mut strategies = Vec::new();

        // 基本戦略（戦略なし）
        strategies.push(LoopStrategy::default());

        // ベクトル化の候補
        if self.can_vectorize(node) {
            for width in [4, 8, 16] {
                let innermost_axis = node.view.shape.len() - 1;
                strategies.push(LoopStrategy {
                    vectorize: Some((innermost_axis, width)),
                    ..Default::default()
                });
            }
        }

        // 並列化の候補
        if self.can_parallelize(node) {
            strategies.push(LoopStrategy {
                parallelize: vec![0], // 最外ループ
                ..Default::default()
            });

            // ベクトル化 + 並列化
            for width in [4, 8, 16] {
                let innermost_axis = node.view.shape.len() - 1;
                strategies.push(LoopStrategy {
                    vectorize: Some((innermost_axis, width)),
                    parallelize: vec![0],
                    ..Default::default()
                });
            }
        }

        // タイリングの候補
        // ...

        strategies
    }

    fn can_vectorize(&self, node: &GraphNode) -> bool {
        // ベクトル化可能かチェック
        matches!(
            node.op,
            GraphOp::Elementwise(_) | GraphOp::FusedElementwise(_, _)
        )
    }

    fn can_parallelize(&self, node: &GraphNode) -> bool {
        // 並列化可能かチェック
        true // 多くの演算が並列化可能
    }
}
```

#### グラフ全体のLowering（後方互換）

```rust
/// 従来のLowererインターフェースを維持
pub struct Lowerer {
    optimizer: LoweringOptimizer,
}

impl Lowerer {
    pub fn new() -> Self {
        Self {
            optimizer: LoweringOptimizer::new(),
        }
    }

    /// グラフ全体をlower（既存のAPIと互換）
    pub fn lower(&mut self, graph: &Graph) -> AstNode {
        // トポロジカルソート
        let sorted_nodes = topological_sort(&graph.outputs);

        // 各ノードを最適化しながらlower
        let mut partial_asts = Vec::new();

        for node in sorted_nodes {
            // 各ノードで最適なLowering戦略を探索
            let partial_ast = self.optimizer.optimize_node_lowering(&node);
            partial_asts.push(partial_ast);
        }

        // PartialASTを統合して完全なプログラムを生成
        self.combine_partial_asts(graph, partial_asts)
    }

    fn combine_partial_asts(&self, graph: &Graph, partials: Vec<PartialAST>) -> AstNode {
        // すべてのPartialASTを統合
        let mut all_statements = Vec::new();
        let mut all_declarations = Vec::new();

        for partial in partials {
            all_declarations.extend(partial.declarations);
            all_statements.extend(partial.statements);
        }

        // kernel_functionとentry_functionを生成
        // （既存のロジックを流用）
        todo!()
    }
}
```

### 案B: AST候補生成レイヤー

Graph最適化の後、Loweringの前に「AST候補生成」レイヤーを追加。

```
Graph
    ↓ Graph Optimization
Optimized Graph
    ↓ AST Candidate Generation
Vec<(Graph with LoopStrategy, estimated AST cost)>
    ↓ Select best
Best Graph
    ↓ Lower
AST
```

**利点**：
- Graphレベルでの探索なので、既存のGraph最適化と統合しやすい
- LoopStrategyの組み合わせを評価

**欠点**：
- ASTを実際に生成せずにコスト推定が必要（精度が低い）
- 部分的なloweringができない

### 案C: 遅延Lowering

GraphノードをASTに変換せず、最後まで遅延させる。

```
Graph
    ↓
LazyAST (GraphノードへのポインタとLowering戦略)
    ↓ 必要に応じて実体化
AST
```

**利点**：
- メモリ効率が良い
- 必要な部分だけlower

**欠点**：
- 複雑な実装
- コスト評価のためには結局lowering必要

## 推奨: 案A（Partial Lowerer）

理由：
1. **部分的なloweringが明示的にサポート**される
2. **ASTコスト評価が正確**（実際にASTを生成）
3. **既存のLowererロジックを再利用**しやすい
4. **段階的な実装が可能**

## 実装計画

### Phase 1: 基礎データ構造
- [ ] `PartialAST`構造体の定義
- [ ] `SubGraph`構造体の定義
- [ ] `PartialLowerer`の基本実装

### Phase 2: 部分的Lowering
- [ ] `lower_subgraph()`の実装
- [ ] 既存の`lower_node`ロジックの統合
- [ ] 単一ノードのloweringテスト

### Phase 3: 戦略探索
- [ ] `LoweringOptimizer`の実装
- [ ] 候補戦略の生成ロジック
- [ ] ASTコスト推定の統合

### Phase 4: 統合と最適化
- [ ] `Lowerer::lower()`の新実装
- [ ] PartialASTの結合ロジック
- [ ] 既存テストの適合

### Phase 5: 並列化対応
- [ ] `AstNode::Range`に`parallel`フラグ追加
- [ ] 並列化戦略の候補生成
- [ ] OpenMPコード生成

## 後方互換性

既存のAPIは維持：

```rust
// 既存のAPI（変わらず使える）
pub fn lower(graph: &Graph) -> AstNode {
    let mut lowerer = Lowerer::new();
    lowerer.lower(graph)
}

// 新しいAPI（オプション）
pub fn lower_with_optimization(graph: &Graph) -> AstNode {
    let mut optimizer = LoweringOptimizer::new();
    // ... 探索ロジック
}
```

## 期待される効果

1. **より細かい最適化粒度**
   - ノード単位で最適なLowering戦略を選択

2. **コスト駆動の最適化**
   - ASTコストを実際に評価して選択

3. **拡張性の向上**
   - 新しいLowering戦略を簡単に追加
   - ユーザー定義の戦略もサポート可能

4. **デバッグの改善**
   - 各ノードのLowering結果を個別に確認可能
   - PartialASTをビジュアライズ可能

## 懸念事項と対策

### 懸念1: 性能オーバーヘッド
- 複数の戦略を試すため、lowering時間が増加

**対策**:
- 候補戦略の数を制限（top-k方式）
- 並列でloweringを実行
- キャッシュ機構（同じノード+戦略は再利用）

### 懸念2: メモリ使用量
- 複数のPartialASTを保持するため、メモリ増加

**対策**:
- ビームサーチで同時に保持する候補数を制限
- 不要なPartialASTは即座に破棄

### 懸念3: 実装の複雑さ
- 新しい抽象化レイヤーの追加

**対策**:
- 段階的な実装（Phase分け）
- 既存ロジックの最大限の再利用
- 十分なテストカバレッジ

## 次のステップ

1. この設計案をレビュー
2. Phase 1の実装開始
3. プロトタイプで検証

## 参考資料

- 既存のLowerer実装: `src/lowerer/`
- AST cost estimator: `src/opt/ast/cost_estimator.rs`
- Graph最適化: `src/opt/graph/optimizer.rs`
