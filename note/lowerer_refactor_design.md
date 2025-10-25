# Lowererリファクタリング設計 v2（シンプル版）

## 要約

**目的**: 部分的なGraphノードのLoweringとASTコスト探索を可能にする

**アプローチ**:
- GraphNodeから直接ASTに変換（中間表現なし）
- 出力ノードから再帰的にlower（メモ化でキャッシュ）
- 複数のLoopStrategy候補を試してASTコストを評価
- 最小コストの戦略を選択
- すべてのパラメータはLoweringConfigで一元管理

**主要コンポーネント**:
1. `LoweringConfig` - 設定パラメータ（ベクトル幅、タイルサイズ、ビーム幅など）
2. `RecursiveLowerer` - 再帰的なlowering + メモ化
3. `LoweringOptimizer` - 戦略探索とコスト評価
4. `Lowerer` - メインAPI

## 設計原則

1. **既存の抽象化を活用**: GraphNodeに戦略を持たせる（`strategy: Option<LoopStrategy>`）
2. **再帰的探索**: 出力ノードから再帰的にlower、メモ化でキャッシュ
3. **新しい中間表現は追加しない**: GraphNode → AstNode の直接変換
4. **コスト評価のオーバーヘッドは気にしない**: 既存のcost_estimatorで十分
5. **設定の柔軟性**: すべてのパラメータを外部から調整可能
6. **後方互換性は不要**: クリーンな再設計を優先

## 設計の全体像

```
Graph with outputs
    ↓
for each output node:
    ↓ 候補戦略生成
    [node_with_strategy1, node_with_strategy2, ...]
    ↓ 再帰的Lower（メモ化）
    [ast1, ast2, ...]
    ↓ ASTコスト評価
    [cost1, cost2, ...]
    ↓ 最小コスト選択
    best_ast
    ↓
完全なASTプログラム
```

## 主要コンポーネント

### 0. LoweringConfig - 設定パラメータ

```rust
/// Lowering処理の設定パラメータ
/// 戦略候補生成に使用される値を一箇所で管理
#[derive(Debug, Clone)]
pub struct LoweringConfig {
    /// ベクトル化幅の候補
    /// 例: vec![4, 8, 16] → SIMD幅として試す候補
    pub vectorize_widths: Vec<usize>,

    /// タイリングサイズの候補
    /// 例: vec![16, 32, 64] → ループタイリングのサイズ候補
    pub tile_sizes: Vec<usize>,

    /// アンロール係数の候補
    /// 例: vec![2, 4, 8] → ループアンロールの展開数
    pub unroll_factors: Vec<usize>,

    /// ビームサーチの幅
    /// 同時に保持する候補数の上限
    pub beam_width: usize,

    /// 最適化を有効にするか
    /// false の場合、戦略探索をスキップして従来の動作
    pub enable_optimization: bool,

    /// 並列化を有効にするか
    pub enable_parallelization: bool,

    /// ベクトル化を有効にするか
    pub enable_vectorization: bool,

    /// タイリングを有効にするか
    pub enable_tiling: bool,
}

impl Default for LoweringConfig {
    fn default() -> Self {
        Self {
            // デフォルトのベクトル幅候補（SSE, AVX, AVX-512相当）
            vectorize_widths: vec![4, 8, 16],

            // デフォルトのタイルサイズ候補（キャッシュフレンドリーなサイズ）
            tile_sizes: vec![16, 32, 64],

            // デフォルトのアンロール係数
            unroll_factors: vec![2, 4, 8],

            // デフォルトのビームサーチ幅
            beam_width: 3,

            // デフォルトで最適化を有効化
            enable_optimization: true,
            enable_parallelization: true,
            enable_vectorization: true,
            enable_tiling: false, // タイリングはデフォルトOFF（実装が複雑なため）
        }
    }
}

impl LoweringConfig {
    /// カスタム設定でビルダーパターン
    pub fn builder() -> LoweringConfigBuilder {
        LoweringConfigBuilder::new()
    }
}

/// ビルダーパターンで設定を構築
pub struct LoweringConfigBuilder {
    config: LoweringConfig,
}

impl LoweringConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: LoweringConfig::default(),
        }
    }

    pub fn vectorize_widths(mut self, widths: Vec<usize>) -> Self {
        self.config.vectorize_widths = widths;
        self
    }

    pub fn tile_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.config.tile_sizes = sizes;
        self
    }

    pub fn unroll_factors(mut self, factors: Vec<usize>) -> Self {
        self.config.unroll_factors = factors;
        self
    }

    pub fn beam_width(mut self, width: usize) -> Self {
        self.config.beam_width = width;
        self
    }

    pub fn enable_parallelization(mut self, enable: bool) -> Self {
        self.config.enable_parallelization = enable;
        self
    }

    pub fn enable_vectorization(mut self, enable: bool) -> Self {
        self.config.enable_vectorization = enable;
        self
    }

    pub fn enable_tiling(mut self, enable: bool) -> Self {
        self.config.enable_tiling = enable;
        self
    }

    pub fn build(self) -> LoweringConfig {
        self.config
    }
}
```

### 1. RecursiveLowerer - 再帰的なLowering

```rust
/// 再帰的にGraphノードをASTに変換するLowerer
pub struct RecursiveLowerer {
    /// ノード → AST のキャッシュ
    cache: HashMap<GraphNode, AstNode>,

    /// ノード → 変数名 のマッピング
    node_to_var: HashMap<GraphNode, String>,

    /// 次の一時変数ID
    next_temp_id: usize,

    /// 変数宣言の収集
    declarations: Vec<VariableDecl>,
}

impl RecursiveLowerer {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            node_to_var: HashMap::new(),
            next_temp_id: 0,
            declarations: Vec::new(),
        }
    }

    /// ノードを再帰的にlower
    /// 入力ノードが未処理なら再帰的に処理
    pub fn lower_node(&mut self, node: &GraphNode) -> AstNode {
        // キャッシュチェック
        if let Some(cached_ast) = self.cache.get(node) {
            return cached_ast.clone();
        }

        // 入力ノードを先に処理（再帰）
        for input_node in node.input_nodes() {
            self.lower_node(&input_node);
        }

        // このノード自体をlower
        let ast = self.lower_node_impl(node);

        // キャッシュに保存
        if let Some(ast) = &ast {
            self.cache.insert(node.clone(), ast.clone());
        }

        ast.unwrap_or(AstNode::Nop)
    }

    /// ノードの実際のlowering処理
    fn lower_node_impl(&mut self, node: &GraphNode) -> Option<AstNode> {
        match &node.op {
            GraphOp::Input(_) => {
                // 入力ノードは変数名をマッピングするだけ
                self.get_or_create_var_name(node);
                None
            }
            GraphOp::Const(lit) => {
                let var_name = self.get_or_create_var_name(node);
                self.declarations.push(VariableDecl {
                    name: var_name.clone(),
                    dtype: node.dtype.clone(),
                    constant: false,
                    size_expr: None,
                });
                Some(AstNode::Assign(
                    var_name,
                    Box::new(AstNode::Const(lit.clone())),
                ))
            }
            GraphOp::Elementwise(op) => {
                // 既存のElementwiseLowererを使用
                // strategy情報を渡す
                ElementwiseLowerer::lower(
                    node,
                    op,
                    |n| self.get_or_create_var_name(n),
                    &mut self.declarations,
                    &node.strategy.clone().unwrap_or_default(),
                )
            }
            // ... 他の演算も同様
            _ => {
                // 既存のlower_nodeロジックを流用
                todo!()
            }
        }
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

### 2. LoweringOptimizer - 戦略探索

```rust
/// Lowering戦略を探索して最適なASTを選択
pub struct LoweringOptimizer {
    /// 設定パラメータ
    config: LoweringConfig,
}

impl LoweringOptimizer {
    pub fn new(config: LoweringConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self::new(LoweringConfig::default())
    }

    /// GraphノードをLoweringし、最適な戦略を探索
    pub fn optimize_and_lower(&self, graph: &Graph) -> AstNode {
        // 出力ノードごとに最適化
        let mut best_programs = Vec::new();

        for output_node in &graph.outputs {
            let best_ast = self.optimize_node(output_node);
            best_programs.push(best_ast);
        }

        // 全体のプログラムを構築
        self.build_program(graph, best_programs)
    }

    /// 単一ノードの最適化
    fn optimize_node(&self, node: &GraphNode) -> (GraphNode, AstNode, usize) {
        // 1. 候補戦略を生成
        let strategies = self.generate_strategies(node);

        // 2. 各戦略でノードのバリエーションを作成
        let node_variants: Vec<GraphNode> = strategies
            .into_iter()
            .map(|strategy| node.clone().with_strategy(strategy))
            .collect();

        // 3. 各バリエーションをlower
        let mut candidates = Vec::new();
        for variant in node_variants {
            let mut lowerer = RecursiveLowerer::new();

            // 出力ノードから再帰的にlower
            let ast = lowerer.lower_node(&variant);

            // コストを評価
            let cost = self.estimate_cost(&ast);

            candidates.push((variant, ast, cost));
        }

        // 4. 最小コストを選択
        candidates.sort_by_key(|(_, _, cost)| *cost);
        candidates.into_iter().next().unwrap()
    }

    /// ノードに適用可能な戦略候補を生成
    /// configの設定に基づいて候補を生成
    fn generate_strategies(&self, node: &GraphNode) -> Vec<LoopStrategy> {
        let mut strategies = Vec::new();

        // 戦略なし（ベースライン）
        strategies.push(LoopStrategy::default());

        // 最適化が無効なら基本戦略のみ返す
        if !self.config.enable_optimization {
            return strategies;
        }

        match &node.op {
            GraphOp::Elementwise(_) | GraphOp::FusedElementwise(_, _) => {
                let innermost_axis = node.view.shape.len().saturating_sub(1);

                // ベクトル化の候補（configから取得）
                if self.config.enable_vectorization {
                    for &width in &self.config.vectorize_widths {
                        if self.can_vectorize(node, innermost_axis, width) {
                            strategies.push(LoopStrategy {
                                vectorize: Some((innermost_axis, width)),
                                ..Default::default()
                            });
                        }
                    }
                }

                // 並列化の候補
                if self.config.enable_parallelization && node.view.shape.len() > 1 {
                    strategies.push(LoopStrategy {
                        parallelize: vec![0], // 最外ループ
                        ..Default::default()
                    });

                    // ベクトル化 + 並列化
                    if self.config.enable_vectorization {
                        for &width in &self.config.vectorize_widths {
                            if self.can_vectorize(node, innermost_axis, width) {
                                strategies.push(LoopStrategy {
                                    vectorize: Some((innermost_axis, width)),
                                    parallelize: vec![0],
                                    ..Default::default()
                                });
                            }
                        }
                    }
                }

                // タイリングの候補（configから取得）
                if self.config.enable_tiling && node.view.shape.len() >= 2 {
                    for &tile_size in &self.config.tile_sizes {
                        strategies.push(LoopStrategy {
                            tile: vec![(0, tile_size), (1, tile_size)],
                            ..Default::default()
                        });
                    }
                }
            }

            GraphOp::Reduce(_, _, _) | GraphOp::FusedReduce(_, _, _) => {
                // 縮約演算の候補
                // 並列化（最外ループのみ）
                if self.config.enable_parallelization && node.view.shape.len() > 1 {
                    strategies.push(LoopStrategy {
                        parallelize: vec![0],
                        ..Default::default()
                    });
                }
            }

            _ => {
                // その他の演算は戦略なし
            }
        }

        // ビーム幅で候補数を制限
        if strategies.len() > self.config.beam_width {
            strategies.truncate(self.config.beam_width);
        }

        strategies
    }

    fn can_vectorize(&self, node: &GraphNode, axis: usize, width: usize) -> bool {
        use crate::graph::shape::Expr;

        if axis >= node.view.shape.len() {
            return false;
        }

        // shape[axis]がwidthで割り切れるかチェック
        match &node.view.shape[axis] {
            Expr::Const(n) => n % width == 0,
            _ => true, // 動的サイズの場合は試してみる
        }
    }

    fn estimate_cost(&self, ast: &AstNode) -> usize {
        use crate::opt::ast::cost_estimator;
        cost_estimator::estimate_cost(ast)
    }

    fn build_program(&self, graph: &Graph, _best_programs: Vec<(GraphNode, AstNode, usize)>) -> AstNode {
        // 既存のcreate_kernel_function, create_entry_functionのロジックを使用
        // ただし、最適化されたノードとASTを使う
        todo!()
    }
}
```

### 3. Lowerer - メインAPI

後方互換性は不要なので、シンプルに再設計。

```rust
/// GraphをASTに変換するLowerer
/// 設定に基づいて最適な戦略を探索
pub struct Lowerer {
    config: LoweringConfig,
}

impl Lowerer {
    /// デフォルト設定で作成
    pub fn new() -> Self {
        Self {
            config: LoweringConfig::default(),
        }
    }

    /// カスタム設定で作成
    pub fn with_config(config: LoweringConfig) -> Self {
        Self { config }
    }

    /// GraphをASTプログラムに変換
    pub fn lower(&self, graph: &Graph) -> AstNode {
        if self.config.enable_optimization {
            // 最適化あり：戦略を探索
            let optimizer = LoweringOptimizer::new(self.config.clone());
            optimizer.optimize_and_lower(graph)
        } else {
            // 最適化なし：単純にlower
            self.lower_without_optimization(graph)
        }
    }

    /// 最適化なしでlower（高速だがコストは最適でない可能性）
    fn lower_without_optimization(&self, graph: &Graph) -> AstNode {
        let mut lowerer = RecursiveLowerer::new();

        // 出力ノードから再帰的にlower
        for output in &graph.outputs {
            lowerer.lower_node(output);
        }

        // プログラムを構築
        self.build_program(graph, &mut lowerer)
    }

    fn build_program(&self, graph: &Graph, lowerer: &mut RecursiveLowerer) -> AstNode {
        // kernel_implとkernel_mainを生成
        // 詳細は実装時に
        todo!()
    }
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}
```

## 実装計画

### Phase 1: RecursiveLowererの実装
- [ ] `RecursiveLowerer`構造体の定義
- [ ] `lower_node()`再帰的lowering
- [ ] キャッシュ機構
- [ ] 既存のlower_nodeロジックの統合
- [ ] テスト: 単純なグラフでloweringが動作すること

### Phase 2: 戦略候補生成
- [ ] `generate_strategies()`の実装
- [ ] ベクトル化候補の生成
- [ ] 並列化候補の生成
- [ ] タイリング候補の生成
- [ ] テスト: 各演算タイプで適切な候補が生成されること

### Phase 3: 最適化探索
- [ ] `LoweringOptimizer::optimize_node()`の実装
- [ ] ASTコスト評価の統合
- [ ] 最小コスト選択
- [ ] テスト: 複数候補から最適なものが選ばれること

### Phase 4: 統合とテスト
- [ ] `Lowerer::lower()`の新実装
- [ ] 既存テストの適合
- [ ] 性能測定
- [ ] ドキュメント更新

### Phase 5: 並列化対応
- [ ] `AstNode::Range`に`parallel: bool`追加
- [ ] 並列化戦略の候補生成
- [ ] Rendererで`#pragma omp parallel for`生成
- [ ] テスト

## 使用例

### 基本的な使用

```rust
// デフォルト設定（最適化あり）
let lowerer = Lowerer::new();
let ast = lowerer.lower(&graph);
```

### カスタム設定

```rust
// 設定をカスタマイズ
let config = LoweringConfig::builder()
    .vectorize_widths(vec![8, 16])  // AVX, AVX-512のみ
    .beam_width(5)                   // ビーム幅を5に
    .enable_tiling(true)             // タイリングを有効化
    .tile_sizes(vec![32, 64])        // タイルサイズを指定
    .build();

let lowerer = Lowerer::with_config(config);
let ast = lowerer.lower(&graph);
```

### 最適化を無効化（高速だがコストは最適でない）

```rust
let config = LoweringConfig::builder()
    .enable_optimization(false)
    .build();

let lowerer = Lowerer::with_config(config);
let ast = lowerer.lower(&graph);
```

### 特定の最適化のみ有効化

```rust
// ベクトル化のみ有効（並列化とタイリングは無効）
let config = LoweringConfig::builder()
    .enable_vectorization(true)
    .enable_parallelization(false)
    .enable_tiling(false)
    .vectorize_widths(vec![4, 8])  // SSE, AVXのみ
    .build();

let lowerer = Lowerer::with_config(config);
let ast = lowerer.lower(&graph);
```

### ターゲットアーキテクチャに応じた設定

```rust
// SSE環境向け（128bit SIMD）
let sse_config = LoweringConfig::builder()
    .vectorize_widths(vec![4])  // float32 x 4
    .enable_parallelization(true)
    .build();

// AVX-512環境向け（512bit SIMD）
let avx512_config = LoweringConfig::builder()
    .vectorize_widths(vec![16])  // float32 x 16
    .enable_parallelization(true)
    .build();

// 組み込み環境向け（最適化なし）
let embedded_config = LoweringConfig::builder()
    .enable_optimization(false)
    .build();
```

## 利点

1. **シンプル**: 新しい中間表現を追加しない（GraphNode → AstNode）
2. **再帰的**: 出力から自然に探索、部分的lowering可能
3. **メモ化**: 重複計算を避ける、効率的
4. **既存の活用**: GraphNode::strategy, cost_estimatorを活用
5. **設定可能**: すべてのパラメータをLoweringConfigで一元管理
6. **柔軟性**: ターゲット環境に応じた最適化を選択可能
7. **コスト駆動**: 実際のASTコストを評価して最適戦略を選択

## 懸念事項への対応

### Q: 複数戦略を試すオーバーヘッドは？
A: コスト評価は高速（ビームサーチで既に使用）なので問題なし

### Q: メモ化のメモリ使用量は？
A: 各ノードは1回だけlowerされるので、グラフサイズに比例（許容範囲）

### Q: 戦略候補の数は？
A: ビーム幅で制限（デフォルト3）、必要に応じて調整可能

## 次のステップ

### 1. 設計の最終確認 ✅
- [x] 設計原則の確認
- [x] パラメータの柔軟性を確保
- [x] 後方互換性は不要と確認

### 2. Phase 1実装: 基礎構造
- [ ] `LoweringConfig`構造体の実装
- [ ] `LoweringConfigBuilder`の実装
- [ ] `RecursiveLowerer`の基本構造
- [ ] テスト: 設定のビルドが正しく動作すること

### 3. Phase 2実装: 再帰的Lowering
- [ ] `RecursiveLowerer::lower_node()`の実装
- [ ] メモ化機構の実装
- [ ] 既存のlower_nodeロジックの統合
- [ ] テスト: 単純なグラフで正しくlowerできること

### 4. Phase 3実装: 戦略探索
- [ ] `LoweringOptimizer`の実装
- [ ] `generate_strategies()`の実装
- [ ] ASTコスト評価の統合
- [ ] テスト: 複数戦略から最小コストが選ばれること

### 5. Phase 4実装: 統合
- [ ] `Lowerer`メインAPIの実装
- [ ] プログラム構築ロジック
- [ ] 既存テストの更新
- [ ] 統合テスト

### 6. Phase 5実装: 並列化対応
- [ ] `AstNode::Range`に`parallel: bool`追加
- [ ] 並列化戦略の候補生成
- [ ] Rendererでの対応
- [ ] テスト

## 実装優先順位

**最初に**: Phase 1-2（基礎とRecursiveLowerer）
- これで既存の機能と同等の動作を再現
- 最適化なしでも動作する基盤

**次に**: Phase 3（戦略探索）
- コスト駆動の最適化を実現
- ベクトル化、並列化候補の探索

**最後に**: Phase 4-5（統合と並列化）
- 完全な機能セット
- 並列化の実装
