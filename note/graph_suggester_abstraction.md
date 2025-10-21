# Graph Suggester の抽象化設計

## 現状の問題

現在の実装では、各Suggesterが個別の構造体として実装されており、共通のインターフェースがない：

```rust
pub struct VectorizationSuggester;
impl VectorizationSuggester {
    pub fn suggest(graph: &Graph) -> Vec<Graph> { ... }
}

pub struct ParallelizationSuggester;
impl ParallelizationSuggester {
    pub fn suggest(graph: &Graph) -> Vec<Graph> { ... }
}
```

**問題点:**
- 新しいSuggesterを追加する際に`CombinedSuggester`を手動で更新する必要がある
- Suggesterを動的に有効/無効にできない
- ユーザーが独自のSuggesterを実装しにくい
- テストコードでモックが作りにくい

## 解決案: GraphSuggester Trait

### 1. Trait定義

```rust
/// Graph最適化の提案を生成するSuggesterの共通インターフェース
pub trait GraphSuggester {
    /// 最適化候補を生成
    fn suggest(&self, graph: &Graph) -> Vec<Graph>;

    /// Suggesterの名前（デバッグ用）
    fn name(&self) -> &str;

    /// 優先度（高いほど優先、デフォルト: 0）
    fn priority(&self) -> usize {
        0
    }

    /// このSuggesterを有効にするか（デフォルト: true）
    fn is_enabled(&self) -> bool {
        true
    }

    /// Suggesterの説明（ログやドキュメント生成用）
    fn description(&self) -> &str {
        ""
    }
}
```

### 2. 各Suggesterでの実装

#### VectorizationSuggester

```rust
pub struct VectorizationSuggester {
    /// 最大ベクトル幅の制限（None = 無制限）
    pub max_vector_width: Option<usize>,
}

impl Default for VectorizationSuggester {
    fn default() -> Self {
        Self {
            max_vector_width: Some(16), // AVX-512相当
        }
    }
}

impl GraphSuggester for VectorizationSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();

        for output in &graph.outputs {
            let configs = Self::find_vectorizable_loops(output);

            for config in configs {
                // max_vector_widthでフィルタリング
                if let Some(max_width) = self.max_vector_width {
                    if config.vector_width > max_width {
                        continue;
                    }
                }

                if let Some(vectorized_graph) = Self::apply_vectorization(graph, output, &config) {
                    suggestions.push(vectorized_graph);
                }
            }
        }

        suggestions
    }

    fn name(&self) -> &str {
        "Vectorization"
    }

    fn priority(&self) -> usize {
        100 // ベクトル化は高優先度
    }

    fn description(&self) -> &str {
        "SIMD vectorization for innermost loops"
    }
}
```

#### ParallelizationSuggester

```rust
pub struct ParallelizationSuggester {
    /// 並列化する最小サイズ
    pub min_parallel_size: usize,
}

impl Default for ParallelizationSuggester {
    fn default() -> Self {
        Self {
            min_parallel_size: 4,
        }
    }
}

impl GraphSuggester for ParallelizationSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();

        for output in &graph.outputs {
            let configs = Self::find_parallelizable_axes(output);

            for config in configs {
                // 十分なサイズの軸のみ並列化
                // （実装詳細は省略）

                if let Some(parallel_graph) = Self::apply_parallelization(graph, output, &config) {
                    suggestions.push(parallel_graph);
                }
            }
        }

        suggestions
    }

    fn name(&self) -> &str {
        "Parallelization"
    }

    fn priority(&self) -> usize {
        90 // 並列化も高優先度
    }

    fn description(&self) -> &str {
        "Multi-threaded parallelization for outer loops"
    }
}
```

#### TilingSuggester

```rust
pub struct TilingSuggester {
    /// 使用可能なタイルサイズ
    pub tile_sizes: Vec<usize>,
}

impl Default for TilingSuggester {
    fn default() -> Self {
        Self {
            tile_sizes: vec![8, 16, 32, 64],
        }
    }
}

impl GraphSuggester for TilingSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        // 実装
        Vec::new() // TODO
    }

    fn name(&self) -> &str {
        "Tiling"
    }

    fn priority(&self) -> usize {
        50
    }

    fn description(&self) -> &str {
        "Cache-friendly loop tiling"
    }
}
```

#### LoopPermutationSuggester

```rust
pub struct LoopPermutationSuggester;

impl GraphSuggester for LoopPermutationSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        // 実装
        Vec::new() // TODO
    }

    fn name(&self) -> &str {
        "LoopPermutation"
    }

    fn priority(&self) -> usize {
        80 // メモリアクセスパターンの改善は高優先
    }

    fn description(&self) -> &str {
        "Reorder loops for better memory access patterns"
    }
}
```

### 3. CombinedSuggesterの改善

#### 動的なSuggester管理

```rust
/// 複数のSuggesterを統合
pub struct CombinedSuggester {
    suggesters: Vec<Box<dyn GraphSuggester>>,
}

impl CombinedSuggester {
    /// デフォルトのSuggesterセットで初期化
    pub fn new() -> Self {
        Self {
            suggesters: vec![
                Box::new(VectorizationSuggester::default()),
                Box::new(ParallelizationSuggester::default()),
                Box::new(LoopPermutationSuggester),
                Box::new(TilingSuggester::default()),
            ],
        }
    }

    /// 空のCombinedSuggesterを作成
    pub fn empty() -> Self {
        Self {
            suggesters: Vec::new(),
        }
    }

    /// Suggesterを追加
    pub fn add_suggester(&mut self, suggester: Box<dyn GraphSuggester>) -> &mut Self {
        self.suggesters.push(suggester);
        self
    }

    /// Suggesterを削除（名前で指定）
    pub fn remove_suggester(&mut self, name: &str) -> &mut Self {
        self.suggesters.retain(|s| s.name() != name);
        self
    }

    /// 全てのSuggesterから候補を収集
    pub fn suggest_all(&self, graph: &Graph) -> Vec<Graph> {
        let mut all_suggestions = Vec::new();

        // 優先度順にソート
        let mut sorted_suggesters: Vec<_> = self.suggesters.iter().collect();
        sorted_suggesters.sort_by_key(|s| std::cmp::Reverse(s.priority()));

        for suggester in sorted_suggesters {
            if !suggester.is_enabled() {
                continue;
            }

            let suggestions = suggester.suggest(graph);
            all_suggestions.extend(suggestions);
        }

        all_suggestions
    }

    /// 候補をコストでランク付け
    pub fn rank_by_cost(&self, suggestions: Vec<Graph>) -> Vec<(Graph, usize)> {
        use crate::opt::graph::cost_estimator;

        let mut ranked: Vec<(Graph, usize)> = suggestions
            .into_iter()
            .map(|g| {
                let cost = cost_estimator::estimate_graph_cost(&g.outputs);
                (g, cost)
            })
            .collect();

        ranked.sort_by_key(|(_, cost)| *cost);
        ranked
    }

    /// Suggesterの一覧を取得
    pub fn list_suggesters(&self) -> Vec<(&str, &str, usize)> {
        self.suggesters
            .iter()
            .map(|s| (s.name(), s.description(), s.priority()))
            .collect()
    }
}

impl Default for CombinedSuggester {
    fn default() -> Self {
        Self::new()
    }
}
```

### 4. ビルダーパターンでの設定

```rust
impl CombinedSuggester {
    /// ビルダーパターンで構築
    pub fn builder() -> CombinedSuggesterBuilder {
        CombinedSuggesterBuilder::new()
    }
}

pub struct CombinedSuggesterBuilder {
    suggesters: Vec<Box<dyn GraphSuggester>>,
}

impl CombinedSuggesterBuilder {
    pub fn new() -> Self {
        Self {
            suggesters: Vec::new(),
        }
    }

    /// ベクトル化を追加
    pub fn with_vectorization(mut self) -> Self {
        self.suggesters.push(Box::new(VectorizationSuggester::default()));
        self
    }

    /// カスタム設定のベクトル化を追加
    pub fn with_vectorization_custom(mut self, max_width: usize) -> Self {
        self.suggesters.push(Box::new(VectorizationSuggester {
            max_vector_width: Some(max_width),
        }));
        self
    }

    /// 並列化を追加
    pub fn with_parallelization(mut self) -> Self {
        self.suggesters.push(Box::new(ParallelizationSuggester::default()));
        self
    }

    /// タイリングを追加
    pub fn with_tiling(mut self) -> Self {
        self.suggesters.push(Box::new(TilingSuggester::default()));
        self
    }

    /// ループ順序変更を追加
    pub fn with_loop_permutation(mut self) -> Self {
        self.suggesters.push(Box::new(LoopPermutationSuggester));
        self
    }

    /// カスタムSuggesterを追加
    pub fn with_custom(mut self, suggester: Box<dyn GraphSuggester>) -> Self {
        self.suggesters.push(suggester);
        self
    }

    /// ビルド
    pub fn build(self) -> CombinedSuggester {
        CombinedSuggester {
            suggesters: self.suggesters,
        }
    }
}
```

### 5. 使用例

#### 基本的な使用

```rust
// デフォルト設定
let suggester = CombinedSuggester::new();
let suggestions = suggester.suggest_all(&graph);

// ビルダーパターン
let suggester = CombinedSuggester::builder()
    .with_vectorization()
    .with_parallelization()
    .build();
```

#### カスタマイズ

```rust
// ベクトル化のみ有効
let suggester = CombinedSuggester::builder()
    .with_vectorization_custom(8)  // AVX相当
    .build();

// 特定のSuggesterを削除
let mut suggester = CombinedSuggester::new();
suggester.remove_suggester("Tiling");  // タイリングを無効化

// Suggesterを追加
suggester.add_suggester(Box::new(MyCustomSuggester::new()));
```

#### デバッグ情報の表示

```rust
let suggester = CombinedSuggester::new();

// 登録されているSuggesterを表示
for (name, desc, priority) in suggester.list_suggesters() {
    println!("[{}] {} (priority: {})", name, desc, priority);
}

// 出力例:
// [Vectorization] SIMD vectorization for innermost loops (priority: 100)
// [Parallelization] Multi-threaded parallelization for outer loops (priority: 90)
// [LoopPermutation] Reorder loops for better memory access patterns (priority: 80)
// [Tiling] Cache-friendly loop tiling (priority: 50)
```

#### ユーザー定義Suggester

```rust
/// ユーザー定義の最適化
pub struct FusionSuggester {
    max_fusion_depth: usize,
}

impl GraphSuggester for FusionSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        // 独自の融合ロジック
        Vec::new()
    }

    fn name(&self) -> &str {
        "CustomFusion"
    }

    fn priority(&self) -> usize {
        60
    }

    fn description(&self) -> &str {
        "Custom operator fusion strategy"
    }
}

// 使用
let suggester = CombinedSuggester::builder()
    .with_vectorization()
    .with_custom(Box::new(FusionSuggester { max_fusion_depth: 3 }))
    .build();
```

### 6. Optimizerでの使用

```rust
impl BeamSearchOptimizer {
    pub fn new() -> Self {
        Self {
            beam_width: 3,
            max_depth: 5,
            suggester: CombinedSuggester::new(),
        }
    }

    pub fn with_suggester(mut self, suggester: CombinedSuggester) -> Self {
        self.suggester = suggester;
        self
    }

    pub fn optimize(&self, initial_graph: &Graph) -> Graph {
        // suggesterを使用
        let suggestions = self.suggester.suggest_all(initial_graph);
        // ...
    }
}
```

### 7. テストの改善

#### モックSuggester

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct MockSuggester {
        name: String,
        suggestions: Vec<Graph>,
    }

    impl GraphSuggester for MockSuggester {
        fn suggest(&self, _graph: &Graph) -> Vec<Graph> {
            self.suggestions.clone()
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn priority(&self) -> usize {
            50
        }
    }

    #[test]
    fn test_custom_suggester() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![10.into()]);
        graph.output(input);

        let mock = MockSuggester {
            name: "Mock".to_string(),
            suggestions: vec![graph.clone()],
        };

        let combined = CombinedSuggester::empty()
            .add_suggester(Box::new(mock));

        let suggestions = combined.suggest_all(&graph);
        assert_eq!(suggestions.len(), 1);
    }
}
```

## 利点

1. **拡張性**: 新しいSuggesterを簡単に追加できる
2. **柔軟性**: Suggesterを動的に有効/無効化できる
3. **テスト容易性**: モックSuggesterでテストしやすい
4. **ユーザーフレンドリー**: ユーザーが独自のSuggesterを実装可能
5. **保守性**: 共通のインターフェースで統一的に管理
6. **デバッグ**: Suggesterの一覧表示や優先度の確認が容易

## 実装計画

### Phase 1: Trait定義と基本実装

1. [ ] `GraphSuggester` traitの定義
2. [ ] 既存SuggesterでのTrait実装
3. [ ] テスト（既存の動作を壊さないことを確認）

### Phase 2: CombinedSuggesterの改善

1. [ ] 動的なSuggester管理機能
2. [ ] ビルダーパターンの実装
3. [ ] 優先度によるソート

### Phase 3: ドキュメントと使用例

1. [ ] APIドキュメント
2. [ ] 使用例の追加
3. [ ] ユーザー定義Suggesterのガイド

## 後方互換性

既存のコードとの互換性を保つため、static methodも残す：

```rust
impl VectorizationSuggester {
    /// 互換性のためのstatic method
    pub fn suggest(graph: &Graph) -> Vec<Graph> {
        Self::default().suggest(graph)
    }
}
```

これにより、既存のコードを壊さずに段階的に移行できる。

## 参考: AST Suggesterとの整合性

AST最適化のSuggesterも同様のパターンを使用しており、整合性がある：

```rust
// AST Suggester（既存）
pub trait AstSuggester {
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode>;
}

// Graph Suggester（新規）
pub trait GraphSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph>;
}
```
