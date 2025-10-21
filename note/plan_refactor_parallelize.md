# 並列化アルゴリズムの大規模改修プラン
## 概要
現在のHarpの実装では、並列化にメモリアクセスの調査などの複雑な処理を含んでおり、コードベースが膨大になってしまっている。そこで、今回の大規模改修では、並列化やループの操作などの巨視的な最適化を、Graph側のヒューリスティックな最適化として実装してしまうことにより、大幅なコードの簡略化と、よりGPUにとって扱いやすいものへの仕様変更を行うことを提案する。

## 最適化の流れについて
現在のパイプラインは、計算グラフのノード融合(決定論的) -> Lowering -> AST最適化（決定論的） -> AST最適化（ビームサーチ） -> コンパイル
のような流れになっており、ASTの最適化の段階で、ループ処理を展開、分割、あるいは最適化するなどの処理を行っている。

それを今回の改修ではグラフの段階で行うことにより、AST上でのメモリ解析を完全に廃止し、ループ処理などの最適化もグラフ上で行う。

## グラフのヒューリスティック最適化
ASTをビームサーチで最適化するように、グラフのコストを評価するモジュール(CostEstimator), グラフの編集を提案するモジュール(Suggester)を使って、ビームサーチなどを行う。
この際、コストの推定に関してだが、部分的にloweringできるなら、そのままASTのコスト推定を流用できるはずだ。
ASTのコスト推定自体が計算が難しいなら、グラフ用のコスト推定アルゴリズムを作っても問題ない。

## グラフ操作でのループの置き換え
これは単純に、loweringするループの順番を入れ替えるか、処理の前後にpermuteノードを挿入し、結果的に同じ値を返すようにしていれば問題ない。

## グラフ操作でのループタイリング
これは展開の係数で割りきれるshapeであれば非常に簡単。対象の軸をより細分化した軸をView操作で新たに作ってしまえばよい。(reshapeのようなビュー操作を行う)
割り切れない場合はView側にパディングの機能を追加する必要があるだろう。
また、タイリング係数がコンパイル時に決定する（定数）のであれば、それをそのままSIMD化, ループ展開できるはず。
GPUでのShared Memory上への転送もこの段階で戦略を決定すれば良さそうだ。

おそらく、グラフ側には各ノードにどのようにSIMD化するか、SharedMemoryをどのように扱うかのループ戦略情報を加える必要があるはず。

## Lowering処理の変更
element-wiseな処理は原則としてカーネルを作って並列処理をする。
世代ごとにトポロジカルソートを行う機能がすでに存在するので、その世代間の区切りでバリアー（同期処理）を挿入する。

## パディング
GraphにPadノードを追加し、明示的にパディングを行う？
おそらく実用上必要になるのは、長さNに対してnで割り切れるような最小のパディングをつけるような処理だろう。

### Padノードの仕様案
```rust
pub enum GraphOp {
    // ... 既存の演算 ...

    // パディング操作
    // Pad(入力ノード, 軸, パディング式)
    // 例: Pad(node, 0, Ceil(N / tile_size) * tile_size - N)
    Pad(GraphNode, usize, Expr),
}
```

パディング値は0で埋める（ゼロパディング）。
他のパディング戦略が必要な場合は、PadMode enum を追加することも検討。

## ループ戦略情報の設計

### LoopStrategy構造体
各GraphNodeに付与される最適化戦略情報:

```rust
pub struct LoopStrategy {
    /// SIMD化する軸とベクトル幅
    /// 例: Some((axis: 2, width: 4)) → 軸2を4要素ずつSIMD化
    pub vectorize: Option<(usize, usize)>,

    /// ループ展開する軸と展開係数
    /// 例: Some((axis: 1, factor: 8)) → 軸1を8回展開
    pub unroll: Option<(usize, usize)>,

    /// GPU並列化する軸
    /// 例: vec![0, 1] → 軸0と軸1をスレッドで並列化
    pub parallelize: Vec<usize>,

    /// タイリングする軸とタイルサイズ
    /// 例: vec![(axis: 0, size: 32), (axis: 1, size: 32)]
    pub tile: Vec<(usize, usize)>,

    /// Shared Memory使用のヒント（GPU向け）
    /// Backend側で実際の戦略を決定
    pub use_shared_memory: bool,
}
```

### GraphNodeDataの拡張
```rust
pub struct GraphNodeData {
    pub op: GraphOp,
    pub dtype: DType,
    pub view: View,
    pub strategy: Option<LoopStrategy>,  // 追加
}
```

## コスト推定の方針

### グラフレベルのコスト指標
部分的なloweringを行わない簡易版として、以下の指標でコストを推定:

1. **メモリアクセスコスト**
   - 各ノードの入出力サイズ（要素数）から推定
   - ストライドが非連続な場合にペナルティ
   - `cost = read_bytes + write_bytes + stride_penalty`

2. **演算コスト**
   - 演算の種類による重み付け
   - Add/Mul: 1, Recip: 5, Sin/Exp: 20, など

3. **カーネル起動コスト**
   - 融合されていないノード数に比例
   - GPU では特に重要（起動オーバーヘッドが大きい）

4. **キャッシュ効率**
   - ループの順序とメモリアクセスパターンから推定
   - 内側ループが連続メモリアクセスならボーナス

```rust
pub fn estimate_graph_cost(graph: &Graph) -> usize {
    let mut cost = 0;
    for node in graph.all_nodes() {
        cost += estimate_node_memory_cost(node);
        cost += estimate_node_compute_cost(node);
    }
    cost += estimate_kernel_launch_cost(graph.unfused_node_count());
    cost
}
```

将来的に精度が必要になったら部分的lowering を実装。

## Lowering処理の詳細

### カーネル生成戦略
1. **Element-wise処理の並列化**
   - Elementwise, FusedElementwise は自動的にGPUカーネル化
   - CPU向けには OpenMP の並列化ディレクティブを挿入

2. **Reduce処理**
   - GPU: 2段階リダクション（スレッド内 + ブロック間）
   - CPU: OpenMP reduce 句を使用

3. **世代間バリア**
   - GPU: カーネル境界が暗黙的なバリア
   - CPU: OpenMP barrier を挿入

### Lowererへの入力
```rust
impl Lowerer {
    pub fn lower(&mut self, graph: &Graph, target: Backend) -> Program {
        // 各ノードの strategy を考慮してコード生成
        // target によって異なるコード生成戦略を適用
    }
}
```

## 既存AST最適化の扱い

### 段階的な移行計画
1. **Phase 1: 並列化関連のみ移行**
   - `parallelize.rs`, `loop_extraction.rs`, `kernelize.rs` を廃止
   - Graph段階で並列化戦略を決定
   - 代数的最適化（`simplify.rs`, `constant_folding.rs`）は残す

2. **Phase 2: ループ変換の移行**
   - `loop_interchange.rs`, `loop_tiling.rs`, `loop_unroll.rs` を移行
   - Graph段階でループ戦略を決定

3. **Phase 3: 完全移行**
   - 残りのAST最適化も段階的に評価
   - 代数的最適化は AST 段階に残す可能性が高い
   - （定数畳み込みなどは AST で行う方が自然）

### 残すべきAST最適化
以下はAST段階に残す:
- `constant_folding.rs` - コンパイル時定数評価
- `simplify.rs` - 代数的簡約化
- `algebraic.rs`, `commutative.rs` などの数式最適化

理由: これらはループ構造ではなく式の変形に関するもの

## 制約と前提条件

### 実装の制約
1. **静的シェイプの優先**
   - 初期実装では静的シェイプ（コンパイル時に決定）を優先
   - 動的シェイプは後続フェーズで対応

2. **タイリング係数の制約**
   - 最初は2のべき乗のみサポート（4, 8, 16, 32, ...）
   - 割り切れない場合は必ずPadノードを挿入

3. **並列化の制約**
   - 並列化可能性の分析は Graph 構造から判定
   - データ依存関係が明確なもののみ並列化

### エラーハンドリング
- 並列化不可能なパターンを検出した場合は、並列化を諦める（フォールバック）
- パディング量が過大な場合は警告を出す（メモリ効率の低下）

## テスト戦略

### ユニットテスト
1. **Graph変換のテスト**
   - Permute挿入の正当性
   - Padノードの正しさ
   - LoopStrategy の適用

2. **コスト推定のテスト**
   - 既知のパターンでコストが正しく計算されるか
   - ビームサーチが収束するか

3. **Lowering のテスト**
   - 各 LoopStrategy が正しくコード生成されるか
   - 生成されたコードがコンパイル可能か

### 統合テスト
- 既存のテストスイートを全て通す
- パフォーマンスリグレッションがないか確認
- 特定のベンチマーク（matmul, convなど）で性能向上を確認

## ビジュアライザーの対応

### Graphビジュアライゼーションの拡張
- `VIZ=1` 環境変数で各最適化ステップを記録（既存機能）
- LoopStrategy 情報をノードのラベルに表示
- Pad ノードを特別な色で表示
- ビームサーチの探索過程を可視化（オプション）

```rust
// ビジュアライザーへの情報追加例
if is_viz_enabled() {
    add_global_snapshot(graph.clone(), "After permutation");
    add_global_snapshot(graph.clone(), "After tiling");
    add_global_snapshot(graph.clone(), "After adding loop strategies");
}
```

## バックエンド抽象化の維持

### Backend固有最適化の分離
Graph段階ではバックエンド非依存の戦略を決定:
- **Graph**: "この軸を並列化する" というヒントのみ
- **Backend**: "CUDAスレッドでどう配置するか" の詳細を決定

### Rendererの責務
```rust
pub trait Renderer {
    fn render(&self, program: &Program, strategy: &GlobalStrategy) -> String;

    // Backend固有の最適化ヒントを適用
    fn apply_backend_hints(&self, strategy: &LoopStrategy) -> BackendSpecificCode;
}
```

例:
- GPU Renderer: `parallelize` → CUDAスレッド配置、Shared Memory管理
- CPU Renderer: `parallelize` → OpenMP スレッド配置

## マイルストーン

### M1: 基盤整備（2-3週間）
- [ ] LoopStrategy 構造体の実装
- [ ] Pad ノードの追加
- [ ] GraphNodeData への strategy フィールド追加
- [ ] 基本的なコスト推定関数の実装

### M2: Graph最適化の実装（3-4週間）
- [ ] Permute 挿入による ループ順序変更
- [ ] タイリングのための reshape + pad
- [ ] LoopStrategy を提案する Suggester 群の実装
- [ ] ビームサーチの実装

### M3: Lowerer の改修（2-3週間）
- [ ] LoopStrategy を考慮したコード生成
- [ ] 並列化コードの生成（GPU/CPU両対応）
- [ ] 世代間バリアの挿入

### M4: 既存コードの削除とテスト（2週間）
- [ ] 古い並列化関連 AST 最適化の削除
- [ ] 既存テストの修正
- [ ] パフォーマンステストの実施

### M5: ドキュメント更新（1週間）
- [ ] spec/ の更新
- [ ] CLAUDE.md の更新
- [ ] 移行ガイドの作成