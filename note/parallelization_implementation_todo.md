# 並列化実装のTODO

## 現状

### 実装済み
- ✅ `ParallelizationSuggester` がGraph最適化候補を生成
- ✅ `LoopStrategy.parallelize: Vec<usize>` フィールドが定義済み
- ✅ Suggesterが`parallelize`フィールドに並列化する軸を設定

### 未実装（問題点）
- ❌ **ASTの`Range`ノードに並列化フラグがない**
- ❌ **Lowererが`strategy.parallelize`を無視している**
- ❌ **並列化されたコードが生成されない**

## 問題の詳細

### 1. ASTノードの不足

`src/ast/node.rs:66-74`に定義されている`AstNode::Range`：

```rust
Range {
    counter_name: String,
    start: Box<Self>,
    max: Box<Self>,
    step: Box<Self>,
    body: Box<Self>,
    unroll: Option<usize>, // アンロールはある
    // parallel: bool が無い！
}
```

**必要な変更**：
```rust
Range {
    counter_name: String,
    start: Box<Self>,
    max: Box<Self>,
    step: Box<Self>,
    body: Box<Self>,
    unroll: Option<usize>,
    parallel: bool,  // 追加が必要
}
```

### 2. Lowererの未対応

`src/lowerer/elementwise.rs`などでループを生成する際、`strategy.parallelize`を確認していない。

**現状**：
- `strategy.vectorize` → ✅ 実装済み（vector_widthに反映）
- `strategy.parallelize` → ❌ 完全に無視

**必要な実装箇所**：
- `src/lowerer/elementwise.rs` - elementwise演算のループ生成
- `src/lowerer/reduce.rs` - reduce演算のループ生成
- `src/lowerer/cumulative.rs` - cumulative演算のループ生成

### 3. コード生成の流れ

```
Graph with LoopStrategy
    ↓
Lowerer (strategy.parallelizeを読む) ← ここが未実装
    ↓
AstNode::Range { parallel: true }  ← フィールドが無い
    ↓
Renderer (#pragma omp parallel for) ← 実装済み？
    ↓
並列化されたCコード
```

## 実装計画

### Phase 1: ASTの拡張
1. `AstNode::Range`に`parallel: bool`フィールドを追加
2. 全ての`Range`生成箇所を修正（`parallel: false`をデフォルトに）
3. `RangeBuilder`に`parallel()`メソッドを追加
4. パターンマッチング、transform、children取得などを更新

**影響範囲**：
- `src/ast/node.rs` - 定義
- `src/ast/range_builder.rs` - ビルダー
- `src/ast/transform.rs` - 変換処理
- `src/ast/pattern.rs` - パターンマッチング
- 既存のテストコード全般

### Phase 2: Lowererの実装
1. ループ生成時に`strategy.parallelize`をチェック
2. 並列化対象の軸（最外ループなど）に`parallel: true`を設定
3. ネストされたループで適切な軸を並列化

**実装箇所**：
- `src/lowerer/elementwise.rs`
- `src/lowerer/reduce.rs`
- `src/lowerer/cumulative.rs`
- `src/lowerer/utils.rs` (共通ヘルパー関数)

**考慮事項**：
- `parallelize: vec![0]` → 最外ループを並列化
- `parallelize: vec![0, 1]` → 2次元並列化（GPU向け、OpenMPでは通常サポートされない）
- ベクトル化との競合を避ける（最内ループは並列化しない）

### Phase 3: Rendererの確認・実装
1. C/OpenCLレンダラーで`parallel: true`を処理
2. OpenMP: `#pragma omp parallel for`を出力
3. OpenCL/CUDA: カーネルの並列実行として扱う

**確認が必要**：
- `src/backend/c/renderer.rs` - 既に実装されているか？
- `src/backend/c_like.rs` - 共通処理

### Phase 4: テスト
1. 並列化されたコードが生成されることを確認
2. OpenMPでコンパイル・実行できることを確認
3. ベクトル化と並列化の組み合わせテスト

## 参考情報

### 既存の類似実装

**ベクトル化**（既に実装済み）：
- `src/lowerer/elementwise.rs:140-145` でvector_widthを抽出
- 最内ループでのみ適用
- `Load/Store`ノードに`vector_width`を渡す

**アンロール**（既に実装済み）：
- `AstNode::Range { unroll: Option<usize> }`
- Rendererで`#pragma unroll`を出力

### 並列化の戦略

`ParallelizationSuggester`が設定する`parallelize`：
```rust
LoopStrategy {
    vectorize: None,           // 最内ループをベクトル化
    parallelize: vec![0],      // 軸0（最外ループ）を並列化
    unroll: None,
    tile: vec![],
    use_shared_memory: false,
}
```

### 典型的な出力例

**入力Graph**：
```rust
let a = graph.input(DType::F32, vec![1024.into(), 1024.into()]);
let b = graph.input(DType::F32, vec![1024.into(), 1024.into()]);
let c = a + b;
graph.output(c);
```

**期待されるCコード**（並列化 + ベクトル化）：
```c
#pragma omp parallel for
for (size_t i0 = 0; i0 < 1024; i0++) {
    for (size_t i1 = 0; i1 < 1024; i1 += 8) {
        *((float8*)(output + i0*1024 + i1)) =
            *((float8*)(input_a + i0*1024 + i1)) +
            *((float8*)(input_b + i0*1024 + i1));
    }
}
```

## 注意事項

### 破壊的変更
`AstNode::Range`の構造変更は**破壊的変更**です：
- 既存の全てのパターンマッチングが壊れる
- 全てのRange生成箇所を修正する必要がある
- テストコードも大量に修正が必要

### 回避策の検討
破壊的変更を避けるには：
1. 新しいノード`ParallelRange`を追加する
2. `Range`のメタデータとして別途管理する（複雑）
3. そのまま破壊的変更として実装する（推奨）

### 優先度
- ベクトル化は既に動作している
- 並列化は多くのユースケースで重要
- GPUバックエンドでは必須の機能

## 関連ファイル

- `src/graph/mod.rs` - LoopStrategy定義
- `src/opt/graph/suggester/parallelization.rs` - 並列化提案
- `src/ast/node.rs` - ASTノード定義
- `src/lowerer/*.rs` - Lowerer実装
- `src/backend/c/renderer.rs` - コード生成
- `note/simd_implementation_plan.md` - SIMD実装計画（参考）
- `note/graph_suggester_abstraction.md` - Suggester抽象化（参考）

## 次のステップ

実装する際は以下の順序で進める：

1. **設計レビュー**: 破壊的変更の影響範囲を確認
2. **Phase 1**: ASTの拡張（全ての既存コードを修正）
3. **Phase 2**: Lowererの実装
4. **Phase 3**: Rendererの確認・実装
5. **Phase 4**: テストと検証

推定作業時間: 4-6時間（ASTの変更影響が大きい）

## 根本的な問題: Lowererの設計

### 現在のLowererの問題点

並列化を実装しようとすると、**Lowerer自体の再設計が必要**になる可能性が高い。

#### 1. ループ生成ロジックの散在

各演算タイプごとに個別のループ生成ロジックがある：

- `src/lowerer/elementwise.rs` - elementwise演算用のループ
- `src/lowerer/reduce.rs` - reduce演算用のループ
- `src/lowerer/cumulative.rs` - cumulative演算用のループ
- `src/lowerer/fold.rs` - fold演算用のループ

**問題**: 並列化、tiling、loop permutationなどを実装するには、これら全てを修正する必要がある。

#### 2. LoopStrategyの部分的な適用

現状：
- `vectorize` → ✅ elementwiseのみ実装
- `parallelize` → ❌ 未実装
- `tile` → ❌ 未実装
- `unroll` → ❌ 未実装（AST側では定義されているが、Lowererは無視）

**問題**: 新しい最適化を追加するたびに、全ての演算タイプを修正する必要がある。

#### 3. ループ構造の硬直性

現在のループ生成は各演算に特化した形で書かれており：

```rust
// elementwise.rsの例
for i0 in 0..shape[0] {
    for i1 in 0..shape[1] {
        for i2 in 0..shape[2] {
            // ベクトル化は最内ループのみ
            result[i0][i1][i2] = lhs[i0][i1][i2] + rhs[i0][i1][i2];
        }
    }
}
```

これでは：
- ループの順序変更（permutation）が困難
- タイリングの挿入が困難
- 並列化する軸の選択が困難
- 最適化の組み合わせ（並列化+ベクトル化+タイリング）が困難

### 必要な再設計

#### Option A: ループジェネレータの抽象化

ループ生成を専用のモジュールに集約：

```rust
// 擬似コード
struct LoopGenerator {
    dimensions: Vec<Dimension>,
    strategy: LoopStrategy,
}

impl LoopGenerator {
    fn generate_loops(&self, body: impl FnOnce(...) -> AstNode) -> AstNode {
        // strategyに基づいて最適化されたループ構造を生成
        // - parallelize: 指定された軸に#pragma omp
        // - vectorize: 最内ループをベクトル化
        // - tile: タイリングループを挿入
        // - permute: ループ順序を変更
    }
}
```

**利点**:
- 全ての演算で同じループ生成ロジックを使用
- 最適化の追加が1箇所で済む
- 最適化の組み合わせを一元管理

**欠点**:
- 大規模なリファクタリングが必要
- reduce/cumulativeなど特殊なループパターンへの対応

#### Option B: IR (Intermediate Representation) の導入

Graph → IR → AST の2段階lowering：

```
Graph (with LoopStrategy)
    ↓
Loop IR (抽象的なループ表現)
    ↓ LoopStrategy適用
Optimized Loop IR
    ↓
AST
```

**利点**:
- ループ最適化をIRレベルで実行
- 既存のコンパイラ技術（polyhedral model等）を適用可能
- より高度な最適化が可能

**欠点**:
- 非常に大規模な設計変更
- 実装コストが高い

#### Option C: 段階的な修正

現在の設計を維持しつつ、最小限の修正で対応：

1. ループ生成ヘルパー関数を`utils.rs`に追加
2. 各演算lowererでそのヘルパーを使用
3. 最適化ロジックはヘルパーに集約

**利点**:
- 比較的小さな変更で済む
- 段階的に移行可能

**欠点**:
- 根本的な解決にはならない
- 将来的にはOption AかBが必要

### 現実的なアプローチ

#### 短期（次のステップ）

**Option Cで最小限の実装**:
1. `src/lowerer/loop_generator.rs`を新規作成
2. 基本的なループ生成ヘルパーを実装
3. 並列化のみサポート（vectorizeは既存のまま）
4. elementwiseから段階的に移行

#### 中期（次の大きなリファクタリング）

**Option Aに移行**:
1. LoopGeneratorを完全実装
2. 全ての最適化（vectorize, parallelize, tile, unroll）を統合
3. 全ての演算lowererを書き換え

#### 長期（将来的な理想形）

**Option Bを検討**:
- Polyhedral compilationの導入
- 自動最適化の実現
- MLコンパイラ（TVM、XLA等）との統合

### 影響範囲の再見積もり

**Option C（最小限）を選んだ場合**:
- 新規ファイル: `src/lowerer/loop_generator.rs`
- 修正ファイル: `elementwise.rs`, `reduce.rs`, `cumulative.rs`
- 推定時間: 8-12時間

**Option A（推奨）を選んだ場合**:
- 新規ファイル: `src/lowerer/loop_generator.rs`, `src/lowerer/loop_ir.rs`
- 修正ファイル: lowerer配下の全ファイル
- 推定時間: 20-30時間

**Option B（理想形）を選んだ場合**:
- 新規モジュール: `src/ir/`
- 大規模な設計変更
- 推定時間: 40-80時間

## 推奨事項

1. **短期**: Option Cで並列化のみ実装し、動作を確認
2. **中期**: 別途リファクタリングタスクとしてOption Aを計画
3. **長期**: プロジェクトの成熟度に応じてOption Bを検討

並列化の実装を進める前に、**Lowererの再設計方針を決定する必要がある**。
