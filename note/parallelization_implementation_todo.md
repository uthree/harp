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
