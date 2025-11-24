# リファクタリング履歴

このファイルは、harpプロジェクトで実施された主要なリファクタリングの記録です。

## 2025-11-25: 優先度：高のリファクタリング実施

### 背景
コードベース全体の探索を実施し、以下の問題を特定：
- **grad_fn.rs**: 930行の大規模ファイル、17個のGradFn実装が混在
- **graph/mod.rs**: 1678行、実装とテストが混在
- **Expr::Const抽出パターン**: 35箇所以上で同じコードが重複
- **エラーハンドリング**: 30箇所以上でpanic!を使用（未対応）

### 実施したリファクタリング

#### 1. Expr型ヘルパーメソッドの追加
**ファイル**: `src/graph/shape/expr.rs`

**追加したメソッド**:
```rust
impl Expr {
    /// 定数値を取得（定数の場合のみ）
    pub fn as_const(&self) -> Option<isize>;

    /// 定数値をusizeとして取得（定数の場合のみ）
    pub fn as_usize(&self) -> Option<usize>;

    /// 定数値を強制的に取得（定数でない場合はパニック）
    pub fn expect_const(&self, msg: &str) -> isize;

    /// 定数値をusizeとして強制的に取得（定数でない場合はパニック）
    pub fn expect_usize(&self, msg: &str) -> usize;
}
```

**効果**:
- 35箇所以上の重複コードを削減
- コードの可読性が向上
- エラーメッセージの統一

**Before**:
```rust
let value = match &shape[i] {
    crate::graph::shape::Expr::Const(v) => *v as usize,
    _ => panic!("requires constant"),
};
```

**After**:
```rust
let value = shape[i].expect_usize("requires constant");
```

#### 2. grad_fn.rsの機能別分割
**ファイル**: `src/autograd/grad_fn.rs` (930行) → `src/autograd/grad_fn/` (モジュール化)

**新しいモジュール構成**:
```
src/autograd/grad_fn/
├── mod.rs (1.3KB)      - GradFnトレイト定義とモジュール再エクスポート
├── basic.rs (4.7KB)    - 基本演算の勾配関数
│   ├── AddBackward
│   ├── MulBackward
│   ├── NegBackward
│   ├── RecipBackward
│   ├── ReduceSumBackward
│   ├── MaxBackward
│   ├── AddConstBackward
│   └── MulConstBackward
├── math.rs (2.8KB)     - 数学関数の勾配関数
│   ├── Log2Backward
│   ├── Exp2Backward
│   ├── SinBackward
│   └── SqrtBackward
├── memory.rs (2.8KB)   - メモリ操作の勾配関数
│   ├── PadBackward
│   └── SliceBackward
└── conv.rs (25KB)      - 畳み込み演算の勾配関数
    ├── Conv1dBackward
    ├── Conv2dBackward
    └── Conv3dBackward
```

**効果**:
- ファイルが適切なサイズに分割（最大25KB）
- 各モジュールの責任が明確化
- 関連する勾配関数が同じファイルにまとまる
- 保守性が大幅に向上

#### 3. graph/mod.rsからテストを分離
**ファイル**: `src/graph/mod.rs` (1678行) → `src/graph/mod.rs` (1082行) + `tests/graph_tests.rs` (596行)

**効果**:
- 実装とテストコードの分離
- 59個のテストが独立したファイルに
- ファイルサイズが35%削減
- テストの可読性と管理性が向上

### 検証結果

#### テスト結果
- **統合テスト**: 676個 - 全て成功 ✓
- **doctests**: 49個 - 全て成功 ✓
- **合計**: 725個のテスト - 100%成功

#### コード品質チェック
- `cargo fmt`: 成功 ✓
- `cargo clippy --all-targets -- -D warnings`: 警告なし ✓

### 影響範囲

#### 変更されたファイル
- `src/graph/shape/expr.rs`: ヘルパーメソッド追加
- `src/autograd/grad_fn.rs`: 削除（モジュール化）
- `src/autograd/grad_fn/mod.rs`: 新規作成
- `src/autograd/grad_fn/basic.rs`: 新規作成
- `src/autograd/grad_fn/math.rs`: 新規作成
- `src/autograd/grad_fn/memory.rs`: 新規作成
- `src/autograd/grad_fn/conv.rs`: 新規作成
- `src/graph/mod.rs`: テスト削除（1678行 → 1082行）
- `tests/graph_tests.rs`: 新規作成
- `spec/graph.md`: リファクタリング履歴更新
- `spec/autograd.md`: リファクタリング履歴更新

#### 影響のなかったファイル
- 公開API: 変更なし（内部実装の変更のみ）
- テストコード: テスト自体は変更なし（配置のみ変更）

### メトリクス

| 項目 | 変更前 | 変更後 | 改善 |
|------|--------|--------|------|
| grad_fn.rs行数 | 930行 | 分割済み | ✓ |
| graph/mod.rs行数 | 1678行 | 1082行 | -35% |
| Expr::Const抽出パターン | 35箇所の重複 | ヘルパー関数に統一 | ✓ |
| モジュール数（autograd） | 2個 | 6個 | +4個 |
| 最大ファイルサイズ | 930行 | 最大25KB (conv.rs) | ✓ |

### 残りのタスク（優先度：高）

#### カスタムエラー型の定義とResult型への移行
**状態**: 未実施

**問題**:
- 現在30箇所以上でpanic!を使用
- 動的形状（式）に対応できず、定数形状のみサポート
- エラーメッセージが一貫性を欠いている

**提案**:
```rust
enum GradFnError {
    IncorrectInputCount { expected: usize, actual: usize },
    NonConstantShape { location: String },
    InvalidConfig { reason: String },
}

// GradFn traitをResult型に変更
trait GradFn {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor])
        -> Result<Vec<Option<Tensor>>, GradFnError>;
}
```

**推定工数**: 1-2日
**優先度**: 高
**タイミング**: 別セッションでの実施を推奨

### 学んだこと

1. **リファクタリングの順序が重要**
   - まずヘルパーメソッドを追加してから分割すると、分割後のコードがクリーンになる
   - テストの分離は最後に行うと、途中での検証が容易

2. **自動化の重要性**
   - Pythonスクリプトでのパターン置換は、手動より正確で高速
   - 置換後も全テストが通ることを確認

3. **モジュール分割の粒度**
   - 500行超えは分割を検討
   - 責任ごとにファイルを分けると保守性が向上
   - ただし、過度な分割は避ける（1モジュール = 1-2KB程度が下限）

### 次回リファクタリングへの推奨事項

1. **エラーハンドリングの改善**（優先度：高）
   - Result型への段階的移行
   - カスタムエラー型の定義

2. **テストセットアップの統一**（優先度：中）
   - `autograd/tests.rs`（768行）のセットアップ重複削減
   - テストヘルパー関数の作成

3. **Conv/Fold/Unfoldの実装統合**（優先度：低）
   - マクロベースの統合
   - コード重複の削減

### 参考資料

- [graph.md - リファクタリング履歴](spec/graph.md#コード構成とリファクタリング履歴)
- [autograd.md - リファクタリング履歴](spec/autograd.md#コード構成とリファクタリング履歴)
- [リファクタリング分析レポート](https://github.com/anthropics/claude-code) - 初期分析結果
