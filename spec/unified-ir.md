# 統一IR（Unified IR）プロトタイプ

## 概要

tinygradのUOpを参考にした、統一中間表現のプロトタイプ実装。現在のharpの設計（AstNode、Graph、Lowerer、Viewを分離）を、単一のグラフ構造とパターンマッチングによる項書き換えに統合できるかを検証するための試験的な実装。

**場所**: `crates/unified_ir/`

**ステータス**: プロトタイプ（本体のharpとは独立）

## 設計思想

### 統一的な中間表現

全ての演算を単一の`UOp`型で表現する：
- **高レベル演算**: テンソル単位の操作（Input, Elementwise, Reduce）
- **中レベル演算**: ループ構造とメモリ操作（Loop, Load, Store）
- **低レベル演算**: スカラー演算とGPU固有操作（Add, Mul, ThreadIdx）

これにより、tinygradのような段階的loweringと統一的な最適化が可能になる。

### パターンマッチングによる最適化

`Wildcard`ノードを使用したパターンマッチングにより、代数的簡約を宣言的に記述：

```rust
// x + 0 → x
pattern: Add(Wildcard(0), Const(0.0))
replacement: Wildcard(0)
```

これは現在のharpにおける`AstRewriteRule`を一般化したもの。

## アーキテクチャ

```
高レベルIR (UOp)
  ↓ Rewriter（パターンマッチング最適化）
最適化済みIR (UOp)
  ↓ Lowerer（段階的lowering）
低レベルIR (UOp)
  ↓ CodeGen
OpenCL/Metal/Cカーネル
```

## 主要コンポーネント

### UOp (uop.rs)

統一中間表現の型定義。直接enumとして定義し、子ノードは`Rc<UOp>`で参照。

**設計上の特徴**:
- 全ての演算を`enum UOp { ... }`のバリアントとして直接表現
- 各バリアントに必要なフィールド（dtype, lhs, rhs等）を含む
- `Rc<UOp>`による参照カウントでDAG構造を実現

**現在のharpとの対応**:
- `UOp::Input/Elementwise/Reduce` ≈ `GraphOp`の高レベル演算
- `UOp::Loop/Load/Store` ≈ Lowererが生成する中間表現
- `UOp::Add/Mul/...` ≈ `AstNode`のスカラー演算

### helper (helper.rs)

`Rc<UOp>`の生成を簡潔に記述するためのヘルパー関数群。

**設計上の特徴**:
- マクロで二項演算・単項演算のヘルパーを自動生成
- 本体のharpの`src/ast/helper.rs`と同様のパターン
- `helper::add()`, `helper::mul()`, `helper::input()` 等で簡潔に構築可能

### Rewriter (rewriter.rs)

パターンマッチングと書き換えルールの適用。

**実装方式**:
- `RewriteRule`: パターン（UOp）と置換関数のペア
- `pattern_match()`: Wildcardを含むパターンのマッチング
- `apply()`: ルールを不動点まで反復適用

**現在のharpとの対応**:
- `AstRewriteRule`を一般化したもの
- Graph最適化とAST最適化を統一的に記述可能

### Lowerer (lowering.rs)

高レベル演算を低レベル演算に段階的に変換。

**主な変換**:
- `Elementwise` → 並列`Loop` + `Load`/`Store`
- `Reduce` → 外側並列`Loop` + 内側シーケンシャル`Loop`

**現在のharpとの対応**:
- `src/lowerer/`の機能を書き換えルールとして実装
- GraphからASTへの変換をUOpからUOpへの変換として表現

### CodeGen (codegen.rs)

低レベルIRからGPUカーネルコードを生成。

**対応バックエンド**:
- OpenCL（実装済み）
- C/Metal（将来的に拡張可能）

**現在のharpとの対応**:
- `src/backend/`のコード生成機能に相当

## 現在の実装範囲

### ✅ 実装済み

- Element-wise演算（Add, Mul, Neg, Recip, Sqrt, Max, Div）
- Reduce演算（Sum, Max, Min）
- 基本的な最適化ルール（`x + 0`, `x * 1`, `x * 0`など）
- 並列ループとシーケンシャルループの生成
- OpenCLカーネル生成

### ⚠️ 制限事項（プロトタイプのため）

- Shape推論が簡易実装（固定値を使用）
- View変換は未実装
- 一部の演算が未実装（Exp, Log等）
- エラーハンドリングが最小限
- 最適化ルールが基本的なもののみ

## 本体のharpとの関係

### 独立性

`crates/unified_ir/`は完全に独立したクレートとして実装されており、本体のharpには影響を与えない。あくまで設計検証用のプロトタイプ。

### 類似点と相違点

**類似するコンセプト**:
- Wildcardとパターンマッチング → harpも同様の機構を持つ
- 段階的lowering → harpのGraph→AST変換に相当
- 代数的簡約 → harpのAST最適化と同様

**主な相違点**:
- **型の統一性**: harpは`GraphNode`と`AstNode`を分離、unified_irは`UOp`で統合
- **最適化の記述**: harpは手続き型（Suggester）、unified_irは宣言型（RewriteRule）
- **責務の分離**: harpは明確な層分離、unified_irは段階的変換

## 評価

### メリット

1. **統一性**: 全ての演算を同じ枠組みで扱える
2. **柔軟性**: 最適化パスの追加・変更が容易
3. **デバッグ性**: 各段階のIRを可視化できる
4. **宣言的**: パターンマッチングによる直感的な最適化記述

### デメリット

1. **型安全性の低下**: 高レベルと低レベルの区別がコンパイル時にチェックできない
2. **移行コスト**: 既存コード（約29,000行）の大規模リファクタリングが必要
3. **パフォーマンス懸念**: `Rc<>`のオーバーヘッド増加の可能性
4. **学習コスト**: 新しい抽象化の理解が必要

## 今後の方向性

### 短期（現状維持を推奨）

- このプロトタイプは**参考実装**として保持
- 本体のharpは現在の設計を継続
- 新しい最適化が必要になった際の選択肢として検討

### 中期（段階的統合の可能性）

もし以下の課題が顕在化した場合、段階的統合を検討：
- Graph最適化とAST最適化の重複が問題になる
- 新しい最適化パスの追加が現在の構造では困難
- より柔軟な最適化順序の制御が必要

### 長期（完全統合は慎重に）

完全な統合は以下を満たす場合のみ検討：
- 明確なパフォーマンス改善が実証される
- 移行コストを上回るメリットが明確
- 段階的移行パスが確立される

## 関連ファイル

- 仕様書: `spec/unified-ir.md`（本ファイル）
- 実装: `crates/unified_ir/src/`
- テスト: `crates/unified_ir/tests/integration_test.rs`
- ドキュメント: `crates/unified_ir/README.md`

## 参考

- tinygrad UOp: https://github.com/tinygrad/tinygrad
- 現在のharpの設計: `spec/graph.md`, `spec/ast.md`, `spec/lowerer.md`
