# Unified IR プロトタイプ

tinygradのUOpを参考にした、統一中間表現（Unified Intermediate Representation）のプロトタイプ実装です。

## 概要

このプロトタイプは、harpの現在の設計（AstNode、Graph、Lowerer、Viewを分離）を、単一のグラフ構造とパターンマッチングによる項書き換えに統合できるか検証するために作成されました。

## 主な特徴

### 1. 統一的なグラフ構造 (UOp)

高レベル演算から低レベル演算まで、全てを単一の `UOp` 型で表現します：

- **高レベル**: `Input`, `Const`, `Elementwise`, `Reduce`
- **中レベル**: `Loop`, `Load`, `Store`, `Sequence`
- **低レベル**: `ThreadIdx`, `Add`, `Mul`, `Max`, etc.

### 2. パターンマッチングと書き換え

`Wildcard` ノードとパターンマッチングにより、代数的簡約や最適化を統一的に記述できます：

```rust
// x + 0 → x
let pattern = UOp::add(UOp::wildcard(0, DType::F32), UOp::const_val(0.0, DType::F32));
let replacement = |mapping| mapping[&0].clone();
```

### 3. 段階的Lowering

高レベル演算を段階的に低レベル演算に変換します：

```
高レベル (Elementwise)
  ↓ lowering
中レベル (Loop + Load/Store)
  ↓ codegen
OpenCLカーネル
```

### 4. OpenCLコード生成

低レベルIRから直接OpenCLカーネルコードを生成できます。

## アーキテクチャ

```
┌──────────────────────────────────────┐
│      高レベルIR (UOp)                 │
│  Input, Elementwise, Reduce          │
└──────────┬───────────────────────────┘
           │
           ↓ Rewriter (パターンマッチング)
┌──────────────────────────────────────┐
│      最適化済みIR (UOp)               │
│  代数的簡約、定数畳み込み             │
└──────────┬───────────────────────────┘
           │
           ↓ Lowerer (段階的lowering)
┌──────────────────────────────────────┐
│      低レベルIR (UOp)                 │
│  Loop, Load, Store, ThreadIdx        │
└──────────┬───────────────────────────┘
           │
           ↓ CodeGen
┌──────────────────────────────────────┐
│      OpenCLカーネルコード             │
└──────────────────────────────────────┘
```

## ファイル構成

```
crates/unified_ir/
├── src/
│   ├── lib.rs              # モジュールのエクスポート
│   ├── uop.rs              # UOp型定義とビルダー
│   ├── rewriter.rs         # パターンマッチングと書き換えルール
│   ├── lowering.rs         # 高レベル→低レベル変換
│   └── codegen.rs          # OpenCLコード生成
├── tests/
│   └── integration_test.rs # 統合テスト
├── Cargo.toml
└── README.md
```

## 使用例

### Element-wise演算

```rust
use unified_ir::*;
use harp::DType;

// 高レベルIRの構築
let a = UOp::input("a", vec![1024], DType::F32);
let b = UOp::input("b", vec![1024], DType::F32);
let add = UOp::elementwise(ElementwiseOp::Add, vec![a, b], DType::F32);

// 最適化
let rules = basic_optimization_rules();
let rewriter = Rewriter::new(rules);
let optimized = rewriter.apply(&add, 10);

// Lowering
let lowerer = Lowerer::new(256);
let lowered = lowerer.lower(&optimized);

// コード生成
let mut codegen = OpenCLCodegen::new();
let kernel = codegen.generate_kernel(&lowered, "add_kernel");
```

### Reduce演算

```rust
let a = UOp::input("a", vec![10, 20], DType::F32);
let sum = UOp::reduce(ReduceOp::Sum, a, 1, vec![10, 20]);

let lowerer = Lowerer::new(256);
let lowered = lowerer.lower(&sum);

let mut codegen = OpenCLCodegen::new();
let kernel = codegen.generate_kernel(&lowered, "reduce_kernel");
```

## テスト

```bash
# 全てのテストを実行
cargo test -p unified_ir

# 統合テスト（詳細出力）
cargo test -p unified_ir --test integration_test -- --nocapture

# コード品質チェック
cargo clippy -p unified_ir
```

## 実装状況

### ✅ 完成した機能

- UOp型定義と基本ビルダー
- パターンマッチングと書き換えシステム
- 基本的な最適化ルール（代数的簡約、定数畳み込み）
- Element-wise演算のlowering
- Reduce演算のlowering
- OpenCLコード生成
- 統合テスト

### ⚠️ 制限事項（プロトタイプのため）

- Shape推論が簡易実装（実際のshapeではなく固定値）
- バッファー名が簡略化（`input0`, `input1`など）
- 一部の演算が未実装（Exp, Log, View変換など）
- エラーハンドリングが最小限
- 最適化ルールが基本的なもののみ

### 🔜 将来的な拡張

- より高度な最適化パス
- View変換のサポート
- 複雑なメモリレイアウト
- 型推論の改善
- ループ融合、タイル化などの高度な最適化
- Metal/CUDAバックエンド

## 評価

### メリット

1. **統一性**: 全ての演算を単一の型で表現できる
2. **柔軟性**: 最適化パスの追加・変更が容易
3. **デバッグ性**: 各段階のIRを可視化できる
4. **シンプルさ**: パターンマッチングによる直感的な最適化記述

### デメリット

1. **型安全性**: Rustの型システムの恩恵を受けにくい
2. **パフォーマンス**: Rc<>のオーバーヘッド
3. **移行コスト**: 既存コードの大規模なリファクタリングが必要

## 結論

このプロトタイプにより、tinygradのUOp的な統一IRは**技術的に実現可能**であることが確認できました。

ただし、harpの現在の設計（明確な層分離）も十分に機能しており、統合のメリットとコストを慎重に検討する必要があります。

短期的には現在の設計を維持し、必要に応じて段階的に統合を検討することを推奨します。
