# Lowerer

## 役割
計算グラフをASTに変換する層。グラフ最適化（LoweringSuggester）によってKernelノードに変換された計算をASTカーネル関数として出力します。

## アーキテクチャ

**グラフ最適化は必須**であり、`lower()`関数は自動的にグラフ最適化を実行します。LoweringSuggesterがほとんどのGraphOpを`GraphOp::Kernel`（`AstNode::Function`を保持）に変換するため、Lowererが直接処理するノードは限定されています：

- **Kernel**: LoweringSuggesterで生成（AST関数を展開）
- **Fold**: LoweringSuggesterでは未対応（直接lowering）
- **FusedReduce**: タプル出力が必要なため未対応（エラー）

## Lowering手順
1. **グラフ最適化**: LoweringSuggesterでGraphOpをKernelノードに変換
2. **トポロジカルソート**: Kahnのアルゴリズムで世代別にグループ化
3. **バッファーマッピング**: 各ノードの出力バッファー名を決定
4. **カーネル生成**: 各Kernelノードのfunction部分をカーネル関数として出力
5. **main関数生成**: 中間バッファーの確保・解放、カーネルの順次呼び出し

## バッファー命名
- グラフ入力ノード → `input0`, `input1`, ...
- 中間ノード → `tmp0`, `tmp1`, ...
- グラフ出力ノード → `output`

## 変数命名法則
- `ridx{n}`: 入力軸インデックス
- `oidx{n}`: 出力軸インデックス（Reduce用）
- `alu{n}`: 一時スカラー変数
- `acc{n}`: アキュムレータ（reduce/cumulative用）

## ファイル構成
- `mod.rs`: コア構造、トポロジカルソート、グラフ最適化呼び出し
- `kernel.rs`: Kernelノードのlowering
- `fold.rs`: Fold演算のlowering
- `utils.rs`: ユーティリティ関数（オフセット計算、型変換、シグネチャ生成）

## 実装状況

### 実装済み
- グラフ最適化の自動実行（LoweringSuggesterによるKernelノード変換）
- Kernelノードのlowering（AST関数の展開）
- Fold演算のlowering
- 中間バッファー管理（main関数でのtmp{n}の自動確保・解放）
- トポロジカルソート（Kahn）
- 動的shape対応のシグネチャ生成

### 未実装
- FusedReduce演算（タプル出力が必要）
- SIMD化のコード生成
