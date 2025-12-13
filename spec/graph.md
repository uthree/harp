# 計算グラフ

テンソル（多次元配列）単位での演算をDAGで表現する。

## 概要

計算グラフはテンソル演算をDAG（有向非巡回グラフ）として表現します。各ノードは演算または入力データを表し、エッジはデータの流れを表します。

## ProgramRootノードアーキテクチャ

### 役割
- **グラフのルート**: ProgramRootノードがグラフ全体のルートとなり、すべての出力を管理
- **Programの保持**: `AstNode::Program`を保持し、最適化で生成されたカーネル群をまとめる
- **参照カウント問題の解決**: 参照カウントを計算することで、複数出力間の依存関係を正しく扱う

### 最適化フロー
1. `Graph::output()`呼び出し時にProgramRootノードを作成/更新
2. `LoweringSuggester`がGraphOpをKernel(Function)に変換
3. `ProgramRootAbsorptionSuggester`がKernel(Function)をProgramRootのProgramに吸収
4. Lowerer がProgramRootのProgramを直接返す

## 設計方針

### GraphOpの設計
GraphOpは最適化の段階で最終的に融合されるため、**最適化よりも演算子の種類を減らすこと**を重視しています。例えば、減算は`Add`と`Neg`を組み合わせて表現します。

### Bufferノード
- **入力バッファ**: `Graph::input()`で作成されたノード
- **出力バッファ**: `LoweringSuggester`がKernelノードの`src`に含める

### 並列化
Cumulative演算（累積和、累積積など）は**Parallel Scan（Prefix Sum）アルゴリズム**を用いて並列化する予定。

## View

Viewは各軸の添え字からメモリオフセットへの線形変換を表現し、**ゼロコストの転置・次元操作**を実現します。

主なView操作：転置（permute）、次元追加/削除（unsqueeze/squeeze）、反転（flip）、拡張（expand）など。

入出力バッファのViewは常にContiguousである必要があります。

## Shape変換

**明示的なshape変換のみを許可**：演算を行う2つのノードは完全に同じshapeである必要があり、異なる場合は実行時にpanicします。ただし、**スカラー（ndim=0）は任意のテンソルにブロードキャスト可能**です。

## DType推論

演算時に自動的にDTypeが推論されます。

### サポートされるDType
- `Bool`: ブール型（内部的には8ビット整数）
- `I32`: 32ビット符号付き整数
- `F32`: 32ビット浮動小数点
- `Complex`: 複素数型（インターリーブF32バッファ`[re, im, ...]`で表現）
- `Unknown`: 型推論前の未確定型

## Kernelノード

`GraphOp::Kernel`は単一カーネル関数（`AstNode::Function`）を保持します。

### プレースホルダー変数
カスタム関数内では以下のプレースホルダー変数を使用（`custom_placeholders`モジュール）：
- `input0`, `input1`, ... : 入力バッファへのポインタ
- `output` : 出力バッファへのポインタ
- `shape0`, `shape1`, ... : 各軸のサイズ
- `ridx0`, `ridx1`, ... : ループインデックス変数

これらはLowering時に実際の値に置換されます。

### 段階的ノード融合

Graph最適化フェーズでは、以下のSuggesterにより段階的に演算が融合されます：

1. **FusionSuggester**: 連続するElementwise演算を`Kernel(Function)`に融合
2. **LoweringSuggester**: 残りのGraphOpを`Kernel(Function)`に変換
3. **ProgramRootAbsorptionSuggester**: `Kernel(Function)`をProgramRootの`Program`に吸収

## モジュール構成

### コアモジュール
- `mod.rs`: Graph、GraphNode、DType等の基本データ構造
- `ops.rs`: GraphOp定義と基本的な演算
- `node_view_ops.rs`: GraphNodeのView操作メソッド
- `strategy.rs`: 並列化戦略の定義
- `visualization.rs`: DOT形式でのグラフ可視化

### 高レベル演算
- `hlops.rs`: 高レベル演算のヘルパー関数
- `hlops_conv.rs`: 畳み込み演算（conv1d/2d/3d、conv_transpose1d/2d/3dのエントリポイント）

### 畳み込みモジュール (conv/)
- `conv/mod.rs`: convモジュールの定義
- `conv/params.rs`: ConvParams構造体（kernel_size, stride, dilation, groups）
- `conv/ops.rs`: N次元畳み込み実装（conv_nd, conv_transpose_nd）

1D/2D/3D畳み込みは共通のN次元ロジック（conv_nd/conv_transpose_nd）に委譲されます。

### Shape関連
- `shape/mod.rs`: Shape関連モジュールの定義
- `shape/expr.rs`: シンボリック式（Expr）の定義と演算
- `shape/view.rs`: View構造体と基本操作
- `shape/view_ops.rs`: View操作（unfold_nd等）

## 未実装

- Thread/ThreadGroupレベルの並列実行のLowering
- ループタイル化（TilingSuggester）
- 行列乗算（matmul、batch_matmul）
- 複素数型のReduce/Cumulative演算のLowering
