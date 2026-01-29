# Eclat アーキテクチャ仕様書

## 概要

Eclatはtinygradの設計思想をRustで再実装した軽量テンソルライブラリである。遅延評価、カーネルフュージョン、複数バックエンド対応を特徴とする。

## アーキテクチャ概要

```
Tensor API (遅延評価)
    ↓
UOp DAG (中間表現)
    ↓
Device (バックエンド実行)
    ↓
Buffer (データ保持)
```

## モジュール構成

### dtype.rs - 型システム
- `DType`: テンソル要素のデータ型（Bool, Int32, Int64, Float32, Float64）
- `Scalar`: バッファへの読み書きを可能にするトレイト
- `ScalarValue`: 型消去されたスカラー値

### shape.rs - 形状システム
- `Shape`: テンソルの次元情報を保持
  - broadcast計算
  - stride計算
  - インデックス変換（flat ↔ multi-dimensional）
- `View`: ストライドとオフセットを持つテンソルビュー

### ops.rs - 操作定義
- `Ops`: 計算グラフ内のすべての操作を列挙
  - バッファ操作: Load, Store, Const
  - 単項操作: Neg, Exp, Log, Sqrt, Sin, Cos, Recip
  - 二項操作: Add, Sub, Mul, Div, Max, CmpLt, CmpEq
  - 三項操作: Where
  - リダクション: Sum, ReduceMax
  - 移動操作: Reshape, Expand, Permute, Pad, Shrink, Stride
  - キャスト: Cast

### uop.rs - 中間表現
- `UOp`: 計算グラフのノード
  - Arc<UOpInner>による参照カウント
  - イミュータブル設計
  - 各ノードは操作、型、形状、ソース、引数を保持
- `UOpArg`: ノードの引数（スカラー、形状、軸など）

### tensor.rs - Tensor API
- `Tensor`: ユーザー向けAPI
  - 遅延評価: 操作はUOpグラフを構築するのみ
  - `realize()`: グラフを評価してバッファに実体化
  - 演算子オーバーロード（+, -, *, /, -）
- `IntoTensorData`: 配列/Vecからテンソルへの変換トレイト

### device.rs - デバイス抽象化
- `Device`: バックエンドインターフェース
  - `name()`: デバイス名
  - `alloc()`: バッファ割り当て
  - `realize()`: UOpグラフを評価
- `Buffer`: デバイス上のデータバッファ
- `BufferMap`: バッファID→バッファの管理

### runtime/cpu.rs - CPUバックエンド
- `CpuDevice`: CPU実行デバイス
- `CpuBuffer`: ホストメモリ上のバッファ
- `CpuInterpreter`: UOpグラフを直接解釈実行

## 設計原則

### 遅延評価
テンソル操作は即座に計算せず、UOpのDAGとして記録する。`realize()`呼び出し時にグラフ全体を評価し、最適化の機会を提供する。

### イミュータビリティ
UOpとTensorはイミュータブル。操作は新しいノードを作成し、古いノードは変更しない。Arc参照カウントにより効率的な共有が可能。

### バックエンド抽象化
Deviceトレイトを通じて複数バックエンドをサポート。現在はCPUインタプリタのみ実装。将来的にOpenCL、CUDA、Metalなどをサポート予定。

## データフロー

1. ユーザーがTensorを作成（`Tensor::new()`, `Tensor::zeros()`など）
2. 操作を適用（`+`, `sum()`など）→ UOp DAGが構築される
3. `realize()`または`to_vec()`呼び出し
4. Deviceが UOp DAGを受け取り評価
5. 結果がBufferに格納される
6. ユーザーがデータを取得

## 現在の制限事項

- 自動微分は未実装
- GPUバックエンドは未実装
- カーネルフュージョンは未実装
- スケジューリング最適化は未実装

## 今後の拡張

### Phase 2: OpenCLバックエンド
- OpenCLコードレンダラ
- カーネルコンパイル・実行

### Phase 3: カーネル最適化
- Linearizer
- カーネルフュージョン

### Phase 4: 自動微分
- requires_grad
- backward
