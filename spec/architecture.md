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

### runtime/opencl/ - OpenCLバックエンド（`opencl` feature）
GPUを利用した計算を行うOpenCLバックエンド。

```
src/runtime/opencl/
├── mod.rs           # モジュール定義
├── device.rs        # OpenCLDevice実装
├── buffer.rs        # OpenCLBuffer実装
├── kernel.rs        # カーネルコンパイル・キャッシュ
├── codegen.rs       # FusedKernelのコード生成
├── interpreter.rs   # UOpグラフのインタープリタ実行（フュージョン対応）
└── ops/             # 演算カーネルソース生成
    ├── mod.rs
    ├── elementwise.rs   # Unary/Binary演算
    ├── compare.rs       # CmpLt, CmpEq, Where
    ├── reduce.rs        # Sum, ReduceMax
    └── movement.rs      # Expand, Permute, Cast
```

主要構造体：
- `OpenCLDevice`: GPU実行デバイス（Context, CommandQueue管理）
- `OpenCLBuffer`: GPUメモリ上のバッファ
- `KernelCache`: コンパイル済みカーネルのキャッシュ
- `OpenCLInterpreter`: UOpグラフを操作ごとにカーネル実行
- `FusedKernelCodeGen`: 複数操作を融合したカーネルのコード生成

使用方法：
```rust
// OpenCL featureを有効にしてビルド
// cargo build --features opencl

eclat::init_opencl()?;  // OpenCLデバイスを初期化・登録
let device = eclat::device::get_device("OPENCL").unwrap();
```

### schedule/ - カーネルフュージョン・スケジューリング
複数のUOp操作を1つのカーネルに融合する最適化を行う。

```
src/schedule/
├── mod.rs           # モジュール定義
├── item.rs          # ScheduleItem（スケジュール単位）
├── kernel.rs        # FusedKernel, FusedOp（融合カーネル表現）
├── scheduler.rs     # フュージョン判定・グループ化
└── analysis.rs      # グラフ解析（参照カウント等）
```

主要構造体：
- `ScheduleItem`: 1回のカーネル実行を表す。複数のUOpを融合可能
- `FusedKernel`: 融合された操作チェーンを持つカーネル表現
- `FusedOp`: 融合カーネル内の個別操作
- `Scheduler`: UOp DAGを解析しフュージョン判定・スケジュール生成
- `GraphAnalysis`: グラフ解析（参照カウント、フュージョン可否判定）

フュージョン対応：
- Elementwiseフュージョン: 連続するelementwise操作を1カーネルに融合
- 条件: 形状一致、単一参照、非Load/Const
- `OpenCLInterpreter::eval_with_fusion()`で有効化

## 設計原則

### 遅延評価
テンソル操作は即座に計算せず、UOpのDAGとして記録する。`realize()`呼び出し時にグラフ全体を評価し、最適化の機会を提供する。

### イミュータビリティ
UOpとTensorはイミュータブル。操作は新しいノードを作成し、古いノードは変更しない。Arc参照カウントにより効率的な共有が可能。

### バックエンド抽象化
Deviceトレイトを通じて複数バックエンドをサポート。現在はCPUインタプリタとOpenCLバックエンドを実装。将来的にCUDA、Metalなどをサポート予定。

## データフロー

1. ユーザーがTensorを作成（`Tensor::new()`, `Tensor::zeros()`など）
2. 操作を適用（`+`, `sum()`など）→ UOp DAGが構築される
3. `realize()`または`to_vec()`呼び出し
4. Deviceが UOp DAGを受け取り評価
5. 結果がBufferに格納される
6. ユーザーがデータを取得

## 現在の制限事項

- 自動微分は未実装
- Reduceのフュージョンは部分的（reduce前のelementwise融合は非対応）
- ブロードキャスト付きのフュージョンは非対応

## 今後の拡張

### Phase 4: 自動微分
- requires_grad
- backward

### Phase 5: さらなる最適化
- Reduceカーネル内でのelementwiseフュージョン
- ブロードキャスト対応のフュージョン
- メモリ最適化（中間バッファ削減）
