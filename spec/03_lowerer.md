# Lowerer モジュール仕様

## 概要

Lowererモジュールは計算グラフ（Graph）を抽象構文木（AST）に変換する役割を担います。高レベルなテンソル演算を具体的なループ構造とメモリアクセスパターンに変換します。

## 主要な構造

### Lowerer

```rust
pub struct Lowerer {
    next_temp_id: usize,
    node_to_var: HashMap<GraphNode, String>,
}
```

**役割:**
- グラフノードから変数名へのマッピング管理
- 一時変数名の生成
- プログラム全体の生成

## 変換プロセス

### 1. lower(graph: &Graph) -> Program

グラフ全体をプログラムに変換。

**生成される関数:**
1. `kernel_impl`: 実際の演算を実行するカーネル関数
2. `kernel_main`: エントリーポイント関数（バッファのキャストを担当）

### 2. create_kernel_function(graph: &Graph) -> Function

カーネル実装関数を生成。

**手順:**
1. トポロジカルソート（世代別）を実行
2. 入力・出力ノードに変数名を事前割り当て
3. 各世代のノードを処理してAST文を生成
4. 世代間にBarrierを挿入
5. 出力ノードへのコピーコードを生成

**引数:**
- `input_0`, `input_1`, ... : 入力バッファへのポインタ
- `output_0`, `output_1`, ... : 出力バッファへのポインタ

### 3. create_entry_function(graph: &Graph) -> Function

エントリーポイント関数を生成。

**役割:**
- `void**` バッファポインタから型付きポインタへのキャスト
- `kernel_impl`の呼び出し

**引数:**
- `bufs: void**` : バッファポインタの配列
- `shape_vars: size_t*` : シェイプ変数の配列

## トポロジカルソート

### topological_sort_by_generation(graph: &Graph) -> Vec<Vec<GraphNode>>

**特徴:**
- 世代（レベル）ごとにノードをグループ化
- 各世代は並列実行可能なノードのグループ
- 依存関係に基づいてレベルを計算

**アルゴリズム:**
1. 各ノードの入次数を計算
2. 入次数が0のノードから開始
3. 処理したノードの隣接ノードの入次数を減らす
4. 新たに入次数が0になったノードを次の世代に追加

## ノードの変換

### lower_node(node: &GraphNode) -> Option<AstNode>

各グラフノードをASTノードに変換。

**演算別の処理:**

#### Input
- 引数として扱われるため、コード生成なし
- 変数名のマッピングのみ実施

#### Const
- 変数宣言と代入文を生成

#### Elementwise
- ElementwiseLowererに委譲
- ビュー情報に基づくループ生成

#### Reduce
- ReduceLowererに委譲
- 初期化ループ + 集約ループを生成

#### Cumulative
- CumulativeLowererに委譲
- 累積演算のループを生成

#### View
- メモリコピーなし
- 変数名をソースと同じにする
- ビュー情報（stride/offset）のみ変更

#### Contiguous
- 非連続メモリを連続メモリに変換
- コピーループを生成
- 入力のビュー（非連続）から出力のビュー（連続）へ

#### Cast
- 型変換ループを生成
- 各要素を新しい型にキャスト

#### Fold (col2im)
- 初期化ループ（ゼロクリア）
- 累積ループ（重複ウィンドウの加算）
- stride と dilation パラメータを考慮

#### FusedElementwise
- FusedLowererに委譲
- ASTノードとキャプチャを使用

#### FusedReduce
- FusedLowererに委譲
- 複数軸の縮約

#### FusedElementwiseReduce
- FusedLowererに委譲
- 要素演算と縮約を融合

#### FusedElementwiseCumulative
- FusedLowererに委譲
- 要素演算と累積演算を融合

## メモリアクセスパターン

### compute_memory_index(strides, offset, dim) -> AstNode

ストライドとオフセットからメモリインデックスを計算。

```
index = offset + sum(ridxN * strideN for each dimension)
```

**ループ変数:**
- `ridx0`, `ridx1`, ... : 各次元のループカウンタ

## コピーループの生成

### create_contiguous_copy_loop(...)

非連続メモリから連続メモリへのコピー。

**構造:**
```
for ridx0 in 0..shape[0]:
  for ridx1 in 0..shape[1]:
    ...
      result[result_index] = input[input_index]
```

### create_cast_loop(...)

型変換を伴うコピー。

**構造:**
```
for ridx0 in 0..shape[0]:
  for ridx1 in 0..shape[1]:
    ...
      result[result_index] = (target_dtype)input[input_index]
```

## Foldループの生成

### create_fold_loops(...)

col2im操作のループを生成。

**Phase 1: 初期化**
```
for ridx0 in 0..result_shape[0]:
  ...
    result[index] = 0.0
```

**Phase 2: 累積**
```
for ridx0 in 0..input_shape[0]:
  ...
    for ridxW in 0..window_size:
      result_idx = ridx0 * stride + ridxW * dilation
      result[result_idx] += input[input_idx]
```

## 変数名の管理

### get_or_create_var_name(node: &GraphNode) -> String

グラフノードに対応する変数名を取得または生成。

**命名規則:**
- 入力ノード: `input_N`
- 出力ノード: `output_N`
- 一時変数: `tempN`

## サブモジュール

### elementwise.rs
- 要素単位演算のループ生成
- ブロードキャストの処理
- 複数入力のシェイプ調整

### reduce.rs
- 縮約演算のループ生成
- 初期値の設定
- 集約処理

### cumulative.rs
- 累積演算のループ生成
- アキュムレータの管理

### fused.rs
- 融合済み演算のループ生成
- ASTノードのキャプチャ処理
- 複数演算の統合

### utils.rs
- 共通ユーティリティ関数
- シェイプ計算
- メモリインデックス計算
