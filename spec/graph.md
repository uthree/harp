# 計算グラフ

テンソル（多次元配列）単位での演算をDAGで表現する。

## 概要

計算グラフはテンソル演算をDAG（有向非巡回グラフ）として表現します。各ノードは演算または入力データを表し、エッジはデータの流れを表します。

## Programへの収束

### 概要
Graph最適化フェーズでは、全てのGraphOpノードが最終的に1つの`AstNode::Program`に収束します。

### 最適化フロー
1. `LoweringSuggester`がGraphOpを`Kernel(Function/Kernel)`に変換
2. `KernelMergeSuggester`が複数の`Kernel`ノードを1つの`Kernel(Program)`にマージ
3. Lowererが`collect_kernels_as_program()`でProgramを抽出

### AstNode::Programの構造
```rust
Program {
    functions: Vec<AstNode>,  // Kernel/Functionのリスト
    execution_order: Option<Vec<KernelExecutionInfo>>,  // 実行順序（オプション）
}
```

`execution_order`が設定されている場合、カーネルの実行順序と依存関係を明示的に指定します。
設定されていない場合（None）、Backend側で実行順序が推測されます。

## 設計方針

### GraphOpの設計
GraphOpは最適化の段階で最終的に融合されるため、**最適化よりも演算子の種類を減らすこと**を重視しています。例えば、減算は`Add`と`Neg`を組み合わせて表現します。

### Bufferノード
- **入力バッファ**: `Graph::input()`で作成されたノード
- **出力バッファ**: `LoweringSuggester`がKernelノードの`src`に含める

### 並列化
Cumulative演算（累積和、累積積など）は**Parallel Scan（Prefix Sum）アルゴリズム**を用いて並列化する予定。

## View

Viewは各軸の添え字からメモリオフセットへの変換を表現し、**ゼロコストの転置・次元操作**を実現します。

### Viewの種類

- **Linear**: 線形変換（shape, strides, offset）。転置、unsqueezeなど多くの操作で使用
- **IndexExpr**: 任意の式による変換。tile（循環アクセス）、flipなど非線形操作で使用

### 主なView操作

| 操作 | 説明 | 結果のView型 |
|------|------|-------------|
| permute | 軸の順序変更（転置） | Linear |
| unsqueeze/squeeze | 次元追加/削除 | 入力と同じ |
| flip | 軸方向の反転 | IndexExpr |
| tile | 循環アクセス（剰余演算） | IndexExpr |
| repeat | サイズ1の軸を拡張 | 入力と同じ |
| reshape | 形状変更 | Linear（要連続性） |
| unfold | スライディングウィンドウ | Linear（要連続性） |
| gather | インデックステンソルによる要素収集 | IndexExpr（LoadIndex含む） |

### Gather操作

`GraphNode::gather(dim, index)`はPyTorchの`torch.gather`に相当する操作です。指定した軸に沿って、indexテンソルの値に従って入力テンソルから要素を収集します。

```rust
// output[i][j][k] = input[i][index[i][j][k]][k]  // dim=1の場合
let gathered = input.gather(1, &index);
```

#### 実装詳細

- **Expr::LoadIndex**: 別のソースバッファからインデックス値を読み込む式
  - `src_index`: GraphNode.srcのインデックス（1以上）
  - `offset_expr`: 読み込み位置を計算する式
- **View::IndexExpr**: LoadIndexを含むindex_exprでGatherパターンを表現
- **GraphNode.src**: `[input, index]`の2つのノードを保持

#### 制約

- indexテンソルの次元数はinputと同じである必要がある
- 出力形状はindexと同じ

#### Lowering

LoadIndexを含むViewのLoweringには、バッファ変数のリストを渡すコンテキスト付き変換が必要です。

- `expr_to_ast_with_sources(expr, src_vars, dtype)`: LoadIndexを`load()`に変換
- `build_strided_offset_with_sources(view, ndim, src_vars, dtype)`: LoadIndex対応のオフセット計算

Contiguous演算では、入力ViewにLoadIndexが含まれる場合、srcのすべてのバッファを変数として渡します。

#### View融合（ViewMergeSuggester）

View→View連鎖を効率的に融合するため、以下の機能が実装されています：

- **`View::compose(outer, inner)`**: 2つのViewを合成。Linear×IndexExprやLoadIndex含む複合パターンに対応
- **`GraphNode::flatten_view_chain()`**: View連鎖を再帰的に辿り、最終Viewとsrc配列をフラット化
  - 連続Gatherでsrc配列をマージ: `[inner_srcs[0], extra_srcs..., inner_srcs[1..]...]`
  - LoadIndex.src_indexを累積シフトして正しいバッファを参照

ViewMergeSuggesterは、LoadIndexを含むViewの場合はViewノードを維持しつつsrc配列を融合します。

#### 将来の拡張

- Scatter操作（Viewの責務外、書き込み+競合解決が必要）は別のGraphOpとして実装予定

### 自動Contiguous化

View操作の中には連続したメモリレイアウト（Linear View）を前提とするものがあります。GraphNodeレベルでは、これらの操作にIndexExpr Viewを渡した場合、自動的にContiguousノードが挿入されます。

```
View構造体 (低レベル)          GraphNode構造体 (高レベル)
─────────────────────          ────────────────────────────
reshape: IndexExprでpanic  →   reshape: 自動でContiguous挿入
unfold:  IndexExprでpanic  →   unfold:  自動でContiguous挿入
```

これにより、ユーザーはメモリレイアウトを意識せずにView操作を連鎖できます。

### 制約

入出力バッファのViewは常にContiguousである必要があります。

## Shape変換

**明示的なshape変換のみを許可**：演算を行う2つのノードは完全に同じshapeである必要があり、異なる場合は実行時にpanicします。ただし、**スカラー（ndim=0）は任意のテンソルにブロードキャスト可能**です。

## DType推論

演算時に自動的にDTypeが推論されます。

### サポートされるDType
- `Bool`: ブール型（内部的には8ビット整数）
- `I32`: 32ビット符号付き整数
- `F32`: 32ビット浮動小数点
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
2. **LoweringSuggester**: 残りのGraphOpを`Kernel(Function/Kernel)`に変換
3. **KernelMergeSuggester**: 複数の`Kernel`ノードを1つの`Kernel(Program)`にマージ

### SIMD化

`LoweringSuggester`はElementwise演算に対してSIMD化されたバージョンを提案できます。

```rust
// スカラー版のみ（デフォルト）
let suggester = LoweringSuggester::new();

// SIMD版も生成（幅4と8の候補）
let suggester = LoweringSuggester::with_simd_widths(vec![4, 8]);
```

SIMD化されたループは以下の構造を持ちます：

1. **SIMDループ**: 最内軸を`simd_width`ずつ処理（ベクトルLoad/Store使用）
2. **テールループ**: 残りの要素をスカラーで処理

```c
// 例: shape=[10, 128], simd_width=4
for (int ridx0 = 0; ridx0 < 10; ridx0++) {
    // SIMDループ (step=4)
    for (int ridx1 = 0; ridx1 < 128; ridx1 += 4) {
        float4 v0 = vload4(0, &input0[offset]);
        float4 v1 = vload4(0, &input1[offset]);
        vstore4(v0 + v1, 0, &output[offset]);
    }
    // テールループ (128は4で割り切れるので不要)
}
```

**対象演算**: Elementwise、FusedElementwise、FusedElementwiseReduce

**制約**:
- 最内軸のサイズがSIMD幅より小さい場合はSIMD化されません
- FusedElementwiseReduceは縮約軸が最内軸を含む場合はSIMD化されません（水平加算が必要なため）

## モジュール構成

### コアモジュール
- `mod.rs`: Graph、GraphNode、DType等の基本データ構造
- `ops.rs`: GraphOp定義と基本的な演算
- `node_view_ops.rs`: GraphNodeのView操作メソッド（unfold/fold統一API、contiguous、自動contiguous化）
- `strategy.rs`: 並列化戦略の定義
- `visualization.rs`: DOT形式でのグラフ可視化

### 高レベル演算
- `hlops.rs`: 高レベル演算のヘルパー関数
- `hlops_conv.rs`: 畳み込み演算（conv/conv_transpose統一API、次元数は入力形状から自動判定）

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

## サブグラフ機能

サブグラフ機能により、グラフを関数のように呼び出すことができます。

### 構造

- **Graph.subgraphs**: `HashMap<String, Graph>` - サブグラフを名前で管理
- **GraphOp::SubgraphCall**: サブグラフ呼び出しを表すノード
- **GraphOp::SubgraphOutput**: 複数出力サブグラフから特定の出力を取り出すノード

### DSL構文

- 動的Shape変数を使用する場合は、`graph<変数名=デフォルト値>` の形式で宣言が必須
- 変数への代入は `let` キーワードなしで直接行う
- 各グラフの最後には `return` 文が必須（単一出力: `return x`、複数出力: `return a, b`）

```harp
// サブグラフ定義（動的Shape変数B, Nにデフォルト値を指定）
graph<B=1, N=1> relu(x: f32[B, N]) -> (y: f32[B, N]) {
    zero = 0.0
    result = max(x, zero)
    return result
}

// メイングラフからサブグラフを呼び出し
graph<B=1, D=1> main(input: f32[B, D]) -> (output: f32[B, D]) {
    result = relu(input)
    return result
}

// 複数出力サブグラフのタプル分解
graph<N=1> multi_output(x: f32[N]) -> (a: f32[N], b: f32[N]) {
    r1 = x + 1.0
    r2 = x * 2.0
    return r1, r2
}

graph main(input: f32[10]) -> (out1: f32[10], out2: f32[10]) {
    (a, b) = multi_output(input)
    return a, b
}
```

### エントリーポイント

- `graph main`がエントリーポイントとして必須
- mainグラフ以外はすべてサブグラフとして扱われる
- 再帰呼び出しは許可（デフォルト最大深度: 10）

### 予約語

以下の識別子は変数名やグラフ名として使用できない：
- `graph`, `return`: 構文キーワード
- `fused`, `fused_reduce`, `fused_cumulative`: fused演算の専用構文
- `true`, `false`: 真偽値リテラル（将来的な使用のため予約）

### Decompile

`decompile()`関数はGraphをDSLソースコードに変換する。出力されるグラフは常に`main`という名前になる（エントリーポイントとして扱われるため）。

### 最適化

サブグラフは各グラフが独立して最適化されます。`SubgraphCall`と`SubgraphOutput`ノードはLoweringSuggesterではスキップされ、バックエンドで関数呼び出しとして処理されます。

## 未実装

- Thread/ThreadGroupレベルの並列実行のLowering
- ループタイル化（TilingSuggester）
- 行列乗算（matmul、batch_matmul）
- サブグラフ呼び出しのバックエンド実装（現在はグラフ構造のみ）
