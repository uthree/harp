# 計算グラフ

テンソル（多次元配列）単位での演算をDAGで表現する。

## 概要

計算グラフはテンソル演算をDAG（有向非巡回グラフ）として表現します。各ノードは演算または入力データを表し、エッジはデータの流れを表します。

## 設計方針

### GraphOpの設計
GraphOpは最適化の段階で最終的に融合されるため、**最適化よりも演算子の種類を減らすこと**を重視しています。例えば、減算は`Add`と`Neg`を組み合わせて表現します。

並列化戦略（`axis_strategies`）は各GraphOpバリアントの一部として保持され、演算の種類と並列化戦略が密接に関連付けられます。

### 並列化戦略（AxisStrategy）
各軸の並列化方法を制御します：
- **Sequential**: 逐次実行
- **Thread**: スレッドレベル並列化
- **ThreadGroup**: GPU向けスレッドグループ/ブロック並列化

各戦略は`simd_width`パラメータを持ち、SIMD化の有無と幅を制御します（`1`でSIMD化なし、`>=2`でSIMD化）。これにより並列化とSIMDベクトル化を独立制御できます。

### Cumulative演算の並列化
Cumulative演算（累積和、累積積など）は逐次依存性が高い演算ですが、**Parallel Scan（Prefix Sum）アルゴリズム**を用いることで効率的に並列化できます。

将来的な実装方針：
- **Sequential版**: 単純なループによる逐次累積
- **Thread/ThreadGroup版**: Work-Efficient Parallel Scanアルゴリズムを実装
  - Up-sweep（reduce）フェーズ: ツリー構造で部分和を計算
  - Down-sweep（distribute）フェーズ: 部分和を分配して最終結果を得る
  - 共有メモリを活用した効率的な実装

この設計により、O(n)の計算量をO(log n)の並列ステップで実現できます。

## View（重要な設計）

Viewは各軸の添え字からメモリオフセットへの線形変換を表現し、**ゼロコストの転置・次元操作**を実現します。現在は線形変換（shape、strides、offset）のみをサポート。

主なView操作：転置（permute）、次元追加/削除（unsqueeze/squeeze）、反転（flip）、拡張（expand）など。

なお、入出力のバッファーのViewは常にContiguousである必要があります。

## Shape変換の方針

**明示的なshape変換のみを許可**：演算を行う2つのノードは完全に同じshapeである必要があり、異なる場合は実行時にpanicします。ただし、**スカラー（ndim=0）は任意のテンソルにブロードキャスト可能**です。

これにより：
- **明示性**: shape変換を全て明示的に記述
- **安全性**: 意図しないbroadcastによるバグを防止
- **拡張性**: 将来的に`expand()`、`broadcast_to()`などを追加しやすい
- **利便性**: スカラー定数との演算は自然に記述可能

## DType推論

演算時に自動的にDTypeが推論されます（両方同じ→そのDType、片方Unknown→もう片方、異なる→Unknown）。

## 演算子オーバーロードと数値型変換

GraphNodeは直感的な数式記法をサポートします。

### 基本的な演算子
`+`, `-`, `*`, `/`, `%`などの演算子が`Into<GraphNode>`を受け取るため、GraphNode同士だけでなく数値型も直接使用可能です：

```rust
let mut graph = Graph::new();
let x = graph.input("x").with_dtype(DType::F32).with_shape([4]).build();

// GraphNode op 数値
let result = x.clone() * 2.0f32 + 1.0f32;

// 数値 op GraphNode（逆演算子も実装済み）
let result = 2.0f32 * x.clone() + 1.0f32;

// 複雑な式
let normalized = (x.clone() - 0.5f32) / 0.5f32;
let scaled = 1.0f32 / x;  // 逆数
```

### 参照ベースの演算子
`.clone()`を避けるため、参照版の演算子も実装されています：

```rust
let x = graph.input("x").with_dtype(DType::F32).with_shape([4]).build();

// &GraphNode op numeric（cloneが不要）
let result = &x * 2.0f32 + 1.0f32;

// numeric op &GraphNode
let scaled = 2.0f32 * &x;

// &GraphNode op &GraphNode（x * xのような式）
let squared = &x * &x;

// xは消費されないので再利用可能
let final_result = &x + 100.0f32;
```

### 対応する数値型
- `f32` → `GraphNode::constant(f32)` → DType::F32
- `isize`, `i32`, `i64` → `GraphNode::constant(isize)` → DType::Unknown
- `&GraphNode` → clone

スカラー定数はndim=0のテンソルとして扱われ、任意のshapeのテンソルにブロードキャストされます。

## 実装状況

### 実装済み
- 基本データ構造（Graph、GraphNode、DType）
- 入出力管理（InputNodeBuilder）
- Elementwise演算（Add、Mul、Neg、Max、Rem、Idiv、Recip）+ 演算子オーバーロード
- Reduce演算（Sum、Product、Max）
- Cumulative演算（cumsum、cumprod）とそのLowering
- Contiguous演算（転置、反転などの実体化）
- View操作（permute、unsqueeze、squeeze、flip、expand）
- Shape/DType推論
- 並列化戦略の定義（ElementwiseStrategy、ReduceStrategy、CumulativeStrategy）
- 融合演算（FusedElementwise、FusedElementwiseReduce、FusedReduce）
- グラフ最適化（詳細は[opt-graph.md](opt-graph.md)を参照）

### 未実装
- Thread/ThreadGroupレベルの並列実行のLowering
- ループタイル化（TilingSuggester - view変更の操作が必要）
- 畳み込み、行列乗算などの高度な演算
