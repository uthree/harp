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

### サポートされるDType
- `Bool`: ブール型（attention maskなど向け）。内部的には8ビット整数で表現
- `F32`: 32ビット浮動小数点
- `Complex`: 複素数型（FFTなど向け）。Lowering時にインターリーブF32バッファ`[re, im, ...]`に分解
- `Unknown`: 型推論前の未確定型

### Complex型の設計
複素数型はGraph層で`DType::Complex`として表現され、Lowering時にインターリーブレイアウトの`F32`バッファに分解されます。

#### メモリレイアウト
複素数配列 `[z0, z1, z2, ...]` はメモリ上で以下のようにインターリーブされます：
```
[re0, im0, re1, im1, re2, im2, ...]
```

このレイアウトにより：
- キャッシュ局所性の向上（実部と虚部が隣接）
- シンプルなバッファ管理（複素数テンソルごとに単一バッファ）
- SIMD演算との親和性

#### コード生成例
```c
// 複素数加算のカーネル
void kernel_0(const float* input0, const float* input1, float* output) {
    for (int i = 0; i < n; i++) {
        output[i * 2]     = input0[i * 2]     + input1[i * 2];     // 実部
        output[i * 2 + 1] = input0[i * 2 + 1] + input1[i * 2 + 1]; // 虚部
    }
}
```

#### 使用例
```rust
// 複素数定数の作成
let z1 = GraphNode::complex_constant(1.0, 2.0);  // 1.0 + 2.0i
let z2 = GraphNode::complex_constant_from((3.0, 4.0));  // 3.0 + 4.0i
let z3: GraphNode = (5.0f32, -1.0f32).into();  // タプルからの変換

// 複素数演算（実装済み）
let sum = &z1 + &z2;  // 複素数加算: (a+bi) + (c+di) = (a+c) + (b+d)i
let prod = &z1 * &z2;  // 複素数乗算: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
let neg = -&z1;        // 複素数否定: -(a+bi) = -a + (-b)i
let inv = z1.recip();  // 複素数逆数: 1/(a+bi) = (a-bi)/(a²+b²)
```

**実装状況**: 複素数Elementwise演算（Add, Mul, Neg, Recip）のLoweringは実装済みです。Reduce、Cumulative等の複素数演算は未実装です。

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
- `bool` → `GraphNode::constant(bool)` → DType::Bool
- `f32` → `GraphNode::constant(f32)` → DType::F32
- `(f32, f32)` → `GraphNode::complex_constant(re, im)` → DType::Complex
- `isize`, `i32`, `i64` → `GraphNode::constant(isize)` → DType::Unknown
- `&GraphNode` → clone

スカラー定数はndim=0のテンソルとして扱われ、任意のshapeのテンソルにブロードキャストされます。

## コード構成とリファクタリング履歴

### 2025-11-25: Expr型ヘルパーメソッド追加とテスト分離
コードの可読性と保守性を向上させるリファクタリングを実施：

#### Expr型のヘルパーメソッド追加
`src/graph/shape/expr.rs`に定数値抽出用のヘルパーメソッドを追加：
- `as_const() -> Option<isize>`: 定数値をOptionで取得
- `as_usize() -> Option<usize>`: 定数値を非負整数として取得
- `expect_const(msg: &str) -> isize`: 定数値を取得（失敗時はパニック）
- `expect_usize(msg: &str) -> usize`: 非負整数を取得（失敗時はパニック）

**効果**: コードベース全体で35箇所以上あった以下のようなパターンを統一：
```rust
// Before
let value = match &shape[i] {
    crate::graph::shape::Expr::Const(v) => *v as usize,
    _ => panic!("requires constant"),
};

// After
let value = shape[i].expect_usize("requires constant");
```

#### graph/mod.rsからテストを分離
- **mod.rs (1678行) → mod.rs (1082行) + tests/graph_tests.rs (596行)**
  - 実装とテストコードを分離
  - 59個のテストを独立したファイルに移動
  - テストの可読性と管理性が向上

全725テスト（676統合テスト + 49 doctest）が合格し、既存機能に影響なく完了。

### 2025-11-24: モジュール分割リファクタリング
大きなファイルを機能ごとに分割し、可読性と保守性を向上させました：

- **view.rs (1294行) → view.rs (600行) + view_ops.rs (705行)**
  - 基本的なView操作とunfold操作を分離
  - テストもview_ops.rsに移動

- **hlops.rs (1222行) → hlops.rs (746行) + hlops_conv.rs (452行)**
  - 畳み込み演算を独立したモジュールに分離
  - 高レベル演算の見通しが改善

全630テストが合格し、既存機能に影響なく分割完了。

## モジュール構成

計算グラフ関連のコードは以下のように分割されています：

### コアモジュール
- `mod.rs`: Graph、GraphNode、DType等の基本データ構造
- `ops.rs`: GraphOp定義と基本的な演算（Elementwise、Reduce、Cumulative等）
- `node_view_ops.rs`: GraphNodeのView操作メソッド（permute、unsqueeze、squeeze等）
- `strategy.rs`: 並列化戦略の定義（ElementwiseStrategy、ReduceStrategy、CumulativeStrategy）
- `visualization.rs`: DOT形式でのグラフ可視化

### 高レベル演算
- `hlops.rs`: 高レベル演算のヘルパー関数（square、powi、mean、variance、数学関数等）
- `hlops_conv.rs`: 畳み込み演算（conv1d、conv2d、conv3d）

### Shape関連
- `shape/mod.rs`: Shape関連モジュールの定義
- `shape/expr.rs`: シンボリック式（Expr）の定義と演算
- `shape/view.rs`: View構造体と基本操作（contiguous、permute、reshape等）
- `shape/view_ops.rs`: View unfold操作（unfold1d、unfold2d、unfold3d）
- `shape/tests.rs`: Shape関連のテスト

この分割により、機能ごとに独立したモジュールとなり、可読性と保守性が向上しています。

## 実装状況

### 実装済み
- 基本データ構造（Graph、GraphNode、DType）
- 入出力管理（InputNodeBuilder）
- Elementwise演算（Add、Mul、Neg、Max、Rem、Idiv、Recip）+ 演算子オーバーロード
- Reduce演算（Sum、Product、Max）
- Cumulative演算（cumsum、cumprod）とそのLowering
- Contiguous演算（転置、反転などの実体化）
- View操作（permute、unsqueeze、squeeze、flip、expand、reshape）
- View unfold操作（unfold1d、unfold2d、unfold3d）- スライディングウィンドウでの畳み込み用
- Shape/DType推論
- 並列化戦略の定義（ElementwiseStrategy、ReduceStrategy、CumulativeStrategy）
- 融合演算（FusedElementwise、FusedElementwiseReduce、FusedElementwiseCumulative、FusedReduce）
- 高レベル演算（square、powi、mean、variance、数学関数）
- 畳み込み演算（conv1d、conv2d、conv3d）- unfold + reduce で実装
- テンソル結合（Concat）- torch.catに相当、複数テンソルを指定軸で結合
- テンソル切り出し（Slice）- テンソルの一部を切り出す
- パディング（Pad）- テンソルの境界を指定値で拡張
- グラフ最適化（詳細は[opt-graph.md](opt-graph.md)を参照）
- 複素数型（Complex）のElementwise演算Lowering（Add、Mul、Neg、Recip）- インターリーブF32バッファ`[re, im, ...]`

### 未実装
- Thread/ThreadGroupレベルの並列実行のLowering
- ループタイル化（TilingSuggester - view変更の操作が必要）
- 行列乗算（matmul、batch_matmul）- unsqueezeとbroadcastの実装待ち
- 複素数型のReduce/Cumulative演算のLowering
