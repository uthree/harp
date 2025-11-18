# Lowerer

## 役割
計算グラフをASTに変換する層。

## Lowering手順
1. **トポロジカルソート**: Kahnのアルゴリズムで世代別にグループ化（同世代は並列実行可能）
2. **バッファーマッピング**: 各ノードの出力バッファー名を決定（`input{n}`, `tmp{n}`, `output`）
3. **カーネル生成**: 各ノードを1つのカーネル関数に変換（lowering戦略に従う）
4. **main関数生成**: 中間バッファーの確保・解放、カーネルの順次呼び出し

## 中間バッファー管理

複数のカーネルを持つパイプラインでは、中間結果を格納するバッファーが必要です。

**バッファー命名:**
- グラフ入力ノード → `input0`, `input1`, ...
- 中間ノード（非最終出力）→ `tmp0`, `tmp1`, ...
- グラフ出力ノード → `output`

**main関数の構造:**
```c
void main(const float* input0, ..., float* output) {
    // 1. 中間バッファーの宣言と確保
    float* tmp0 = (float*)malloc(size0 * sizeof(float));

    // 2. カーネルの順次呼び出し（依存関係順）
    kernel_0(input0, tmp0);        // input → tmp0
    kernel_1(tmp0, input1, output); // tmp0 + input1 → output

    // 3. 中間バッファーの解放
    free(tmp0);
}
```

**グラフ最適化との連携:**
グラフ最適化後、新しいノードが作成されてポインタが変わる可能性があるため、
入力・出力ノードの識別にはトポロジカルソート結果を使用します：
- 入力ノード: `GraphOp::Input`を持つノード
- 出力ノード: トポロジカルソートの最初の世代（出力→入力の順序）

**Viewノードの処理:**
Viewノード（`GraphOp::View`）はメモリアクセスパターンを記述するだけで、独自のバッファーを持ちません。
- lowering時にViewノードはスキップされる（カーネル生成なし）
- Viewノードをソースとして持つカーネルは、Viewの基底ストレージノードまでトレースバック
- Viewのstride情報はオフセット計算時に使用される

## 命名法則（tinygradを参考）

**変数:**
- `lidx{n}`, `gidx{n}`: スレッドID、グループ番号
- `ridx{n}`, `oidx{n}`: 入力軸インデックス、出力軸インデックス（Reduce用）
- `alu{n}`: 一時スカラー/SIMDベクタ
- `acc{n}`: アキュムレータ（reduce/cumulative用）
- `shape{n}`: 各軸のサイズ（パラメータ）

**バッファー:**
- `input{n}`, `output{n}`, `tmp{n}`: 入力、出力、一時バッファー

## ファイル構成（リファクタリング済み）

lowerer/mod.rsを演算タイプごとに分割：
- `mod.rs`: コア構造（Lowerer、トポロジカルソート、ディスパッチ）- 203行
- `utils.rs`: ユーティリティ関数（オフセット計算、型変換、シグネチャ生成）
- `elementwise.rs`: Elementwise演算のlowering
- `reduce.rs`: Reduce演算のlowering
- `contiguous.rs`: Contiguous演算のlowering
- `cumulative.rs`: Cumulative演算のlowering（累積和、累積積）
- `fold.rs`: 将来の拡張用スケルトン
- `fused_elementwise.rs`: FusedElementwise演算のlowering（195行）
- `fused_elementwise_cumulative.rs`: FusedElementwiseCumulative演算のlowering
- `fused_elementwise_reduce.rs`: FusedElementwiseReduce演算のlowering（405行）
- `fused_reduce.rs`: 将来の拡張用スケルトン

## 演算のLowering方針

### Elementwise演算
要素ごとの演算をネストループに変換。Viewを考慮したオフセット計算を行います。
- ループアンローリングサポート（`unroll_factor`）

### Reduce演算
指定軸を縮約する演算。スカラー出力（全縮約）と指定軸縮約に対応。
- スカラー出力: 全軸でループ→アキュムレート→結果を`output[0]`に書き込み
- テンソル出力: 出力軸でループ→各位置でアキュムレート→結果を書き込み

### Contiguous演算
非連続なView（転置、反転など）を連続メモリレイアウトに変換。入力Viewでロード→出力（contiguous）にストア。

### Cumulative演算
累積軸に沿った累積和（cumsum）・累積積（cumprod）を生成。
- 累積軸以外の軸でネストループを生成
- 内側でアキュムレータを初期化し、累積軸に沿ってループ
- 各反復で入力値をロード→アキュムレータを更新→結果をストア

```c
for (int idx0 = 0; idx0 < shape0; idx0 += 1) {
    float acc0 = 0f;  // cumsum: 0, cumprod: 1
    for (int cumidx1 = 0; cumidx1 < shape1; cumidx1 += 1) {
        float alu0 = input0[idx0 * stride0 + cumidx1 * stride1];
        acc0 = acc0 + alu0;  // or * for cumprod
        output[idx0 * out_stride0 + cumidx1 * out_stride1] = acc0;
    }
}
```

### 融合演算

融合演算は **AstNode式**を使用して、elementwise演算パターンを表現します。
`Wildcard("0")`, `Wildcard("1")` 等が `src[0]`, `src[1]` に対応し、`substitute()` メソッドで実際の入力値に置き換えられます。

#### FusedElementwise
複数のelementwise演算を1つのカーネルに融合。
- 全入力をロード→AstNode式のWildcardを置き換え→結果をストア
- 中間バッファを削減し、メモリアクセスを削減

```rust
use harp::ast::helper::wildcard;
// (a + b) * c を融合
let expr = (wildcard("0") + wildcard("1")) * wildcard("2");
let result = fused_elementwise(vec![a, b, c], expr);
```

#### FusedElementwiseReduce
elementwise演算とそれに続くreduce演算を融合。
- 出力軸でループ（`oidx{i}`）→縮約軸でループ（`ridx{reduce_axis}`）
- 各反復でAstNode式を評価→アキュムレート
- スカラー出力と指定軸縮約の両方に対応
- インデックス管理が複雑（`oidx` + `ridx`の組み合わせ）

```rust
// reduce_sum(a * b, axis=0)
let expr = wildcard("0") * wildcard("1");
let result = fused_elementwise_reduce(vec![a, b], expr, ReduceOp::Sum, 0);
```

#### FusedElementwiseCumulative
elementwise演算とそれに続く累積演算を融合。
- 累積軸以外の軸でループ（`idx{i}`）→累積軸でループ（`cumidx{axis}`）
- 各反復でAstNode式を評価→アキュムレータ更新→結果書き込み
- 出力shapeは入力と同じ（reduceとは異なり軸を消さない）

```rust
// cumsum(x^2)
let expr = wildcard("0") * wildcard("0");  // 二乗
let result = fused_elementwise_cumulative(vec![x], expr, CumulativeOp::Sum, 1);
```

生成されるコード例：
```c
// 例: cumsum(x^2)
for (int idx0 = 0; idx0 < shape0; idx0 += 1) {
    float acc0 = 0f;
    for (int cumidx1 = 0; cumidx1 < shape1; cumidx1 += 1) {
        float alu0 = input0[idx0 * stride0 + cumidx1 * stride1];
        float alu1 = alu0 * alu0;  // elementwise演算（AstNode式の評価結果）
        acc0 = acc0 + alu1;        // cumulative演算
        output[idx0 * out_stride0 + cumidx1 * out_stride1] = acc0;
    }
}
```

**行列積のサポート:**
FusedElementwiseReduceを使用して行列積を実装できます：
```rust
// A[M, K] @ B[K, N] -> C[M, N]
// 1. Aを[M, 1, K]に拡張、Bを転置して[N, K]、[1, N, K]に拡張
// 2. 両者を[M, N, K]にブロードキャスト
// 3. 要素積 + 軸2でsum reduce
let a_expanded = a.view(...unsqueeze(1).expand([M, N, K]));
let b_transposed = b.view(...permute([1, 0]));
let b_t_expanded = b_transposed.view(...unsqueeze(0).expand([M, N, K]));
let expr = wildcard("0") * wildcard("1");
fused_elementwise_reduce(
    vec![a_expanded, b_t_expanded],
    expr,
    ReduceOp::Sum,
    axis=2
)
```
生成されるコード（最適化前）：
```c
for (int oidx0 = 0; oidx0 < M; oidx0 += 1) {
    for (int oidx1 = 0; oidx1 < N; oidx1 += 1) {
        float acc0 = 0f;
        for (int ridx2 = 0; ridx2 < K; ridx2 += 1) {
            acc0 += A[oidx0*K + ridx2] * B[oidx1 + ridx2*N];
        }
        C[oidx0*N + oidx1] = acc0;
    }
}
```

#### FusedReduce
複数のreduce演算を同じ軸で融合（将来の拡張）。
- 注: 複数の出力が必要なため、tuple出力のサポートが必要

## 実装状況

### 実装済み
- Elementwise、Reduce、Contiguous演算
- **Cumulative演算**: cumsum（累積和）、cumprod（累積積）
- **融合演算**: FusedElementwise、FusedElementwiseReduce、FusedElementwiseCumulative
- トポロジカルソート（Kahn）
- 動的shape対応のシグネチャ生成
- ループアンローリング（`unroll_factor`）
- **中間バッファー管理**: main関数でのtmp{n}の自動確保・解放
- **main関数生成**: カーネルの順次呼び出しとバッファーマッピング
- **Viewノード処理**: View操作をスキップし、基底ストレージまでトレースバック
- **行列積サポート**: View展開 + FusedElementwiseReduceによるmatmul
- ファイル分割による保守性向上（1176行 → 203行 + 個別ファイル）

### 未実装
- SIMD化のコード生成（simd_width=1のみ）
- FusedReduce演算（tuple出力が必要）
- AST最適化後のmain関数更新（関数インライン化との整合性）
- OpenCLバックエンドでの並列化コード生成（Kernel関数ボディの実装）

### 既知の問題
- **AST最適化の変数スコープバグ**: 関数インライン化後に変数宣言が適切に伝播されない場合がある（重複宣言が生成されることがある）。