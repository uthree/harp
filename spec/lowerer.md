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
- `cumulative.rs`, `fold.rs`: 将来の拡張用スケルトン
- `fused_elementwise.rs`: FusedElementwise演算のlowering（195行）
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

### 融合演算

#### FusedElementwise
複数のelementwise演算を1つのカーネルに融合。
- 全入力をロード→演算チェーンを順に評価→最後の結果をストア
- 中間バッファを削減し、メモリアクセスを削減

#### FusedElementwiseReduce
elementwise演算とそれに続くreduce演算を融合。
- 出力軸でループ（`oidx{i}`）→縮約軸でループ（`ridx{reduce_axis}`）
- 各反復でelementwise演算チェーンを評価→アキュムレート
- スカラー出力と指定軸縮約の両方に対応
- インデックス管理が複雑（`oidx` + `ridx`の組み合わせ）

#### FusedReduce
複数のreduce演算を同じ軸で融合（将来の拡張）。
- 注: 複数の出力が必要なため、tuple出力のサポートが必要

## 実装状況

### 実装済み
- Elementwise、Reduce、Contiguous演算
- **融合演算**: FusedElementwise、FusedElementwiseReduce
- トポロジカルソート（Kahn）
- 動的shape対応のシグネチャ生成
- ループアンローリング（`unroll_factor`）
- **中間バッファー管理**: main関数でのtmp{n}の自動確保・解放
- **main関数生成**: カーネルの順次呼び出しとバッファーマッピング
- ファイル分割による保守性向上（1176行 → 203行 + 個別ファイル）

### 未実装
- Thread/ThreadGroupレベルの並列化のコード生成
- SIMD化のコード生成（simd_width=1のみ）
- Cumulative演算のlowering
- FusedReduce演算（tuple出力が必要）
- AST最適化後のmain関数更新（関数インライン化との整合性）
- バリア同期の挿入（同世代カーネルの並列実行サポート）