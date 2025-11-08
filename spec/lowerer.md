# Lowerer

## 役割
計算グラフをASTに変換する層。

## Lowering手順
1. **トポロジカルソート**: Kahnのアルゴリズムで世代別にグループ化（同世代は並列実行可能）
2. **カーネル生成**: 各ノードを1つのカーネル関数に変換（lowering戦略に従う）
3. **kernel_main生成**: カーネルを順次呼び出し、世代の区切りにバリア同期を挿入（未実装）

## 命名法則（tinygradを参考）

**変数:**
- `lidx{n}`, `gidx{n}`: スレッドID、グループ番号
- `ridx{n}`, `oidx{n}`: 入力軸インデックス、出力軸インデックス（Reduce用）
- `alu{n}`: 一時スカラー/SIMDベクタ
- `acc{n}`: アキュムレータ（reduce/cumulative用）
- `shape{n}`: 各軸のサイズ（パラメータ）

**バッファー:**
- `input{n}`, `output{n}`, `tmp{n}`: 入力、出力、一時バッファー

## 演算のLowering方針

### Elementwise演算
要素ごとの演算をネストループに変換。Viewを考慮したオフセット計算を行います。

### Reduce演算
指定軸を縮約する演算。スカラー出力（全縮約）と指定軸縮約に対応。
- スカラー出力: 全軸でループ→アキュムレート→結果を`output[0]`に書き込み
- テンソル出力: 出力軸でループ→各位置でアキュムレート→結果を書き込み

### Contiguous演算
非連続なView（転置、反転など）を連続メモリレイアウトに変換。入力Viewでロード→出力（contiguous）にストア。

## 実装状況

### 実装済み
- Elementwise、Reduce、Contiguous演算（Sequential版のみ）
- トポロジカルソート（Kahn）
- 動的shape対応のシグネチャ生成

### 未実装
- Thread/ThreadGroupレベルの並列化
- SIMD化（simd_width=1のみ）
- Cumulative演算
- カーネル融合
- kernel_main関数生成