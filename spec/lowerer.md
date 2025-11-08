# lowerer

## ファイル構成
- `src/lowerer/mod.rs` - lowerer本体の実装 (856行)
- `src/lowerer/tests.rs` - テストコード (632行、17テスト)

## 役割
計算グラフをASTに変換する。

## 手順
1. ノードをKahnのアルゴリズムを用いてトポロジカルソートと世代別のグループ分けを行う。同じ世代のノードは同時に計算することができる。
2. 各ノードが持つlowering戦略に従ってASTを生成、1ノードに対して一つのカーネル関数を生成する。
3. グラフ全体の処理を表す`kernel_main`関数に対して、それぞれのカーネルを順番に呼び出す。世代の区切りにのみバリア（同期）を挿入する。

## 実装状況

### 実装済み機能
- **Elementwise演算**: Add, Mul, Neg, Max, Rem, Idiv, Recip
  - 要素ごとの演算をネストしたループに変換
  - Viewを考慮したオフセット計算に対応
- **Reduce演算**: Add (Sum), Mul (Product), Max
  - Sequential版のみ実装済み（逐次実行）
  - スカラー出力（全縮約）と指定軸縮約の両方に対応
  - 1次元、2次元、3次元テンソルでテスト済み
- **Contiguous演算**: Viewに従った要素の並べ替え
  - Sequential版のみ実装済み（逐次実行）
  - 入力のViewに従ってメモリからロードし、出力のcontiguous Viewに従ってストア
  - 1次元（flip）、2次元（transpose）でテスト済み
- **トポロジカルソート**: Kahnのアルゴリズムによる世代別グループ化
- **シグネチャ生成**: 動的shapeに対応したKernelSignature生成

### 未実装機能
- **Cumulative演算**: 累積演算
- **並列化**: Thread/ThreadGroupレベルの並列実行（現在はSequentialのみ）
- **SIMD**: ベクトル化（現在はsimd_width=1のみ）
- **カーネル融合**: 複数ノードの融合最適化
- **グラフ全体のlowering**: kernel_main関数の生成

### エントリーポイントの必要性
libloadingライブラリは、堅安全のため動的ライブラリのシグネチャが固定であるため、以下のようなエントリーポイントとなる関数を作る必要がある。
```c
void entry_point(void** inputs, void** outputs, size_t *shape_vars) {
    // 入力バッファー群を展開
    input0 = (float*)*inputs[0];
    input1 = (float*)*inputs[1];
    
    // 出力バッファー群を展開
    output0 = (float*)*outputs[0];
    output1 = (float*)*outputs[1];

    // shape_vatsを展開
    size_t N = shape_vars[0];
    size_t M = shape_vars[1];

    // 処理の本体を呼び出し
    kernel_main(input0, input1, output0, output1, N, M);
}
```

## 命名法則
### スタックまたはレジスタ上の変数
tinygradを参考に命名する。`{n}`の部分は重複を回避するための自然数が入る。
- `lidx{n}`: グループ内のスレッドのIDを示す変数
- `gidx{n}`: グループ番号を示す変数
- `ridx{n}`: Rangeノード（for文）で使用するループカウンター（入力テンソルの軸インデックス）
- `oidx{n}`: 出力テンソルの軸インデックス（Reduce演算で使用）
- `alu{n}`: 一時的な（スカラーまたはSIMDベクタ）値を格納するための変数。スタック上にあるものを表す。
- `acc{n}`: アキュムレーター。累積(cumulative)や縮約(reduce)演算に使う。
- `shape{n}`: 各軸のサイズを格納する変数（カーネル関数のパラメータ）

### バッファー(メモリ上のサイズの大きな値)
`input{n}`: 入力バッファー
`output{n}`: 出力バッファー
`tmp{n}`: 一時的に確保されるバッファー

### 関数の命名
未定。
入出力のShapeにちなんだ名前にするか、ただの識別子にするか？

## Reduce演算のlowering

### 概要
Reduce演算は指定された軸を縮約して次元を減らす演算です。現在はSequential版のみ実装されており、以下の縮約演算をサポートしています：
- `ReduceOp::Add` - 合計（初期値: 0）
- `ReduceOp::Mul` - 積（初期値: 1）
- `ReduceOp::Max` - 最大値（初期値: -∞）

### 実装方針
1. **スカラー出力の場合**（全縮約）:
   - 全ての軸についてネストしたループを生成
   - アキュムレータを初期化し、各要素を順次アキュムレート
   - 最終結果をoutput[0]に書き込み

2. **テンソル出力の場合**（指定軸縮約）:
   - 出力の各軸についてループ（`oidx{n}`変数を使用）
   - 各出力位置でアキュムレータを初期化
   - 縮約軸についてループ（`ridx{n}`変数を使用）してアキュムレート
   - 結果を出力の対応位置に書き込み

### 生成されるコード例

#### 1次元テンソルの合計（スカラー出力）
入力: `[10]` → 出力: `[]`（スカラー）
```metal
void reduce_sum_kernel(const device float* input0, device float* output, const uint shape0) {
    auto alu0 = 0f;  // アキュムレータを初期化
    for (uint ridx0 = 0u; ridx0 < shape0; ridx0 += 1u) {
        auto alu0 = (alu0 + input0[(0 + (ridx0 * 1))]);
    }
    output[0u] = alu0;  // 結果を書き込み
}
```

#### 2次元テンソルの軸1方向の合計
入力: `[3, 4]` → 出力: `[3]`（軸1を縮約）
```metal
void reduce_sum_2d_kernel(const device float* input0, device float* output,
                          const uint shape0, const uint shape1) {
    for (uint oidx0 = 0u; oidx0 < shape0; oidx0 += 1u) {  // 出力軸0のループ
        auto alu0 = 0f;  // アキュムレータを初期化
        for (uint ridx1 = 0u; ridx1 < shape1; ridx1 += 1u) {  // 縮約軸1のループ
            auto alu0 = (alu0 + input0[((0 + (oidx0 * 4)) + (ridx1 * 1))]);
        }
        output[(0 + (oidx0 * 1))] = alu0;  // 結果を書き込み
    }
}
```

#### 3次元テンソルの中間軸縮約
入力: `[2, 3, 4]` → 出力: `[2, 4]`（軸1を縮約）
```metal
void reduce_3d_kernel(const device float* input0, device float* output,
                      const uint shape0, const uint shape1, const uint shape2) {
    for (uint oidx0 = 0u; oidx0 < shape0; oidx0 += 1u) {  // 出力軸0のループ
        for (uint oidx1 = 0u; oidx1 < shape2; oidx1 += 1u) {  // 出力軸1のループ（入力軸2に対応）
            auto alu0 = 0f;  // アキュムレータを初期化
            for (uint ridx1 = 0u; ridx1 < shape1; ridx1 += 1u) {  // 縮約軸1のループ
                auto alu0 = (alu0 + input0[(((0 + (oidx0 * 12)) + (oidx1 * 1)) + (ridx1 * 4))]);
            }
            output[((0 + (oidx0 * 4)) + (oidx1 * 1))] = alu0;
        }
    }
}
```

### 実装関数一覧
- `lower_reduce_kernel`: Reduce演算のカーネル関数生成のエントリーポイント
- `generate_reduce_loops`: ループ構造の生成（スカラー/テンソル出力を判定）
- `generate_reduce_to_scalar`: スカラー出力への全縮約コード生成
- `generate_reduce_body_with_axis`: 指定軸での縮約本体の生成
- `get_reduce_init_value`: 縮約演算の初期値を取得
- `generate_accumulate_statement`: アキュムレート文の生成
- `generate_accumulate_statement_with_reduce_axis`: 縮約軸を含むアキュムレート文の生成
- `apply_reduce_op`: 縮約演算のASTノード生成
- `compute_offset_for_input`: 入力のオフセット計算（ridx変数使用）
- `compute_offset_for_output`: 出力のオフセット計算（oidx変数使用）
- `compute_offset_for_input_with_reduce_axis`: 縮約軸を含む入力のオフセット計算

### 今後の課題
- **並列化**: Thread/ThreadGroupレベルでの並列実行
  - Threadレベル: 各スレッドがローカルアキュムレータを持ち、最後にアトミック操作でマージ
  - ThreadGroupレベル: 共有メモリを使ったツリー縮約が必要
- **SIMD化**: ベクトル化による高速化
- **最適化**: カーネル融合、メモリアクセスパターンの最適化

## Contiguous演算のlowering

### 概要
Contiguous演算は、非連続なメモリレイアウトを持つテンソルを連続したメモリレイアウトに変換します。主に転置（transpose）や反転（flip）などのView操作の後に使用されます。

### 実装方針
1. **入力と出力**:
   - 入力: 非連続なViewを持つテンソル（permute、flipなどの適用後）
   - 出力: 連続したメモリレイアウトを持つテンソル

2. **処理の流れ**:
   - 出力の各要素について、対応する入力の位置を計算
   - 入力のViewを使って入力オフセットを計算
   - 出力のViewを使って出力オフセットを計算
   - 入力から値をロードし、出力にストア

### 生成されるコード例

#### 1次元テンソルの反転（flip）
入力: `[10]` (flipped) → 出力: `[10]` (contiguous)
```metal
void contiguous_flip_kernel(const device float* input0, device float* output, const uint shape0) {
    for (uint ridx0 = 0u; ridx0 < shape0; ridx0 += 1u) {
        auto alu0 = input0[(9 + (ridx0 * -1))];  // 反転されたインデックス
        output[(0 + (ridx0 * 1))] = alu0;
    }
}
```

#### 2次元テンソルの転置（transpose）
入力: `[3, 4]` (transposed to [4, 3]) → 出力: `[4, 3]` (contiguous)
```metal
void contiguous_transpose_kernel(const device float* input0, device float* output,
                                  const uint shape0, const uint shape1) {
    for (uint ridx0 = 0u; ridx0 < shape0; ridx0 += 1u) {
        for (uint ridx1 = 0u; ridx1 < shape1; ridx1 += 1u) {
            auto alu0 = input0[((0 + (ridx0 * 1)) + (ridx1 * 4))];  // 転置されたインデックス
            output[((0 + (ridx0 * 3)) + (ridx1 * 1))] = alu0;
        }
    }
}
```

### 実装関数一覧
- `lower_contiguous_kernel`: Contiguous演算のカーネル関数生成のエントリーポイント
- `generate_contiguous_loops`: ループ構造の生成
- `generate_contiguous_body`: ループ本体の生成（ロード＆ストア）

### 今後の課題
- **並列化**: Thread/ThreadGroupレベルでの並列実行
- **SIMD化**: ベクトル化による高速化
- **最適化**: メモリアクセスパターンの最適化（特に転置の場合）