# lowerer
## 役割
計算グラフをASTに変換する。

## 手順
TODO: loweringの手順を決める

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
    kernel(input0, input1, output0, output1, N, M);
}
```

## 命名法則
### スタックまたはレジスタ上の変数
tinygradを参考に命名する。`{n}`の部分は重複を回避するための自然数が入る。
`lidx{n}`: グループ内のスレッドのIDを示す変数
`gidx{n}`: グループ番号を示す変数
`ridx{n}`: Rangeノード（for文）で使用するループカウンター
`alu{n}`: 一時的な（スカラーまたはSIMDベクタ）値を格納するための変数。スタック上にあるものを表す。
`acc{n}`: アキュムレーター。累積(cumulative)や縮約(reduce)演算をに使う。
TODO: 動的shapeのための変数の命名をどうするか決める

### バッファー(メモリ上のサイズの大きな値)
`input{n}`: 入力バッファー
`output{n}`: 出力バッファー
`tmp{n}`: 一時的に確保されるバッファー

### 関数の命名
未定。
入出力のShapeにちなんだ名前にするか、ただの識別子にするか？