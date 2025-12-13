# 既知の問題

## ループタイリング時の変数未宣言バグ

**発見日**: 2024-12

**状態**: 未修正

**症状**:
AST最適化でループタイリング(`LoopTilingSuggester`)が適用された際、タイル化されたループ変数（例: `ridx2`）が宣言される前に使用されるコードが生成される。

**再現条件**:
- `RuntimeSelector`による実測値ベース最適化を有効化
- 1024x1024程度の大きな行列積を最適化
- タイリングが適用される候補が選択された場合に発生

**エラー例**:
```
OpenCL build error:
program_source:11:21: error: use of undeclared identifier 'ridx2'
                    ridx2 = (ridx2_outer + ridx2_inner);
                    ^
```

**生成される問題のあるコード**:
```c
for (int ridx2_outer = 0; ridx2_outer < 1024; ridx2_outer += 128) {
    for (int ridx2_inner = 0; ridx2_inner < 128; ridx2_inner += 1) {
        ridx2 = (ridx2_outer + ridx2_inner);  // ridx2が未宣言
        acc = (acc + (input0[...ridx2...] * input1[...ridx2...]));
    }
}
```

**期待されるコード**:
```c
for (int ridx2_outer = 0; ridx2_outer < 1024; ridx2_outer += 128) {
    for (int ridx2_inner = 0; ridx2_inner < 128; ridx2_inner += 1) {
        int ridx2 = (ridx2_outer + ridx2_inner);  // 宣言が必要
        acc = (acc + (input0[...ridx2...] * input1[...ridx2...]));
    }
}
```

**影響範囲**:
- `src/opt/ast/suggesters/loop_tiling.rs`
- OpenCL/C/Metalレンダラー全般

**回避策**:
`set_runtime_buffer_factory()`を呼び出さず、静的コスト推定のみで最適化を行う。

**調査ポイント**:
1. `LoopTilingSuggester`がタイル化時に元の変数宣言を保持していない可能性
2. `AstNode::Assign`の左辺が新規変数の場合の宣言生成ロジック
