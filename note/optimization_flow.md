# 最適化とコンパイルのフロー

`harp`ライブラリにおける計算の実行は、`Tensor::realize()`メソッドの呼び出しを起点とする遅延評価モデルに基づいています。このメソッドが呼び出されると、高レベルな`Tensor`の計算グラフは、一連の変換パイプラインを経て、最終的に実行可能なネイティブコードへと変換されます。

以下に、その主要なフローを示します。

## 1. `Tensor` -> `UOp` への低レベル化 (Lowering)

- **コンポーネント**: `Lowerizer` (`src/lowerizer.rs`)
- **入力**: `Tensor`の計算グラフ
- **出力**: `UOp` (Micro-operation) の計算グラフ

パイプラインの最初のステップは、ユーザーが定義した`Tensor`の操作（加算、乗算など）を、よりハードウェアに近い、低レベルな`UOp`のグラフに変換することです。`Lowerizer`は`Tensor`グラフを再帰的にたどり、各`TensorOp`に対応する`UOp`のノード（`Op::Add`, `Op::Mul`, `Op::Load`など）を構築します。

この段階で、メモリ上のデータアクセス（`Load`）やインデックス計算も`UOp`として明示的に表現されます。

## 2. `UOp`グラフの最適化 (Optimization)

- **コンポーネント**: `Optimizer` (`src/optimizer.rs`)
- **入力**: `UOp`グラフ
- **出力**: 最適化された`UOp`グラフ

次に、生成された`UOp`グラフは`Optimizer`に渡されます。現在の`Optimizer`は、パターンマッチングに基づいた代数的な簡約化を主に行います。

定義済みのルール（例: `x + 0 => x`, `x * 1 => x`）のセットを`UOp`グラフに繰り返し適用し、冗長な計算を除去します。この処理は、グラフに変化がなくなる（固定点に達する）まで続けられます。

```rust
// src/optimizer.rs 内のルール定義例
pats!({
    (a) | &a + &UOp::from(0.0f32) => a,
    (a) | &a * &UOp::from(1.0f32) => a,
    // ...
})
```

## 3. `UOp`グラフの線形化 (Linearization)

- **コンポーネント**: `Linearizer` (`src/linearizer.rs`)
- **入力**: 最適化された`UOp`グラフ
- **出力**: 線形化された`UOp`命令のリスト (`Vec<UOp>`)

グラフ構造は計算の依存関係を表現するには適していますが、Cコードのような手続き的な言語を生成するには不向きです。そこで`Linearizer`は、`UOp`グラフを平坦な命令のシーケンスに変換します。

このプロセスでは、計算結果を一時変数に格納する`Store`命令が導入され、ループ構造（`Op::LoopStart`, `Op::LoopEnd`）が明示的に挿入されます。これにより、最終的なコード生成が単純な命令のイテレーションで実現できるようになります。

## 4. コード生成とコンパイル (Code Generation & Compilation)

- **コンポーネント**: `Backend` (例: `ClangCompiler` in `src/backends/c/compiler.rs`)
- **入力**: 線形化された`UOp`命令リスト
- **出力**: 実行可能なカーネル (`dyn Kernel`)

最後に、線形化された`UOp`命令リストは、選択された`Backend`に渡されます。

1. **レンダリング**: `Backend`は`UOp`命令を一つずつ解釈し、ターゲット言語のソースコード（現在はC言語）を生成します。
2. **コンパイル**: 生成されたCソースコードは、`clang`のような外部コンパイラを使って共有ライブラリ（`.so`ファイル）としてコンパイルされます。この際、最適化レベル（`-O3`など）を指定できます。
3. **実行**: コンパイルされた共有ライブラリは動的にロードされ、その中のカーネル関数が実行されて最終的な計算結果が生成されます。

## 全体の流れ in `Tensor::realize()`

```rust
// In src/tensor.rs, Tensor::realize()

// 1. Lowering
let mut lowerizer: Lowerizer<T> = Lowerizer::new();
let uop_graph = lowerizer.lower(self);

// 2. Optimization
let optimizer = Optimizer::new();
let optimized_uop_graph = optimizer.optimize(&uop_graph);

// 3. Linearization
let mut linearizer = Linearizer::new();
let kernel_instructions = linearizer.linearize(&optimized_uop_graph, self.shape());

// 4. Compilation & Execution
self.0.backend.compile_and_exec(&kernel_instructions, ...);
```

このパイプラインにより、高レベルなAPIで定義されたテンソル計算が、実行時に効率的なネイティブコードへと自動的に変換されます。
