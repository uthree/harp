# harp-lazy-array: 配列計算クレート

ndarray/PyTorchライクなAPIで配列計算を行うクレート。遅延評価による計算グラフの構築と、キャッシュ機構による効率的なカーネル再利用を提供する。

## 設計方針

- **遅延評価**: 演算はグラフとして構築され、必要になったときに実行
- **キャッシュ**: 同じ計算グラフは再コンパイルせずに再利用
- **型付け次元**: `Array<T, Dim2, ...>` など、次元数をコンパイル時に検証
- **動的次元**: `DimDyn` で実行時に次元数が決まる配列もサポート
- **バックエンド抽象化**: ジェネリクスで`Device`/`Compiler`を受け取り、具体的な実装は外部から注入

## モジュール構成

```
crates/lazy-array/src/
├── lib.rs              # モジュール定義、型エイリアス、prelude
├── dim.rs              # 次元トレイト (Dim0-Dim6, DimDyn)
├── dyn_backend.rs      # Array<T, D> + ArrayState (遅延/評価済み)
├── device.rs           # Device抽象化
├── cache.rs            # ProgramCache, CacheStats
├── execution.rs        # OpenCL/Metal ExecutionContext
└── generators.rs       # zeros, ones, full, arange, rand
```

## 主要な型

### Dimension トレイト (`dim.rs`)

次元数を表すトレイト。静的次元と動的次元の両方をサポート。

- `Dim0` - `Dim6`: 静的次元（コンパイル時に次元数が確定）
- `DimDyn`: 動的次元（実行時に次元数が決まる）

```rust
pub trait Dimension: Clone + fmt::Debug + 'static {
    const NDIM: Option<usize>;  // 静的次元数（動的の場合はNone）
    fn ndim(&self) -> usize;     // 実行時の次元数
}
```

### Array (`array.rs`)

遅延評価を透過的に扱う配列型。`ArrayState`で未評価（グラフのみ）と評価済み（バッファあり）を管理。

```rust
pub struct Array<T, D, R, Dev, Comp, Buf> {
    ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
    state: RefCell<ArrayState<Buf>>,
    shape: Vec<usize>,
}

pub enum ArrayState<Buf> {
    Lazy { graph_node: GraphNode },
    Materialized { buffer: Arc<Buf>, graph_node: Option<GraphNode> },
}
```

### ExecutionContext (`context.rs`)

バックエンドを抽象化し、計算グラフのコンパイルと実行を管理。

- `Pipeline`のラッパーとして機能
- `KernelCache`でコンパイル済みカーネルを再利用
- 設定は`ExecutionConfig`で管理

### ProgramCache (`cache.rs`)

コンパイル済みプログラムを再利用するためのキャッシュ機構。

- **キャッシュキー**: `harp_dsl::decompile(&graph)` の出力文字列（決定論的）
- **ヒット/ミス統計**: `CacheStats` で追跡
- **ExecutionContext統合**: OpenCL/Metal実行コンテキスト内でキャッシュを管理

## 生成関数 (`generators.rs`)

| 関数 | 説明 |
|------|------|
| `zeros(ctx, shape)` | ゼロ初期化配列（f32） |
| `ones(ctx, shape)` | 1初期化配列（f32） |
| `full_f32(ctx, shape, value)` | 指定値初期化（f32） |
| `arange(ctx, size)` | 連番配列 [0, 1, ..., size-1]（i32） |
| `rand(ctx, shape)` | 一様乱数配列 [0, 1)（f32） |
| `zeros_like(arr)` | 同形状のゼロ配列 |
| `ones_like(arr)` | 同形状の1配列 |

## 演算子 (`ops/elementwise.rs`)

`Array`同士および`Array`とスカラーの四則演算をサポート。

- `+`, `-`, `*`, `/`: 二項演算
- `-` (単項): 符号反転
- 参照版（`&Array`）もサポート
- スカラー演算は`f32`をサポート

## 使用例

```rust
// 配列の作成（遅延評価）
let a = zeros::<Dim2, ...>(ctx.clone(), [100, 100]);
let b = ones::<Dim2, ...>(ctx.clone(), [100, 100]);

// 演算（グラフ構築のみ）
let c = &a + &b;
let d = &c * 2.0f32;

// データ取得時に計算実行（将来実装予定）
// let data: Vec<f32> = d.to_vec()?;
```

## 今後の実装予定

- `to_vec()`: データのホスト読み出し
- `ops/reduce.rs`: sum, mean, max, min
- `ops/transform.rs`: reshape, transpose, squeeze
- `ops/linalg.rs`: matmul
- autogradとの統合（`Variable<Array<T, D>>`パターン）
