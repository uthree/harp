# harpの設計

harpは、高度かつ高速な配列演算をサポートするライブラリです。
配列の操作を計算グラフで表現し、評価を遅延させることで、最適化とC言語やGPUカーネルへのコンパイルを可能にします。
また、計算グラフを扱う性質上、自動微分を実装することが可能で、深層学習や数値最適化などのタスクにも親和性があります。

## コンパイルパイプライン概要

`harp`は、`Tensor`で行われた操作を、以下のステップを経て実行可能な`Kernel`に変換します。

1. **Lowering (`Tensor`グラフ -> `UOp`グラフ):** `Lowerizer`が`Tensor`の計算グラフを辿り、ハードウェアに依存しない中間表現(`UOp`)のグラフに変換します。要素ごとの単純な演算は式ツリーに、`Reduce`のような複雑な操作はアキュムレーターの初期化やループを含む手続き的な`Block`に変換されます。
2. **最適化 (`UOp`グラフ -> `UOp`グラフ):** `Optimizer`が代数法則の適用などを行い、`UOp`グラフを最適化します。
3. **Linearization (`UOp`グラフ -> `Vec<UOp>`):** `Linearizer`が`UOp`グラフを、カーネルの線形な命令リストに変換します。このステップで、出力テンソルの形状に基づいた明示的なループ (`LoopStart`/`LoopEnd`) を生成します。`Reduce`の場合、`Lowerizer`が生成した`Block`をこのループの内側に入れ子にすることで、ネストされたループ構造を構築します。また、複雑な式は一時変数 (`Op::Declare`) に分解されます。
4. **レンダリング (`Vec<UOp>` -> `String`):** `Renderer`が命令リストを順に辿り、`Op::Declare`を変数宣言、`Op::Store`を再代入として解釈しながら、C言語などのソースコードを生成します。
5. **コンパイル (`String` -> `Kernel`):** `Compiler`がソースコードをコンパイルし、実行可能な`Kernel`を生成します。

この一連のフローは、後述する`Autotuner`によってラップされ、最適な設定を見つけるために繰り返し実行されます。

---

## オートチューニングによる最適化

同じ計算グラフでも、適用する最適化ルールやコンパイラのオプションによって最終的な実行パフォーマンスは大きく変わります。`harp`では、最も良い設定の組み合わせを自動的に探索する「オートチューナー」の仕組みを提供します。

`Autotuner`は、指定された探索空間（Search Space）の中から、探索戦略（Search Strategy）に従って設定を一つずつ試し、計算グラフの`realize`（コンパイルと実行）にかかる時間を計測します。これを指定された回数繰り返し、最も時間が短かった設定を最良の結果として報告します。

### 主要コンポーネント (`autotuner.rs`)

- **`Configuration`**: 1回の試行で使われる全設定を保持する構造体。有効にする最適化ルールのセットや、Clangバックエンドに渡すコンパイルオプション（最適化レベル `-O` など）が含まれます。

- **`SearchSpace`**: `Configuration`が取りうる値の範囲やリストを定義します。どの最適化ルールを有効/無効の対象にするか、どのコンパイラオプションを試すかなどを指定します。

- **`SearchStrategy`**: `SearchSpace`の中から、次に試すべき`Configuration`を決定するロジックです。現在は、すべての組み合わせを試す`GridSearch`が実装されています。

- **`Autotuner`**: チューニングプロセス全体を管理するメインコンポーネントです。`run`メソッドが呼び出されると、内部で以下の処理をループします。
    1. `SearchStrategy`から次の`Configuration`を取得する。
    2. `Tensor`のキャッシュをクリアする (`clear_cache`)。
    3. `Tensor::realize_with_config`を呼び出し、時間を計測する。
    4. 結果を`TrialResult`として保存する。

### `realize` との連携

オートチューニングを実現するため、`Tensor`の`realize`メソッドは内部的に`realize_with_config`を呼び出すように拡張されています。

```rust
// tensor.rs
impl<T> Tensor<T> {
    // ユーザーが通常呼び出すメソッド
    pub fn realize(&self) -> Buffer {
        self.realize_with_config(&Configuration::default())
    }

    // Autotunerが内部で呼び出すメソッド
    pub fn realize_with_config(&self, config: &Configuration) -> Buffer {
        // ...
        // configに基づいてOptimizerを初期化
        let optimizer = Optimizer::new(config);
        let optimized_uop_graph = optimizer.optimize(&uop_graph);
        // ...
        // configのコンパイラオプションをBackendに渡す
        self.0.backend.compile_and_exec(..., &Some(config.clang_options.clone()));
        // ...
    }
}
```

この設計により、通常の利用者は設定を意識することなく`realize()`を呼び出すだけで最適化されたカーネルを実行でき、一方でパフォーマンスを追求したい場合は`Autotuner`を使って特定の計算グラフに特化した最良の設定を見つけ出すことができます。

---

## 主要コンポーネント詳細

### 1. `Tensor` (ユーザー向けAPI)

配列を表す中心的な構造体。ユーザーはこの `Tensor` に対して演算を行います。
本ライブラリはシングルスレッドでの使用を前提としており、`Rc`と`RefCell`を用いて内部状態を管理します。

`Tensor`はジェネリック構造体 `Tensor<T>` であり、`f32`, `f64`など様々な数値型を扱うことができます。

- **具体的な構造 (Rust):**

    ```rust
    pub struct Tensor_<T> {
        pub op: TensorOp,
        pub src: Vec<Tensor<T>>,
        pub tracker: ShapeTracker,
        pub dtype: DType,
        pub backend: Rc<dyn Backend>,
        pub realized: RefCell<Option<Buffer>>,
        phantom: std::marker::PhantomData<T>,
    }

    #[derive(Clone)]
    pub struct Tensor<T>(pub Rc<Tensor_<T>>);
    ```

#### `ndarray`との連携

利便性のため、`Tensor<T>`は`ndarray::ArrayD<T>`との相互変換をサポートしています。`From`トレイトが実装されているため、`.into()`メソッドでシームレスに変換できます。

```rust
use ndarray::ArrayD;
use harp::prelude::Tensor;

// ndarray -> Tensor
let nd_array: ArrayD<f32> = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
let tensor: Tensor<f32> = nd_array.into();

// Tensor -> ndarray
let nd_array_again: ArrayD<f32> = tensor.into();
```

### 2. `Context` (実行コンテキスト)

計算を実行するための、スレッドごとの設定や環境を管理するモジュールです。
主に、バックエンドのインスタンスをスレッドローカルなレジストリで管理する役割を担います。

#### `backend()` ファクトリ関数

`harp::backend("clang")`のように文字列でバックエンドを指定すると、対応するインスタンス(`Rc<dyn Backend>`)を取得できます。
同じスレッド内でこの関数を複数回呼び出しても、常に同じインスタンスが返されるため、複数のテンソルが意図せず異なるバックエンドを持ってしまう問題を回避できます。

- **具体的な構造 (Rust):**

    ```rust
    // context.rs
    thread_local! {
        static BACKEND_REGISTRY: RefCell<HashMap<String, Rc<dyn Backend>>> = RefCell::new(HashMap::new());
    }

    pub fn backend(name: &str) -> Rc<dyn Backend> {
        // ... レジストリから取得または新規作成 ...
    }
    ```

### 3. `UOp` (抽象構文木)

`Tensor`の計算グラフから変換される中間表現(IR)。ノードの構造(`UOp_`)と操作の種類(`Op`)が分離されており、拡張性が高い。当初はDAGとして構築され、Loweringの過程でツリー構造に変換されます。

- **具体的な構造 (Rust):**

    ```rust
    // UOpが表現する操作の種類
    pub enum Op {
        // Binary Ops
        Add, Mul, Div, Recip, Rem,
        // Unary Ops
        Exp2, Log2, Sin, Sqrt,
        // Memory Ops
        Load, Store,
        // Control Flow & Variables
        Declare(String, DType),
        Const(Number),
        Var(String),
        LoopStart, LoopEnd, Block, If,
        // Other
        Cast(DType),
    }

    // UOpノードの実体
    pub struct UOp_ {
        pub op: Op,
        pub dtype: DType, // このUOpが返す値の型 (文の場合はUnit型など)
        pub src: Vec<UOp>,  // この操作への入力となる子ノード
    }

    // ユーザーが主に扱うRcラッパー
    #[derive(Clone)]
    pub struct UOp(pub Rc<UOp_>);
    ```

### 4. `Buffer` (メモリバッファ)

デバイス（CPU/GPU）上のメモリバッファへの参照。以前の`Variable`から改名されました。

- **具体的な構造 (Rust):**

    ```rust
    pub struct Buffer_ {
        id: usize,
        size: usize,
        backend: Rc<dyn Backend>,
    }

    impl Drop for Buffer_ {
        fn drop(&mut self) {
            self.backend.free(self.id);
        }
    }

    #[derive(Clone)]
    pub struct Buffer(Rc<Buffer_>);
    ```

### 5. `Backend` (実行エンジン)

`UOp`グラフのコンパイルから実行までを統括するオーケストレーターです。`Context`を通じて取得されます。

- **`Backend`トレイトと高レベルAPI:**

    ```rust
    pub trait Backend {
        fn compile_and_exec(
            &self,
            uops: &[UOp],
            bufs: &[&Buffer],
            shape_args: &[usize],
            options: &Option<ClangCompileOptions>,
        );
        fn alloc(&self, size: usize, backend: Rc<dyn Backend>) -> Buffer;
        // ... etc
    }
    ```

- **`ClangBackend`実装例:**

    ```rust
    pub struct ClangBackend {
        compiler: ClangCompiler,
        // ...
    }
    ```

### 6. `Compiler` (コンパイラ)

`Renderer`が生成したソースコードをコンパイルし、実行可能な`Kernel`を生成します。

- **具体的な構造 (Rust):**

    ```rust
    pub trait Compiler {
        type Options: Default + Clone;
        fn is_available(&self) -> bool;
        fn compile(&self, source_code: &str, options: &Self::Options) -> Result<Rc<dyn Kernel>, Error>;
    }
    ```

### 7. `Kernel` (実行可能カーネル)

コンパイル済みの、特定の計算を実行するための自己完結型オブジェクト。

- **具体的な構造 (Rust):**

    ```rust
    pub trait Kernel {
        fn exec(&self, args: &[&Buffer], shape_args: &[usize]);
        fn metadata(&self) -> &KernelMetadata;
    }
    ```