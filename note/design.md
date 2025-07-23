# harpの設計

harpは、高度かつ高速な配列演算をサポートするライブラリです。
配列の操作を計算グラフで表現し、評価を遅延させることで、最適化とC言語やGPUカーネルへのコンパイルを可能にします。
また、計算グラフを扱う性質上、自動微分を実装することが可能で、深層学習や数値最適化などのタスクにも親和性があります。

## コンパイルパイプライン概要

`harp`は、`Tensor`で行われた操作を、以下のステップを経て実行可能な`Kernel`に変換します。

1. **グラフ構築 (`Tensor` -> `UOp`グラフ):** `Tensor`の演算履歴から、有向非巡回グラフ(DAG)構造の`UOp`を構築します。
2. **最適化 (`UOp`グラフ -> `UOp`グラフ):** `Optimizer`が代数法則の適用などを行い、`UOp`グラフを最適化します。
3. **Lowering (`UOp`グラフ -> `UOp`ツリー):** 最適化されたグラフを、ループなどの構造を考慮した**抽象構文木(AST)ツリー**に変換します。このステップで、共有ノードは変数への代入などに置き換えられます。
4. **レンダリング (`UOp`ツリー -> `String`):** `Renderer`が`UOp`ツリーを再帰的に辿り、C言語などのソースコードを生成します。
5. **コンパイル (`String` -> `Kernel`):** `Compiler`がソースコードをコンパイルし、実行可能な`Kernel`を生成します。

---

## 主要コンポーネント

### 1. `Tensor` (ユーザー向けAPI)

配列を表す中心的な構造体。ユーザーはこの `Tensor` に対して演算を行います。

- **具体的な構造 (Rust):**

    ```rust
    use crate::shapetracker::ShapeTracker;
    use crate::tensor::TensorOp;
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    struct Tensor_ {
        op: TensorOp,
        src: Vec<Tensor>,
        tracker: ShapeTracker,
        dtype: DType,
        backend: Arc<dyn Backend>,
        realized: RefCell<Option<Variable>>,
    }

    #[derive(Clone)]
    pub struct Tensor(Rc<Tensor_>);
    ```

### 2. `UOp` (抽象構文木)

`Tensor`の計算グラフから変換される中間表現(IR)。ノードの構造(`UOp_`)と操作の種類(`Op`)が分離されており、拡張性が高い。当初はDAGとして構築され、Loweringの過程でツリー構造に変換されます。

- **具体的な構造 (Rust):**

    ```rust
    use std::rc::Rc;

    // UOpが表現する操作の種類
    pub enum Op {
        Add, Mul, Recip, Rem, // Binary
        Exp2, Log2, Sin, Sqrt, // Unary
        Load, Store,
        Const(Number),
        Var(String),
        Loop, Block, If,
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

### 3. `Variable` (メモリバッファ)

デバイス（CPU/GPU）上のメモリバッファへの参照。

- **具体的な構造 (Rust):**

    ```rust
    use std::sync::Arc;
    use std::rc::Rc;

    pub struct Variable_ {
        id: usize,
        size: usize,
        backend: Arc<dyn Backend>,
    }

    impl Drop for Variable_ {
        fn drop(&mut self) {
            self.backend.free(self.id);
        }
    }

    #[derive(Clone)]
    pub struct Variable(Rc<Variable_>);
    ```

### 4. `Backend` (実行エンジン)

`UOp`グラフのコンパイルから実行までを統括するオーケストレーターです。

#### `backend::get` ファクトリ

ユーザーが `"cpu"` や `"cuda"` のような文字列で、対応する`Backend`の共有インスタンス(`Arc<dyn Backend>`)を簡単に取得できるようにする機能です。

#### `Backend`トレイトと高レベルAPI

`Backend`は、バックエンドの種類（CPU, GPUなど）に依らない共通の操作を定義する、汎用的なインターフェースです。

- **具体的な構造 (Rust):**

    ```rust
    pub trait Backend {
        fn compile_and_exec(&self, uop: &UOp, args: &[&Variable]);
        fn alloc(&self, size: usize, backend: Arc<dyn Backend>) -> Variable;
        // ... etc
    }
    ```

#### `ClangBackend`実装例

`ClangBackend`のような具体的なバックエンドは、`Backend`トレイトを実装すると同時に、自身に固有の設定メソッドを提供します。これにより、汎用的な操作はトレイト経由で、バックエンド固有の設定は具象型経由で、というように責務を分離します。

- **具体的な構造 (Rust):**

    ```rust
    use std::sync::Mutex;

    pub struct ClangBackend {
        compiler: ClangCompiler,
        compile_options: Mutex<ClangCompileOptions>,
        // ...
    }

    impl ClangBackend {
        // 固有の設定メソッド
        pub fn configure_compiler<F>(&self, f: F) where F: FnOnce(&mut ClangCompileOptions) {
            // ...
        }
    }

    impl Backend for ClangBackend {
        // ...
    }
    ```

### 5. `Compiler` (コンパイラ)

`Renderer`が生成したソースコードをコンパイルし、実行可能な`Kernel`を生成します。

- **責務**:
  - 自身の利用可能性を報告する。
  - **自身専用のコンパイルオプション**と共にソースコードを受け取り、`Kernel`を生成する。
  - `Kernel`に、実行に必要なメタデータ（引数情報、ワークサイズ等）を焼き込む。

- **具体的な構造 (Rust):**

    ```rust
    pub trait Compiler {
        type Options: Default + Clone;
        fn is_available(&self) -> bool;
        fn compile(&self, source_code: &str, options: &Self::Options) -> Result<Arc<dyn Kernel>, Error>;
    }

    #[derive(Clone, Default)]
    pub struct ClangCompileOptions { /* ... */ }
    pub struct ClangCompiler;
    impl Compiler for ClangCompiler {
        type Options = ClangCompileOptions;
        // ...
    }
    ```

### 6. `Kernel` (実行可能カーネル)

コンパイル済みの、特定の計算を実行するための自己完結型オブジェクト。

- **責務**:
  - 実行に必要な引数のメタデータ（データ型、サイズ等）を内部に保持する。
  - `exec`が呼ばれた際に、受け取った引数がメタデータと一致するかを検証する。
  - 検証後、コンパイル済みのコードを安全に実行する。

- **メタデータ構造 (Rust):**

    ```rust
    pub struct ArgInfo { pub dtype: DType, pub size: usize }
    pub struct KernelMetadata {
        pub args_info: Vec<ArgInfo>,
        pub global_work_size: usize,
        pub local_work_size: usize,
    }
    ```

- **具体的な構造 (Rust):**

    ```rust
    pub trait Kernel {
        fn exec(&self, args: &[&Variable]);
        fn metadata(&self) -> &KernelMetadata;
    }

    pub struct ClangKernel {
        // ...
        metadata: KernelMetadata,
    }
    impl Kernel for ClangKernel {
        // ...
    }
    ```
