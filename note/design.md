# harpの設計

harpは、高度かつ高速な配列演算をサポートするライブラリです。
配列の操作を計算グラフで表現し、評価を遅延させることで、最適化とC言語やGPUカーネルへのコンパイルを可能にします。
また、計算グラフを扱う性質上、自動微分を実装することが可能で、深層学習や数値最適化などのタスクにも親和性があります。

## 主要コンポーネント

### 1. `Tensor` (ユーザー向けAPI)

配列を表す中心的な構造体。ユーザーはこの `Tensor` に対して演算を行います。
`Tensor`のグラフ構築はシングルスレッドで行われることを想定しているため、内部の参照カウントには`Rc`を、内部可変性には`RefCell`を使用し、オーバーヘッドを最小限に抑えています。

- **責務**:
  - ユーザーフレンドリーな配列操作API（`+`, `*`, `reshape`, `sum`など）の提供。
  - 内部に計算グラフを保持し、演算の履歴を記録する（遅延評価）。
  - 計算結果をキャッシュし、再計算を防ぐ。
- **具体的な構造 (Rust):**

    ```rust
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    // Tensorが表現する演算の種類
    pub enum Op {
        // このTensorがBackend上のバッファを直接表現することを示す(srcは空)
        Load,
        // 単項演算
        Neg,
        // 二項演算
        Add, Sub, Mul, Div,
        // ...その他の高レベル演算（reshape, sumなど）
    }

    // Tensorの実体
    // `Tensor`は`Rc<Tensor_>`の型エイリアスとすることで、クローンが軽量になるようにする。
    struct Tensor_ {
        op: Op,
        src: Vec<Tensor>,
        shape: Vec<usize>,
        dtype: DType,
        // このTensorを生成したBackendへの参照(スレッドセーフなArc)
        backend: Arc<dyn Backend>,
        // 計算済みの場合は、その値(Variable)をキャッシュする
        realized: RefCell<Option<Variable>>,
    }

    // ユーザーが触るのはこのRcラッパー
    #[derive(Clone)]
    pub struct Tensor(Rc<Tensor_>);
    ```

- **主なメソッド**:
  - `realize(&self) -> Variable`:
        1. `realized`フィールドにキャッシュされた`Variable`があればそれをクローンして返す。
        2. なければ、`op`と`src`の`Tensor`を辿って計算グラフを`UOp`に変換する。
        3. `backend`に`UOp`を渡し、コンパイルと実行を依頼する。
        4. 結果の`Variable`を`realized`にキャッシュして、クローンを返す。

### 2. `UOp` (中間表現)

`Tensor` の計算グラフから変換される、より低レベルな中間表現(IR)。

- **責務**:
  - ハードウェアに依存しない、最小限の演算（Micro-Operations）で計算を表現する。
  - `Backend` による最適化とコード生成の対象となる。
- **具体的な構造 (Rust):**

    ```rust
    // UOpの実体
    // `UOp`は`Rc<UOp_>`の型エイリアスとすることで、`Tensor`と同様にグラフ構造を表現する。
    pub struct UOp_ {
        op: Op,
        dtype: DType,
        src: Vec<UOp>,
    }
    ```

### 3. `Variable` (メモリバッファ)

デバイス（CPU/GPU）上のメモリバッファへの参照。

- **責務**:
  - バッファのポインタやIDを保持する。
  - `Drop` トレイトを実装し、スコープを抜けたら自動的に `Backend` にメモリ解放を通知する。
- **具体的な構造 (Rust):**

    ```rust
    use std::rc::Rc;
    use std::sync::Arc;

    // Variableの実体
    pub struct Variable_ {
        id: usize, // Backendが管理するID
        size: usize,
        // このVariableを管理するBackendへの参照(スレッドセーフなArc)
        backend: Arc<dyn Backend>,
    }

    impl Drop for Variable_ {
        fn drop(&mut self) {
            self.backend.free(self.id); // BackendにIDを渡して解放を依頼
        }
    }

    // ユーザー（やTensor）が主に扱うのはこのRcラッパー
    #[derive(Clone)]
    pub struct Variable(Rc<Variable_>);
    ```

### 4. `Backend` (実行エンジン)

`Backend`は、`UOp`グラフのコンパイルから実行までを統括するオーケストレーターです。内部に`Optimizer`, `Renderer`, `Compiler`といったコンポーネントを保持し、それらを連携させて`Kernel`を生成・実行します。

#### `backend::get` ファクトリ

ユーザーが簡単に計算デバイスを選択できるようにするため、`torch.device()`に似たファクトリ機能 `backend::get` を提供します。

- **責務**:
  - `"cpu"`や`"cuda"`のような文字列を受け取り、対応する`Backend`の共有インスタンス(`Arc<dyn Backend>`)を返す。
  - `Backend`インスタンスを内部のグローバルキャッシュで管理し、デバイスごとにシングルトンであることを保証する。
- **使い方**:

    ```rust
    // ユーザーは簡単な文字列でデバイスを指定できる
    let cpu_backend = backend::get("cpu");
    let tensor_a = Tensor::from_data(&[1.0, 2.0], cpu_backend);
    ```

#### Backendの内部コンポーネント

##### a. `Optimizer`

`UOp` グラフを最適化する。

- **責務**: `PatternMatcher` を使い、代数法則の適用や定数畳み込みなどの変換を、グラフが不動点に達するまで行う。

##### b. `Renderer`

最適化された`UOp`グラフから、ターゲット言語（C言語やCUDA C++など）のソースコードを生成する。

- **責務**: `UOp`の木構造を辿り、等価な処理を行うソースコード文字列を構築する。
- **具体的な構造 (Rust):**

    ```rust
    pub trait Renderer {
        fn render(&self, uop: &UOp) -> String;
    }
    ```

##### c. `Compiler`

`Renderer`が生成したソースコードをコンパイルし、実行可能な`Kernel`を生成する。

- **責務**:
  - 自身の利用可能性（例: `gcc`コマンドの存在）を報告する。
  - ソースコード文字列とコンパイルオプションを受け取り、外部コンパイラを呼び出して動的ライブラリ等を生成し、`Kernel`オブジェクトを作成する。

- **コンパイルオプション**:

    ```rust
    // コンパイル設定を保持する構造体
    pub struct CompileOptions {
        pub optimization_level: u8, // 0, 1, 2, 3
        pub debug_info: bool,       // -g フラグに相当
        // ... その他のフラグ (e.g., fast-math)
    }

    impl Default for CompileOptions {
        fn default() -> Self {
            Self { optimization_level: 2, debug_info: false }
        }
    }
    ```

- **具体的な構造 (Rust):**

    ```rust
    pub trait Compiler {
        /// このコンパイラがシステムで利用可能かを確認する
        fn is_available(&self) -> bool;

        /// ソースコードをコンパイルし、Kernelを返す
        fn compile(&self, source_code: &str, options: &CompileOptions) -> Result<Arc<dyn Kernel>, Error>;
    }

    // gccを使ったCPU用のCompiler実装例
    pub struct GccCompiler;
    impl Compiler for GccCompiler {
        fn is_available(&self) -> bool {
            // `which gcc` や `gcc --version` を実行し、成功するかどうかで判断
            std::process::Command::new("gcc").arg("--version").output().is_ok()
        }

        fn compile(&self, source_code: &str, options: &CompileOptions) -> Result<Arc<dyn Kernel>, Error> {
            // 1. `options`を解釈してgccのコマンドライン引数を組み立てる
            let opt_level = format!("-O{}", options.optimization_level);
            let mut args = vec!["-shared", "-fPIC", &opt_level];
            if options.debug_info {
                args.push("-g");
            }

            // 2. `source_code`を一時ファイルに書き出し、`gcc`を呼び出す
            // ... (前述の実装と同様)

            // 3. `libloading`で.soをロードし、CpuKernelを返す
            // ...
            unimplemented!()
        }
    }
    ```

- **`Backend`との連携**:
  - 各`Backend`（例: `CpuBackend`）は、自身の初期化時に、対応する`Compiler`（例: `GccCompiler`）が利用可能かを確認し、デフォルトの`CompileOptions`と共に保持します。
  - ユーザーは、`Backend`のメソッドを通じてこれらのコンパイルオプションを後から変更することも可能です。

### 5. `Kernel` (実行可能カーネル)

コンパイル済みの、特定の計算を実行するためのオブジェクト。

- **責務**:
  - 実行に必要な引数（入力/出力バッファのポインタやID）を受け取り、コンパイル済みのコード（ネイティブ関数やGPUカーネル）を実行する。
- **具体的な構造 (Rust):**

    ```rust
    pub trait Kernel {
        fn exec(&self, args: &[&Variable]);
    }

    // CPUカーネルの実装例
    use libloading::{Library, Symbol};
    pub struct CpuKernel {
        _lib: Library, // ライブラリの所有権を保持し、Drop時にアンロードする
        func: Symbol<unsafe extern "C" fn(*mut f32, *const f32, ...)>,
    }
    impl Kernel for CpuKernel {
        fn exec(&self, args: &[&Variable]) {
            // argsのVariableからメモリアドレスを取得し、
            // unsafeブロック内でロードした関数シンボル `self.func` を呼び出す。
            // ...
        }
    }
    ```
