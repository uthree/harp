# harpの設計

harpは、高度かつ高速な配列演算をサポートするライブラリです。
配列の操作を計算グラフで表現し、評価を遅延させることで、最適化とC言語やGPUカーネルへのコンパイルを可能にします。
また、計算グラフを扱う性質上、自動微分を実装することが可能で、深層学習や数値最適化などのタスクにも親和性があります。

## 主要コンポーネント

### 1. `Tensor` (ユーザー向けAPI)

配列を表す中心的な構造体。ユーザーはこの `Tensor` に対して演算を行います。

-   **責務**:
    *   ユーザーフレンドリーな配列操作API（`+`, `*`, `reshape`, `sum`など）の提供。
    *   内部に計算グラフ (`TensorOp`) を保持し、演算の履歴を記録する（遅延評価）。
    *   計算結果をキャッシュし、再計算を防ぐ。
-   **具体的な構造 (Rust):**
    ```rust
    use std::cell::RefCell;
    use std::rc::Rc;

    // 計算の履歴を表現するEnum
    pub enum TensorOp {
        // Backend上のバッファを参照する
        Load(Variable), 
        // 算術演算など
        Add(Tensor, Tensor),
        Mul(Tensor, Tensor),
        // ...その他の演算
    }

    // Tensorの実体
    struct Tensor_ {
        op: TensorOp,
        shape: Vec<usize>,
        dtype: DType,
        // このTensorを生成したBackendへの参照
        backend: Rc<dyn Backend>,
        // 計算済みの場合は、その値(Variable)をキャッシュする
        realized: RefCell<Option<Variable>>,
    }

    // ユーザーが触るのはこのRcラッパー
    #[derive(Clone)]
    pub struct Tensor(Rc<Tensor_>);
    ```
-   **主なメソッド**:
    - `realize(&self) -> Variable`:
        1. `realized`フィールドにキャッシュされた`Variable`があればそれをクローンして返す。
        2. なければ、`backend`にコンパイルと実行を依頼する。
        3. 結果の`Variable`を`realized`にキャッシュする。
-   **ポイント**: `Tensor` がどの `Backend` で作られたかを保持することで、計算実行時に適切な `Backend` を呼び出せます。

### 2. `UOp` (中間表現)

`Tensor` の計算グラフから変換される、より低レベルな中間表現(IR)。

-   **責務**:
    *   ハードウェアに依存しない、最小限の演算（Micro-Operations）で計算を表現する。
    *   `Backend` による最適化とコード生成の対象となる。
-   **具体的な構造 (Rust):**
    ```rust
    // 現在の`uop.rs`の実装がこれに相当。
    pub enum Op {
        // メモリ操作
        Load,  // srcs: [buffer, index]
        Store, // srcs: [buffer, index, value]
        // 算術演算
        Add,
        Mul,
        // ...など
        // 制御フロー
        If, // srcs: [condition, true_branch, false_branch]
    }

    pub struct UOp_ {
        op: Op,
        dtype: DType,
        src: Vec<UOp>,
    }
    ```
-   **ポイント**: `Load`/`Store` がどのバッファを指すのか、`Variable` との対応付けを明確にする必要があります。

### 3. `Backend` (実行エンジン)

UOpグラフを解釈し、実際に計算を実行するコンポーネント。

-   **責務**:
    *   `Tensor` の計算グラフを `UOp` グラフに変換する (`compile` メソッド)。
    *   `UOp` グラフを最適化する (`Optimizer` を利用)。
    *   `UOp` グラフをターゲット言語のコードに変換する (`Renderer` を利用)。
    *   メモリ管理 (`Variable` の確保・解放)。
    *   コンパイル済みカーネルの実行。
-   **具体的な構造 (Rust):**
    ```rust
    pub trait Backend {
        // Tensorの計算グラフを受け取り、実行可能なカーネルを返す
        fn compile(&self, tensor: &Tensor) -> Box<dyn Kernel>;
        
        // メモリを確保する
        fn alloc(&self, size: usize) -> Variable;
    }

    // CPUで実行するBackend
    pub struct CpuBackend {
        optimizer: Optimizer,
        renderer: C_Renderer, // C言語を描画
    }

    // GPUで実行するBackend
    pub struct GpuBackend {
        optimizer: Optimizer,
        renderer: MetalRenderer, // Metalシェーダーを描画
    }
    ```

#### 3a. `Optimizer`

`UOp` グラフを最適化する。

-   **責務**: `PatternMatcher` を使い、代数法則の適用や定数畳み込みなどの変換を、グラフが不動点に達するまで行う。
-   **具体的な構造 (Rust):**
    ```rust
    pub struct Optimizer {
        // ルールセットを保持
        matcher: PatternMatcher,
    }

    impl Optimizer {
        pub fn optimize(&self, uop: &UOp) -> UOp {
            self.matcher.apply_all_with_limit(uop, 100) // 繰り返し上限付きで適用
        }
    }
    ```
-   **提案**: 「ブラックボックス最適化」は非常に高度なトピックなので、まずは手動で定義したルールセットを適用する形から始めるのが現実的です。

#### 3b. `Renderer`

`UOp` グラフからターゲット言語のソースコードを生成する。

-   **具体的な構造 (Rust):**
    ```rust
    pub trait Renderer {
        fn render(&self, uop: &UOp) -> String;
    }

    pub struct C_Renderer;
    impl Renderer for C_Renderer {
        // UOpの木をC言語の式に変換する
        fn render(&self, uop: &UOp) -> String { ... }
    }
    ```

#### 3c. `Kernel`

コンパイル済みの実行可能な計算カーネル。

-   **責務**:
    *   実行に必要な引数（入力/出力バッファ）を受け取り、計算を実行する。
-   **具体的な構造 (Rust):**
    ```rust
    pub trait Kernel {
        fn exec(&self, args: &[&Variable]);
    }
    ```

#### 3d. `Variable`

デバイス（CPU/GPU）上のメモリバッファへの参照。

-   **責務**:
    *   バッファのポインタやIDを保持する。
    *   `Drop` トレイトを実装し、スコープを抜けたら自動的に `Backend` にメモリ解放を通知する。
-   **具体的な構造 (Rust):**
    ```rust
    pub struct Variable {
        id: usize, // Backendが管理するID
        size: usize,
        // このVariableを管理するBackendへの参照
        backend: Rc<dyn Backend>, 
    }

    impl Drop for Variable {
        fn drop(&mut self) {
            // self.backend.free(self.id);
        }
    }
    ```