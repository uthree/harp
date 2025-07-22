# harpの設計

harpは、高度かつ高速な配列演算をサポートするライブラリです。
配列の操作を計算グラフで表現し、評価を遅延させることで、最適化とC言語やGPUカーネルへのコンパイルを可能にします。
また、計算グラフを扱う性質上、自動微分を実装することが可能で、深層学習や数値最適化などのタスクにも親和性があります。

## コンパイルパイプライン概要

`harp`は、`Tensor`で行われた操作を、以下のステップを経て実行可能な`Kernel`に変換します。

1. **グラフ構築 (`Tensor` -> `UOp`グラフ):** `Tensor`の演算履歴から、有向非巡回グラフ(DAG)構造の`UOp`を構築します。
2. **最適化 (`UOp`グラフ -> `UOp`グラフ):** `Optimizer`が代数法則の適用などを行い、`UOp`グラフを最適化します。
3. **Lowering (`UOp`グラフ -> `UOp`ツリー):** 最適化されたグラフを、ループなどの構造を考慮した**抽象構文木(AST)ツリー**に変換します。このステップで、共有ノードは変数への代入などに置き換えられます。
4. **レンダリング (`UOp`ツリー -> `String`):** `Renderer`が`UOp`ツリーを辿り、C言語などのソースコードを生成します。
5. **コンパイル (`String` -> `Kernel`):** `Compiler`がソースコードをコンパイルし、実行可能な`Kernel`を生成します。

## 主要コンポーネント

### 1. `Tensor` (ユーザー向けAPI)

配列を表す中心的な構造体。ユーザーはこの `Tensor` に対して演算を行います。

### 2. `UOp` (抽象構文木)

`Tensor`の計算グラフから変換される中間表現(IR)。当初はDAGとして構築され、Loweringの過程でツリー構造に変換されます。式と文の両方を表現できる、C言語に近いASTとして機能します。

- **具体的な構造 (Rust):**

    ```rust
    use std::rc::Rc;

    pub enum UOp {
        // --- 式 (値を返すノード) ---
        Binary { op: BinaryOp, lhs: Rc<UOp>, rhs: Rc<UOp>, dtype: DType },
        Unary { op: UnaryOp, src: Rc<UOp>, dtype: DType },
        Load { buf_idx: usize, idx: Rc<UOp>, dtype: DType }, // buf[idx]
        Const { val: f32, dtype: DType },
        Var { name: String, dtype: DType }, // ループ変数などを参照

        // --- 文 (値を返さないノード) ---
        // for (int var = 0; var < limit; i++) { body }
        Loop {
            var: String,
            limit: Rc<UOp>,
            body: Rc<UOp>, // 通常はBlockノード
        },
        // { stmts[0]; stmts[1]; ... }
        Block {
            stmts: Vec<Rc<UOp>>,
        },
        // buf[idx] = value;
        Store {
            buf_idx: usize,
            idx: Rc<UOp>,
            value: Rc<UOp>,
        },
        // if (condition) { true_branch }
        If {
            condition: Rc<UOp>,
            true_branch: Rc<UOp>, // 通常はBlockノード
        },
    }
    ```

### 3. `Variable` (メモリバッファ)

デバイス（CPU/GPU）上のメモリバッファへの参照。

### 4. `Backend` (実行エンジン)

`UOp`グラフのコンパイルから実行までを統括するオーケストレーターです。

#### `backend::get` ファクトリ

ユーザーが `"cpu"` や `"cuda"` のような文字列で、対応する`Backend`の共有インスタンス(`Arc<dyn Backend>`)を簡単に取得できるようにする機能です。

#### `Backend`トレイトと高レベルAPI

`Backend`は、ユーザー向けにコンパイラの種類を意識させない、高レベルな設定APIを提供します。

### 5. `Compiler` (コンパイラ)

`Renderer`が生成したソースコードをコンパイルし、実行可能な`Kernel`を生成します。

- **責務**:
  - 自身の利用可能性を報告する。
  - **自身専用のコンパイルオプション**と共にソースコードを受け取り、`Kernel`を生成する。
  - `Kernel`に、実行に必要なメタデータ（引数情報、ワークサイズ等）を焼き込む。

### 6. `Kernel` (実行可能カーネル)

コンパイル済みの、特定の計算を実行するための自己完結型オブジェクト。

- **責務**:
  - 実行に必要な引数のメタデータ（データ型、サイズ等）を内部に保持する。
  - `exec`が呼ばれた際に、受け取った引数がメタデータと一致するかを検証する。
  - 検証後、コンパイル済みのコードを安全に実行する。
- **メタデータ構造 (Rust):**

    ```rust
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
    ```
