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

### 2. `UOp` (中間表現)

`Tensor` の計算グラフから変換される、より低レベルな中間表現(IR)。

- **責務**: ハードウェアに依存しない、最小限の演算で計算を表現する。

### 3. `Variable` (メモリバッファ)

デバイス（CPU/GPU）上のメモリバッファへの参照。

- **責務**: バッファのIDやサイズを保持し、`Drop`時に`Backend`にメモリ解放を通知する。

### 4. `Backend` (実行エンジン)

`Backend`は、`UOp`グラフのコンパイルから実行までを統括するオーケストレーターです。内部に`Optimizer`, `Renderer`, `Compiler`といったコンポーネントを保持します。

#### `backend::get` ファクトリ

ユーザーが `"cpu"` や `"cuda"` のような文字列で、対応する`Backend`の共有インスタンス(`Arc<dyn Backend>`)を簡単に取得できるようにする機能です。

#### `Backend`トレイトと高レベルAPI

`Backend`は、ユーザー向けにコンパイラの種類を意識させない、高レベルな設定APIを提供します。

- **具体的な構造 (Rust):**

    ```rust
    pub trait Backend {
        // ...
        fn set_optimization_level(&self, level: u8);
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
    ```

### 6. `Kernel` (実行可能カーネル)

コンパイル済みの、特定の計算を実行するための自己完結型オブジェクト。

- **責務**:
  - 実行に必要な引数のメタデータ（データ型、サイズ等）を内部に保持する。
  - `exec`が呼ばれた際に、受け取った引数がメタデータと一致するかを検証する。
  - 検証後、コンパイル済みのコード（ネイティブ関数やGPUカーネル）を安全に実行する。

- **メタデータ構造 (Rust):**

    ```rust
    // カーネルが期待する単一の引数の情報
    pub struct ArgInfo {
        pub dtype: DType,
        pub size: usize,
    }

    // カーネル全体の実行情報
    pub struct KernelMetadata {
        pub args_info: Vec<ArgInfo>,      // 引数情報のリスト (入力と出力を含む)
        pub global_work_size: usize,    // ループ回数や、GPUのグローバルワークサイズ
        pub local_work_size: usize,     // GPUのローカルワークサイズ (CPUでは未使用)
    }
    ```

- **具体的な構造 (Rust):**

    ```rust
    pub trait Kernel {
        fn exec(&self, args: &[&Variable]);
        fn metadata(&self) -> &KernelMetadata;
    }

    // CPUカーネルの実装例
    use libloading::{Library, Symbol};
    pub struct CpuKernel {
        _lib: Library, // ライブラリの所有権を保持し、Drop時にアンロードする
        // C関数は引数としてループ回数(work_size)とポインタ配列を受け取る
        func: Symbol<unsafe extern "C" fn(usize, *const *const u8)>,
        metadata: KernelMetadata,
    }

    impl Kernel for CpuKernel {
        fn metadata(&self) -> &KernelMetadata {
            &self.metadata
        }

        fn exec(&self, args: &[&Variable]) {
            // 1. --- 安全性のための検証 (Sanity Check) ---
            assert_eq!(args.len(), self.metadata.args_info.len(), "Mismatched number of arguments");
            for (i, var) in args.iter().enumerate() {
                assert_eq!(var.0.size, self.metadata.args_info[i].size, "Mismatched size for argument {}", i);
                // dtypeのチェックも可能
            }

            // 2. --- 実行 ---
            // Variableからメモリアドレスのリストを準備する
            let raw_ptrs: Vec<*const u8> = args.iter().map(|v| v.backend().get_buffer_ptr(v.id())).collect();

            // unsafeブロック内でロードした関数シンボル `self.func` を呼び出す
            unsafe {
                (self.func)(self.metadata.global_work_size, raw_ptrs.as_ptr());
            }
        }
    }
    ```

- **`Backend`との連携**:
  - `Backend`は`Variable`の`id`から実際のメモリアドレス（ポインタ）を取得するためのメソッド（例: `get_buffer_ptr`）を提供する必要があります。
