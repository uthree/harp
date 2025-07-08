# Harp 中間表現 (IR) 仕様書

このドキュメントは、Harpの計算グラフを実行可能な形式に変換するための中間表現（IR）の設計仕様を概説します。

## 1. 全体方針

このIRの主な目的は、以下の特徴を持つ表現を作成することです。
- ハードウェア、特にGPUの実行モデルに近いこと。
- 様々な最適化に適していること。
- 高レベルな計算グラフからのコンパイル先として適切であること。

設計は静的メモリアロケーションモデルを採用し、実行前に必要なすべてのメモリを単一のアリーナとして確保します。

## 2. モジュール構成

プロジェクトは以下のように構成されます。
- `src/graph/`: ユーザーが直接触れるグラフ表現（ノード、テンソル、演算子など）を含みます。
- `src/ir/`: 新しい中間表現と、グラフをこのIRに変換するためのロジックを���みます（今後実装予定）。
- `src/shape/`: 他のモジュールからも利用されるため、トップレベルに維持します。

## 3. IRの構成要素

IRは主に`Function`、`Kernel`、`Instruction`の3つの階層構造で構成されます。

### 3.1. `Function`

`Function`は、実行可能な一連の計算全体を格納するトップレベルのコンテナです。メモリを管理し、`Kernel`の実行を統括します。

```rust
pub struct Function {
    pub name: String,
    pub kernels: Vec<Kernel>,
    // この関数で必要となるすべてのメモリバッファの定義リスト
    pub buffers: Vec<Buffer>,
    // 実行に必要となる総メモリ領域のサイズ（バイト単位）
    pub required_memory: usize,
    // 関数の引数として渡されるバッファのIDリスト
    pub args: Vec<BufferId>,
    // 関数の戻り値となるバッファのID
    pub ret: BufferId,
}
```

### 3.2. `Kernel`

`Kernel`は、並列実行可能な計算の単位を表し、GPUカーネルのような概念とよく対応します。`Function`は複数の`Kernel`を持つことができ、データ依存関係に応じて逐次的または並列的に実行されます。

```rust
pub struct Kernel {
    pub name: String,
    pub instructions: Vec<Instruction>,
    // このカーネル内で使用される各仮想レジスタのデータ型
    pub vregs: Vec<DType>,
    // 並列デバイス上でのカーネルの起動設定（例：GPUのグリッド/ブロックサイズ）
    pub launch_dims: [usize; 3],
    // このカーネルが読み込むバッファのIDリスト
    pub reads: Vec<BufferId>,
    // このカーネルが書き込むバッファのIDリスト
    pub writes: Vec<BufferId>,
}
```

### 3.3. `Instruction`

`Instruction`は、`Kernel`内での最も基本的な操作単位です。

```rust
// 計算途中の中間値を保持する仮想レジスタ
pub type VReg = usize;

// メモリバッファを一意に識別するためのID
pub type BufferId = usize;

// 実行可能な命令セット
pub enum Instruction {
    // 定数値を仮想レジスタにロードする
    Const { out: VReg, val: Scalar },

    // 算��論理演算（Add, Mul, Sinなど）を実行する
    Alu {
        op: AluOp,
        out: VReg,
        lhs: VReg,
        rhs: Option<VReg>, // 単項演算の場合はNone
    },

    // メモリバッファから仮想レジスタにデータをロードする
    Load { out: VReg, from: BufferId, shape: ShapeTracker },

    // 仮想レジスタからメモリバッファにデータをストアする
    Store { to: BufferId, from: VReg, shape: ShapeTracker },

    // 乱数を生成する
    Rand { op: RandOp, out: VReg, shape: ShapeTracker },

    // 将来的な制御フローのための拡張
    // Loop { ... },
    // If { ... },
}

// 算術論理演算の種類
pub enum AluOp {
    Add, Mul, Exp2, Log2, Sin, Sqrt, Recip, LessThan, // など
}

// 乱数生成演算の種類
pub enum RandOp {
    Uniform, // 0-1の一様分布
    Normal,  // 標準正規分布
}
```

## 4. メモリ管理

メモリは、`Function`ごとに単一のメモリアリーナを使用する静的方式で管理されます。

- **静的確保**: `Function`の実行前に、`required_memory`で指定されたサイズの���続したメモリ領域（アリーナ）を一度だけ確保します。この領域は、関数が完了した後に一度だけ解放されます。
- **バッファはビューとして**: `Buffer`オブジェクトは自身ではメモリを所有しません。代わりに、`offset`と`size`によって定義される、メインアリーナへの「ビュー」またはスライスとして機能します。
- **メモリの再利用**: グラフからIRへのコンパイラは、テンソルの生存期間（liveness）を分析し、生存期間が重複しないテンソル同士が同じメモリ領域を共有できるように最適化します。これにより、`required_memory`の総量を最小化します。

```rust
// メモリアリーナへのビューを表す
pub struct Buffer {
    pub id: BufferId,
    // メモリアリーナ内の開始オフセット（バイト単位）
    pub offset: usize,
    // このバッファのサイズ（バイト単位）
    pub size: usize,
    pub dtype: DType,
    // バッファが存在するメモリ空間（例：CPUまたはGPU）
    pub memory_space: MemorySpace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    Host,   // CPUメモリ
    Device, // GPUメモリ
}
```
