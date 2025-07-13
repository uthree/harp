# コンパイル処理の概略
harpでは、多次元配列(Tensor)の計算を遅延評価で行うことにより、計算グラフから自動的に最も早いコードを生成し、最速で計算を行います。
計算グラフからコードの生成は以下のような手順で行われます。  

## 1. テンソルレベルの計算グラフの処理
### 1.1. グラフの構築
`Tensor`構造体に対して演算を行うことで、自動的に計算グラフが構築されます。
演算子をオーバーロードする形で使用できるAPIにより、numpy(pythonのライブラリ)に近い使い方が可能です。  
また、この段階では実際の計算は行われません。（遅延評価）

### 1.2. グラフの最適化
テンソルレベルでのグラフが構築され、実際に値を計算する必要性が出た場合、計算を行う前にグラフを最適化します。これは、不要な演算ノードを削除したりすることで、パフォーマンスを最適化する目的があります。

### 1.3. ShapeTrackerの実行
テンソルの形状とループの添え字がどのように変化するかを処理します。
ShapeTrackerは、各次元の添え字 -> メモリオフセット の変換を行う計算式です。  
これを使うことで、同じ要素を並べたり、numpyやpytorchにおけるブロードキャストのような処理をする際に、実際の値を複製せずに計算できるため、メモリの使用量を削減することができます。

## 2. lowerグラフの処理
### 2.1. lowering
テンソルレベルでのグラフは、そのままではC言語やCuda, Metal向けのコードに変換できないため、より低レベルな演算子のみから構成される計算グラフに変換されます。便宜上、この低レベルなグラフをlowerグラフと呼ぶことにします。

### 2.2. lowerグラフの最適化
ここでももう一段階最適化処理が走ります。  
不要な演算子を削除したり、同じ添字で行われるループ演算を一回のループに融合したりなどが行われます。  

### 2.3. グラフのLinearize (手続き的IRへの変換)
計算グラフ構造をターゲット言語に変換しやすくするため、より手続き的な中間表現（IR: Intermediate Representation）に変換します。このプロセスを`linearize`と呼びます。

このIRは、仮想的なレジスタを使って計算の途中結果を管理し、単純な命令のシーケンスとしてプログラムを表現します。

```rust
// --- 中間表現 (Intermediate Representation) ---

/// 手続き的IRのトップレベル構造。C言語の関数に相当する。
pub struct Procedure {
    /// 関数名
    pub name: String,
    /// 引数のリスト (バッファ名とデータ型)
    pub args: Vec<(String, DType)>,
    /// 命令の本体
    pub body: Vec<Instruction>,
}

/// 仮想的なレジスタ。計算の途中結果を保持する。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VReg(pub usize);

/// IRの各命令
pub enum Instruction {
    /// `dest = op(src1, src2, ...)` のような計算を表現する。
    Compute {
        dest: VReg,
        op: PrimitiveOp, // 具体的なプリミティブ演算子enum
        src: Vec<VReg>,
    },
    /// 定数をレジスタにロードする。
    Const {
        dest: VReg,
        value: Box<dyn DType>,
    },
    /// ループを表現する。
    Loop {
        /// ループカウンタとして使うレジスタ
        counter: VReg,
        /// ループの最大回数（この値未満までループ）
        max: VReg,
        /// ループ本体の命令
        body: Vec<Instruction>,
    },
    /// メモリからのロード `dest = buffer[index]`
    Load {
        dest: VReg,
        buffer_name: String,
        index: VReg,
    },
    /// メモリへのストア `buffer[index] = value`
    Store {
        buffer_name: String,
        index: VReg,
        value: VReg,
    },
}

/// プリミティブな演算子のみを集めたenum
/// (src/op.rsの各種Op構造体とは別)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveOp {
    Add,
    Sub,
    Mul,
    Div,
    // ... その他のプリミティブ演算
}
```

### 2.4. 手続きの最適化
手続き的IRに対して、ループアンローリングや定数伝播などの最適化を適用することができます。ただし、この工程は後回しにし、まずは最小限の機能でコンパイルフローを完成させることを目指します。

## 3. コードのレンダリング
手続き的IRをターゲット言語のソースコードに変換します。この責務を担うのが`Renderer`です。

```rust
/// 手続き的IRからソースコードを生成するトレイト
trait Renderer {
    fn render(proc: &Procedure) -> String;
}

struct CRenderer;
impl Renderer for CRenderer {
    // ... Cコードを生成する ...
}
```
`Renderer`は、`Procedure`構造体全体を受け取り、それを文字列に変換します。ターゲットごとに`Renderer`を実装することで、C、CUDA、Metalなど、様々なバックエンドに対応できます。

`PrimitiveOp`のサポートは、各`Renderer`が責任を持ちます。将来的には、`FusedOp`をサポートするための拡張も考えられますが、��ずは`PrimitiveOp`の完全なサポートを目指します。
