# AST (抽象構文木)

## AstNode

カーネルレベルの計算を表現する列挙型。

### 算術演算

```rust
// 二項演算
Add(Box<AstNode>, Box<AstNode>)      // 加算
Mul(Box<AstNode>, Box<AstNode>)      // 乗算
Max(Box<AstNode>, Box<AstNode>)      // 最大
Min(Box<AstNode>, Box<AstNode>)      // 最小
Rem(Box<AstNode>, Box<AstNode>)      // 剰余
Idiv(Box<AstNode>, Box<AstNode>)     // 整数除算

// 単項演算
Recip(Box<AstNode>)     // 逆数 (1/x)
Sqrt(Box<AstNode>)      // 平方根
Log2(Box<AstNode>)      // 対数（底2）
Exp2(Box<AstNode>)      // 指数（底2）
Sin(Box<AstNode>)       // 正弦
Floor(Box<AstNode>)     // 床関数
Neg(Box<AstNode>)       // 符号反転

// 型変換
Cast(Box<AstNode>, DType)

// 融合積和
Fma { a, b, c }         // a * b + c
```

### ビット演算

```rust
BitwiseAnd(Box<AstNode>, Box<AstNode>)
BitwiseOr(Box<AstNode>, Box<AstNode>)
BitwiseXor(Box<AstNode>, Box<AstNode>)
BitwiseNot(Box<AstNode>)
LeftShift(Box<AstNode>, Box<AstNode>)
RightShift(Box<AstNode>, Box<AstNode>)
```

### 比較・論理演算

```rust
Lt(Box<AstNode>, Box<AstNode>)      // <
Le(Box<AstNode>, Box<AstNode>)      // <=
Gt(Box<AstNode>, Box<AstNode>)      // >
Ge(Box<AstNode>, Box<AstNode>)      // >=
Eq(Box<AstNode>, Box<AstNode>)      // ==
Ne(Box<AstNode>, Box<AstNode>)      // !=
And(Box<AstNode>, Box<AstNode>)     // &&
Or(Box<AstNode>, Box<AstNode>)      // ||
Not(Box<AstNode>)                   // !
```

### メモリ操作

```rust
// ロード
Load {
    ptr: Box<AstNode>,      // ポインタ
    offset: Box<AstNode>,   // オフセット
    count: usize,           // 要素数（ベクトルロード用）
    dtype: DType,           // データ型
}

// ストア
Store {
    ptr: Box<AstNode>,
    offset: Box<AstNode>,
    value: Box<AstNode>,
}

// メモリ確保/解放
Allocate { size, dtype }
Deallocate { ptr }

// アトミック演算
AtomicAdd { ptr, offset, value, dtype }
AtomicMax { ptr, offset, value, dtype }

// WMMA行列積（Tensor Core）
WmmaMatmul {
    a_ptr, a_offset, a_stride,  // 行列A [M,K]
    b_ptr, b_offset, b_stride,  // 行列B [K,N]
    c_ptr, c_offset, c_stride,  // 行列C [M,N]
    m, k, n,                     // サイズ
    dtype_ab,                    // 入力型（F16）
    dtype_c,                     // 出力型（F32）
}
```

### 制御フロー

```rust
// ループ
Range {
    var: String,            // ループ変数名
    start: Box<AstNode>,
    step: Box<AstNode>,
    stop: Box<AstNode>,
    body: Box<AstNode>,
    parallel: Option<ParallelInfo>,
    unroll: Option<usize>,
}

// 条件分岐
If {
    cond: Box<AstNode>,
    then_body: Box<AstNode>,
    else_body: Option<Box<AstNode>>,
}

// 三項演算子
Select {
    cond: Box<AstNode>,
    then_val: Box<AstNode>,
    else_val: Box<AstNode>,
}

// バリア同期
Barrier
```

### 関数・カーネル

```rust
// 関数定義
Function {
    name: String,
    params: Vec<VarDecl>,
    body: Box<AstNode>,
    return_type: DType,
}

// カーネル定義
Kernel {
    name: String,
    params: Vec<VarDecl>,
    body: Box<AstNode>,
    workgroup_size: Option<(usize, usize, usize)>,
}

// 関数呼び出し
Call {
    name: String,
    args: Vec<AstNode>,
}

// プログラム（カーネル集合）
Program {
    kernels: Vec<AstNode>,
    waves: Vec<Wave>,   // 実行順序
}
```

### その他

```rust
Const(Literal)          // 定数
Var(String)             // 変数参照
Wildcard(String)        // パターンマッチ用
Block { statements, result }  // ブロック
Assign { var, value }   // 代入
Rand                    // 乱数生成
```

## ParallelInfo

並列化メタデータ。

```rust
pub struct ParallelInfo {
    pub parallel_type: ParallelType,
    pub factor: Option<usize>,
}

pub enum ParallelType {
    Local,      // OpenMP local
    Group,      // GPU workgroup
    GPU,        // GPU global
    Rayon,      // Rayon (Rust)
}
```

## VarDecl

変数宣言。

```rust
pub struct VarDecl {
    pub name: String,
    pub dtype: DType,
    pub is_buffer: bool,    // バッファポインタか
}
```

## Wave

カーネル実行順序。

```rust
pub struct Wave {
    pub kernel_indices: Vec<usize>,  // 同時実行可能なカーネル
}
```

## AstNodeのメソッド

| メソッド | 説明 |
|----------|------|
| `children()` | 子ノード取得 |
| `map(f)` | 子ノードに関数適用 |
| `infer_type()` | 型推論 |
| `substitute(var, expr)` | 変数置換 |
| `check_scope(scope)` | スコープ検証 |
