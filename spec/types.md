# データ型

## DType

プリミティブなデータ型を表す列挙型。

```rust
pub enum DType {
    // 特殊型
    Void,       // 戻り値なし
    Unknown,    // 未推論

    // 論理型
    Bool,

    // 符号付き整数
    I8, I16, I32, I64,

    // 符号なし整数
    U8, U16, U32, U64,

    // 浮動小数点
    F16,    // half precision
    BF16,   // bfloat16
    F32,    // single precision
    F64,    // double precision

    // インデックス型
    Int,    // プラットフォーム依存（通常i64）

    // 複合型
    Ptr(Box<DType>),         // ポインタ
    Vec(Box<DType>, usize),  // 固定長ベクトル（SIMD用）
    Tuple(Vec<DType>),       // タプル
}
```

### サイズ

| 型 | バイト数 |
|----|----------|
| Bool, I8, U8 | 1 |
| I16, U16, F16, BF16 | 2 |
| I32, U32, F32 | 4 |
| I64, U64, F64 | 8 |
| Int | プラットフォーム依存 |

### メソッド

- `size_in_bytes()`: バイト単位のサイズ
- `bit_width()`: ビット幅
- `is_float()`, `is_integer()`, `is_numeric()`: 型判定
- `zero()`, `one()`: ゼロ/1リテラル生成

## Literal

リテラル値を表す列挙型。

```rust
pub enum Literal {
    Bool(bool),
    I8(i8), I16(i16), I32(i32), I64(i64),
    U8(u8), U16(u16), U32(u32), U64(u64),
    F16(f16), BF16(bf16), F32(f32), F64(f64),
}
```

### メソッド

- `dtype()`: 対応するDTypeを取得
- `as_i64()`, `as_f64()`, `as_bool()`: 値の取得
- `is_zero()`, `is_one()`: 値判定

## Expr

シンボリックな式を表す列挙型。シェイプ、ストライド、インデックス計算に使用。

```rust
pub enum Expr {
    Const(i64),                     // 定数
    Bool(bool),                     // 真偽値
    Idx(usize),                     // ループインデックス（ridx0, ridx1, ...）
    Sym(String),                    // シンボル変数（実行時値）
    Add(Box<Expr>, Box<Expr>),      // 加算
    Sub(Box<Expr>, Box<Expr>),      // 減算
    Mul(Box<Expr>, Box<Expr>),      // 乗算
    Div(Box<Expr>, Box<Expr>),      // 除算
    Rem(Box<Expr>, Box<Expr>),      // 剰余
    Lt(Box<Expr>, Box<Expr>),       // 比較（<）
    And(Box<Expr>, Box<Expr>),      // 論理積
    Not(Box<Expr>),                 // 論理否定
    LoadIndex {                     // ギャザー演算
        src_index: usize,
        offset_expr: Box<Expr>,
    },
}
```

### 主要メソッド

- `simplify()`: 式の簡略化
- `evaluate(vars)`: 変数を与えて評価
- `substitute_idx(index, expr)`: インデックス変数の置換
- `to_ast()`: AstNodeへ変換

## TensorDType trait

Rustの型からDTypeへのマッピング。

```rust
pub trait TensorDType: Clone + Debug + Send + Sync + 'static {
    const DTYPE: DType;
    fn to_literal(value: Self) -> Literal;
    fn zero() -> Self;
    fn one() -> Self;
}
```

対応する型: `bool`, `i8`-`i64`, `u8`-`u64`, `f16`, `bf16`, `f32`, `f64`
