# 計算グラフ

## GraphNode

計算グラフのノード。参照カウント(`Rc`)で共有される。

```rust
pub struct GraphNode(Rc<GraphInner>);

pub struct GraphInner {
    pub src: Vec<GraphNode>,        // 入力ノード（複数可）
    pub view: View,                 // メモリレイアウト
    pub op: GraphOp,                // 演算種別
    pub dtype: DType,               // 出力データ型
    pub name: Option<String>,       // デバッグ用名前
    pub buffer_id: Option<usize>,   // 外部バッファ参照
}
```

## GraphOp

グラフ演算の種別。

```rust
pub enum GraphOp {
    // レイアウト変更のみ（計算なし）
    View(View),

    // 要素演算 + オプショナルリダクション
    MapReduce {
        map: AstNode,                           // 要素演算
        reduce: Option<(ReduceOp, usize)>,      // リダクション（演算, 軸）
    },

    // スライディングウィンドウ抽出
    Unfold {
        dim: usize,         // 展開軸
        size: Expr,         // ウィンドウサイズ
        step: Expr,         // ステップ
    },

    // scatter-add（Unfoldの逆演算）
    Scatter {
        dim: usize,
        size: Expr,
        step: Expr,
    },

    // 累積演算
    Scan {
        reduce_op: ReduceOp,    // 累積演算種別
        dim: usize,             // 軸
        exclusive: bool,        // 排他的か
        reverse: bool,          // 逆順か
    },
}
```

## ReduceOp

リダクション演算の種別。

```rust
pub enum ReduceOp {
    Sum,    // 総和
    Max,    // 最大値
    Min,    // 最小値
    Prod,   // 総乗
}
```

## View

メモリレイアウトを表現。

```rust
pub enum View {
    // ストライド形式（標準的なレイアウト）
    Linear {
        shape: Vec<Expr>,       // 各軸のサイズ
        strides: Vec<Expr>,     // 各軸のストライド
        offset: Expr,           // オフセット
        bounds: ViewBounds,     // 境界条件
    },

    // 任意インデックス式（ギャザー、アンフォールド等）
    IndexExpr {
        shape: Vec<Expr>,       // 出力シェイプ
        index_expr: Expr,       // インデックス計算式
        bounds: ViewBounds,     // 境界条件
    },
}
```

### ViewBounds

境界条件とパディング。

```rust
pub struct ViewBounds {
    pub lower: Vec<Expr>,           // 下限
    pub upper: Vec<Expr>,           // 上限
    pub pad_value: Option<Literal>, // 境界外の値
}
```

### Viewの演算

| メソッド | 説明 |
|----------|------|
| `reshape(shape)` | シェイプ変更 |
| `permute(order)` | 軸の並べ替え |
| `expand(shape)` | ブロードキャスト |
| `tile(axis, tiles)` | タイリング |
| `slice(ranges)` | スライス |
| `pad(padding, value)` | パディング |
| `unfold(dim, size, step)` | スライディングウィンドウ |
| `compose(other)` | ビュー合成 |

## グラフ構築

```rust
// 入力ノード作成
let a = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);

// 演算
let b = &a + &a;        // 要素演算
let c = b.sum(1);       // リダクション
let d = c.reshape(vec![Expr::Const(32), Expr::Const(1)]);
```

## 融合ルール

Lowererによる演算融合:

1. **View融合**: 連続するView演算は合成
2. **MapReduce融合**: 要素演算 + リダクションを1カーネルに
3. **1 GraphNode = 1 Kernel**: 融合後
