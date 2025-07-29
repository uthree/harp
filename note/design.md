# 全体的な設計

## 中間表現

テンソルレベル・ASTレベルでそれぞれ構造体を持つ。
両者ともパターンマッチングによる最適化が可能（不要ノードを削除したりなど）

### Tensor グラフ

テンソルレベルでの演算を表す計算グラフ(DAG)。
自動微分(autograd)で自身の微分グラフを作ることができる。

### UOp グラフ

テンソルより低レベルなDAG。

### `struct Kernel`: カーネル

単一のカーネルを表す構造体。
構文木のほかに入出力の型などの情報を持つ。

### `struct UOp`

式や文を表現する

## バックエンド

後述のレンダラーやコンパイラーを統括管理し、コードの最適化・コンパイル・実行などをする。

```rust
trait Backend {
    ... // TODO
}
```

### レンダラー

ASTを実際のコード(e.g. C言語, CUDA, Metal...)にレンダリングする責務を持つ。

```rust
trait Renderer {
    type CodeRepresentation = String; // コードの表現形式。通常は文字列だが、バイト列を扱うケースも想定して一応ジェネリックにしておく。
    fn new() -> Self; // 内部状態を初期化。
    fn render(&mut self, ast::Kernel) -> CodeRepresentation;

    fn render_stmt(&mut self, ast::Statement) -> CodeRepresentation;
    fn render_expr(&mut, self, ast::Expression) -> CodeRepresentation;
}
```

### コンパイラー

レンダリングされたコードを実行可能なバイナリにへ関する責務を持つ。

```rust
trait Compiler {
    type CompilerOption; // コンパイラオプションを表現する型。
    fn new() -> Self; // 内部状態を初期化。
    fn default_option() -> CompilerOption // コンパイラオプションの既定値を取得
    fn compile(&mut self, ) // コンパイル処理を実行。
}
```

### オプティマイザー

TODO, 最適化処理を担う。最適化手法を探索する。ベイズ推定とかグリッドサーチとか色々使う。

## パターンマッチとグラフ構造の最適化

アルゴリズムで自動的にされる数式やコードは、ときに冗長になりえます。そんな冗長な構造をより簡素で効率的なものに置き換えることが目的です。

### `TPat`, `TPatternMatcher`

テンソルに対して置き換え処理を行います。

```rust
// 置き換え規則
struct TPat {
    pattern: Tensor, // 検出するパターン
    rewriter: FnOnce(Vec<Tensor>) -> Tensor // 検出した際にそれをどう置き換えるかのクロージャー
}

// 置き換え規則の集まり。
struct TPatternMatcherData {
    name: String, // name for debug
    patterns: Vec<Rc<TRule>>, // Rules
}

// 上記の置き換え規則の集まりを扱うためのRcポインタ。
struct TPatternMatcher(Rc<TPatternMatcherData>);
```

### `UPat`, `UPatternMatcher`

UOpに対するパターンマッチング。
テンソルの場合とほぼ同じなので割愛。

## ShapeTracker

添え字からメモリオフセットを計算する関数のExprです。  
TensorグラフをExprツリーに変換する(lower)ときに使用します。
