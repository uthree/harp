# 全体的な設計

## 中間表現

テンソルレベル・ASTレベルでそれぞれ構造体を持つ。
両者ともパターンマッチングによる最適化が可能（不要ノードを削除したりなど）

### Tensor 計算グラフ

ユーザーが直接操作する表現。
テンソルレベルでの演算を表す計算グラフ(DAG)。
自動微分(autograd)で自身の微分グラフを作ることができる。（TODO）

### AST

構文木に相当する。
C言語の構造に簡単に変換できることを想定しており、演算やループ処理、メモリへの読み書きの機能を持つ。

## バックエンド

後述のレンダラーやコンパイラーを統括管理し、コードの最適化・コンパイル・実行などをする。

```rust
pub trait Backend {
    type Var: Buffer;
    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn call(
        &mut self,
        graph: Graph,
        buffers: Vec<Self::Var>,
        shape_variables: Vec<usize>,
    ) -> Self::Var;
}
```

### レンダラー

ASTを実際のコード(e.g. C言語, CUDA, Metal...)にレンダリングする責務を持つ。

```rust
trait Renderer<CodeRepr = String> {
    fn new() -> Self;
    fn render(&mut self, ast: AstNode) -> CodeRepr;
}
```

### コンパイラー

レンダリングされたコードを実行可能な形式（カーネル）に変換する責務を持つ。

```rust
trait Compiler<CodeRepr = String, CompilerOption = ()> {
    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, option: CompilerOption);
    fn compile(&mut self, code: &CodeRepr, details: KernelDetails) -> Box<dyn Kernel>;
}
```

### カーネル

コンパイル済みの実行可能な関数を表す。

```rust
pub trait Kernel {
    fn details(&self) -> &KernelDetails;
    fn call(&self, buffers: Vec<Box<dyn Buffer>>, shape_variables: &[usize]) -> Vec<Box<dyn Buffer>>;
}
```

### バッファ

デバイス上のデータを保持するコンテナ。

```rust
pub trait Buffer: AsAny {
    fn as_mut_bytes(&mut self) -> &mut [u8];
    fn dtype(&self) -> DType;
    fn shape(&self) -> Vec<usize>;
    fn size(&self) -> usize;
}
```

### オプティマイザー

TODO, 最適化処理を担う。最適化手法を探索する。ベイズ推定とかグリッドサーチとか色々使う。
とりあえず今は触らない方針で。

## ShapeTracker

添え字からメモリオフセットを計算する関数のExprです。
TensorグラフをAstに変換する(lower)ときに使用します。

## Lowerer

TensorグラフからASTに変換する。
状態を持つので毎回初期化して使う必要がある。

