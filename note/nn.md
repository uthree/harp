# Tensor
Bufferのラッパー、遅延評価でGraphへ変換して計算したり、自動微分をしたりする。

# Module
ニューラルネットワークの一部として最適化することができることを表すトレイト。  
最適化対象のパラメータを取得・読み込みをする機能を持つ。  
将来的にはderiveマクロで自動実装できるようにしたい。

```rust
trait Module {
    fn get_parameters(&self) -> HashMap<String, nn::Parameter>; // パラメータを辞書型で取得する
    fn load_parameters(&mut, self, parameters: HashMap<String, nn::Parameter>); // パラメータを辞書からロードする
}
```
あるいは、パラメータ辞書を格納する専用の型を作っても良いかも。

# Parameter
Tensorのラッパー、Tensorとほぼ同じ機能を持つが、Optimizerによる最適化対象であることを明示する。

# Optimizer
ニューラルネットワークのパラメータを最適化するオプティマイザー。  
SGD, Adamなどはこれをベースに実装する。
```rust
trait Optimizer {
    // TODO: 最適化対象のパラメータを追加する機能
    fn init_grad(&mut self) {...} ; // 最適化対象のパラメータの勾配をリセットする
    fn step(&mut self); // 1ステップ分最適化するコールバック関数
}
```