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

# Dataset, DataLoader, Transform
PyTorchに寄せたい。  
データの読み込みや加工のパイプラインを表す。

# Preprocess
[tts_implリポジトリ](https://github.com/uthree/tts_impl)で[前処理の抽象化を行なっている](https://github.com/uthree/tts_impl/blob/main/src/tts_impl/utils/preprocess/base.py)ので、これを参考に前処理パイプラインを作れるようにしたい。

# TrainerやLightningModuleに相当する何か
PyTorch Lightningに寄せたい。
初期化から勾配の計算、パラメータの更新なんかをひとまとめにするtrait。

# 各種メディアへの対応
テキスト、画像、音声、動画などをはじめとするデータをTensorとして読み書きできるような機能をつけたい。ただし、ffmpegなんかに依存したりしそうなので、featureフラグでオプション機能としてつけるか、サブクレートとして分離する貸した方が良さそう。

