# harp-nn
自動微分や多次元配列の計算の抽象化、ニューラルネットワークを構築するための基本機能を提供します。

## 機能
 - [ndarray](https://github.com/rust-ndarray/ndarray)のような多次元配列を操作するためのAPI（内部での計算は遅延評価）
    - `harp`プロジェクトの基本的な設計思想に則って、少数のプリミティブ演算子の組み合わせですべての演算を表現する。
 - [micrograd](https://github.com/karpathy/micrograd)のような自動微分
 - [PyTorch](https://github.com/pytorch/pytorch)のようなニューラルネットワークを構築するための機能
    - `Module`: 学習可能パラメータを持つニューラルネットを構成するモジュール
    - `opt`: SGDやAdamなどの最適化アルゴリズムの実装
    - `data`: データセット、前処理パイプライン、データ読み込みなど
