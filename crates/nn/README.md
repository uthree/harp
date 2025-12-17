# harp-nn
自動微分や多次元配列の計算の抽象化、ニューラルネットワークを構築するための基本機能を提供します。

## 機能
 - [ndarray](https://github.com/rust-ndarray/ndarray)のような多次元配列を操作するためのAPI（内部での計算は遅延評価）
    - `harp`プロジェクトの基本的な設計思想に則って、少数のプリミティブ演算子の組み合わせですべての演算を表現する。
    - `TensorD`(次元を型システムで管理しないテンソル)と、そのラッパーとして、`Tensor1`, `Tensor2`などの次元数を固定する型を用意する。
 - [micrograd](https://github.com/karpathy/micrograd)のような自動微分
    - `WithGradient<T>`ラッパー（後で命名は変えるかもしれない）を被せることによって、型レベルで勾配を伝播し得るどうかを判定可能にする。
 - [PyTorch](https://github.com/pytorch/pytorch)のようなニューラルネットワークを構築するための機能
    - `Module`: 学習可能パラメータを持つニューラルネットを構成するモジュール
    - `Parameter`: 学習可能なパラメータであることを表すラッパー
    - `opt`: SGDやAdamなどの最適化アルゴリズムの実装
    - `data`: データセット、前処理パイプライン、データ読み込みなど
    - `backend`: 演算を行うバックエンドを提供する、`torch.device`のようなAPI
