# Harp: High-level n-dimensional array processing library

[![crates.io](https://img.shields.io/crates/v/harp.svg)](https://crates.io/crates/harp)
[![Build Status](https://github.com/uthre/harp/actions/workflows/rust.yml/badge.svg)](https://github.com/uthre/harp/actions/workflows/rust.yml)

Harpは、高レベルなN次元配列処理のためのRust製ライブラリです。テンソル計算グラフの構築、最適化、自動微分などの機能を提供し、機械学習や数値計算アプリケーションの開発を強力にサポートします。

## ✨ Features

*   **コンパイル可能なテンソル計算グラフ:** 効率的な計算実行のためのグラフを構築します。
*   **軽量なオペレーターセット:** [Tinygrad](https://github.com/tinygrad/tinygrad/tree/master)や[luminal](https://github.com/jafioti/luminal)にインスパイアされた、最小限かつ強力なオペレーターを提供します。
*   **容易なデバイスサポート拡張:** トレイト実装により、様々なハードウェアデバイスへの対応を簡単に追加できます。
*   **グラフ最適化:** 計算グラフの最適化により、実行効率を向上させます。
*   **オペレーター融合:** メモリアクセスを削減し、パフォーマンスを最大化するためのオペレーター融合をサポートします。
*   **自動微分:** 複雑なモデルの勾配計算を自動化します。

## 🚀 Getting Started

Harpをプロジェクトに追加するには、`Cargo.toml`に以下を追加してください。

```toml
[dependencies]
harp = "0.1.0" # 最新バージョンに置き換えてください
```

### 使用例

簡単なテンソル計算グラフを構築し、実行する例です。

```rust
use harp::prelude::*;
use std::sync::{Arc, Mutex};

fn main() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape = ShapeTracker::full(vec![2.into(), 3.into()]);

    // 入力テンソルの定義
    let a = Graph::new_input(graph.clone(), shape.clone());
    let b = Graph::new_input(graph.clone(), shape.clone());

    // テンソル演算
    let c = &a + &b; // a + b

    // グラフの出力を設定
    graph.lock().unwrap().add_output(&c);

    // グラフの実行 (Interpreterを使用)
    let mut interpreter = Interpreter::new();
    let result = interpreter.evaluate(
        c.node_index,
        &graph.lock().unwrap().graph,
        &std::collections::HashMap::new(),
        &std::collections::HashMap::new(),
    ).unwrap();

    println!("Result: {:?}", result);
}
```

## 🗺️ Roadmap

*   より多くのオペレーターの実装
*   様々なデバイスバックエンドのサポート (CUDA, OpenCLなど)
*   高度なグラフ最適化パスの追加
*   より使いやすいAPIの提供

## 🤝 Contributing

Harpへの貢献を歓迎します！バグ報告、機能リクエスト、プルリクエストなど、どのような形でも構いません。詳細は`CONTRIBUTING.md` (準備中) を参照してください。

## 📄 License

このプロジェクトはApacheライセンスの下で公開されています。詳細は`LICENSE`ファイルを参照してください。
