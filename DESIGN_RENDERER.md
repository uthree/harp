こんな感じの設計考えたんだけど、できそう？

# コード生成について
harpでは、計算グラフを受け取ってGPU向けのカーネルやC言語での高速な実装などにコンパイルすることが目的です。  

## レンダラー
レンダラーは、Nodeを受け取ってソースコードを生成します。

```rust

// あるオペレーターがレンダリングできることを表すトレイト。
trait Render<Op>
where Op: Operator // レンダラーによって利用可能な演算子が異なるため、ジェネリックトレイトを使って型システムでそれ表現する。
{
    fn render(&mut self, op: Op, src: Node) -> String;
}

// たとえば、Metal(Apple版CUDAのようなもの)への変換はこのように実装する。
struct MetalRenderer {}
// オペレータごとに実装を追加していくイメージ。
impl Render<AddOp> for MetalRenderer {
    ...
}

// FusedOpに関しては、レンダリング処理が実装されていない場合、フォールバックしてからレンダリングする。
```