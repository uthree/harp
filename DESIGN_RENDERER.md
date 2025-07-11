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

## コンパイラー
生成されたコードをコンパイルする機能と、実際にそのコンパイラが利用可能であるかをチェックする機能を持ちます。
```rust

trait Compiler {
    fn is_available(&self) -> bool; //利用可能かチェックする。内部的には、コンパイラが使えるかどうかのシェルコマンドとかを叩く。
    fn compile(&self, code: String) -> impl Kernel // コンパイルされたカーネルを表すオブジェクトを返す。
}

// コンパイラを追加する場合はこう
struct CudaCompiler {}
impl Compiler for CudaCompiler {
    ...
}
```

## バックエンド
コードの解釈からコンパイル、最適化などを統括して行うためのインターフェース。
```rust

trait Backend {
    fn name(&self) -> String // 表示名を取得する。
    fn is_available(&self) -> bool // バックエンドが利用可能であるかチェックする。
}
trait Kernel {
    // GPU向けにコンパイルされたバイナリを呼び出すFFIのラッパー。
} 
trait DeviceTensor { // デバイス上にあるデータのポインタ。
    // 命名は思いつかないけど、ndarrayやVec<T>に変換する機能とか欲しいね。
    // デバイス間を移動する機能も必要になるはず。
}

// たとえば、CUDAには複数のグラフィックカードを扱う機能があるため、以下のようにする。
struct Cuda<const device_id: usize = 0> {} // デバイス番号を型情報として持つ。
impl Backend for Cuda {
    ...
}
```