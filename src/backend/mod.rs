use std::collections::HashMap;

use crate::ast::AstNode;
pub mod c_like;
pub mod metal;

// レンダラー。
// ASTを受け取って文字列としてレンダリングする
pub trait Renderer {
    type CodeRepr;
    type Option;
    fn render(&self, ast: AstNode) -> Self::CodeRepr;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, _option: Self::Option) {} // default implementation is "do nothing".
}
pub trait Compiler {
    type CodeRepr;
    type Buffer: Buffer;
    type Kernel: Kernel<Buffer = Self::Buffer>;
    type Option;
    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, _option: Self::Option) {} // default implementation is "do nothing".
    fn compile(&mut self, code: &Self::CodeRepr) -> Self::Kernel;
}
pub trait Buffer {
    // get buffer size
    fn shape(&self) -> Vec<usize>;
    // TODO: 初期化と（CPU上の）バイト列への相互変換
}

pub trait Kernel {
    type Buffer: Buffer;
    fn signature(&self) -> KernelSignature;
    // QueryBuilderを経由してメソッドチェーンで引数を指定して最後にcall()みたいな仕組みにしたい
    // 出力用バッファーを自動初期化みたいな機能も欲しい
}

// カーネルへの指示をまとめる構造体
pub struct Query<'a, B: Buffer> {
    inputs: HashMap<String, &'a B>, // inputsは読み取り専用なので借用
    outputs: HashMap<String, B>,    // outputsは書き込み対象
    shape_vars: HashMap<String, usize>,
}
// TODO: QueryBuilderを追加

pub struct KernelSignature {}
