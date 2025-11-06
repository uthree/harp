use crate::backend::Renderer;

// C言語に近い構文の言語のためのレンダラー
// Metal, CUDA, C(with OpenMP), OpenCLなどのバックエンドは大体C言語に近い文法を採用しているので、共通化したい。
pub trait CLikeRenderer: Renderer {
    //TODO
}
