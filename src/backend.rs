use crate::kernel::Kernel;
use crate::tensor::Tensor;
use crate::variable::Variable;

pub trait Backend {
    // Tensorの計算グラフを受け取り、実行可能なカーネルを返す
    fn compile(&self, tensor: &Tensor) -> Box<dyn Kernel>;

    // メモリを確保する
    fn alloc(&self, size: usize) -> Box<dyn Variable>;
}
