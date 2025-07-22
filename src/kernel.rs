use crate::dtype::DType;
use crate::tensor::Variable;

// カーネルが期待する単一の引数の情報
pub struct ArgInfo {
    pub dtype: DType,
    pub size: usize,
}

// カーネル全体の実行情報
pub struct KernelMetadata {
    pub args_info: Vec<ArgInfo>,
    pub global_work_size: usize,
    pub local_work_size: usize,
}

pub trait Kernel {
    fn exec(&self, args: &[&Variable]);
    fn metadata(&self) -> &KernelMetadata;
}
