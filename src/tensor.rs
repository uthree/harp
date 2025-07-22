use crate::backends::Backend;
use std::rc::Rc;
use std::sync::Arc;

pub struct Variable_ {
    pub id: usize,
    pub size: usize,
    pub backend: Arc<dyn Backend>,
}

impl Drop for Variable_ {
    fn drop(&mut self) {
        self.backend.free(self.id);
    }
}

#[derive(Clone)]
pub struct Variable(pub Rc<Variable_>);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{Backend, CpuBackend};
    use crate::dtype::DType;
    use crate::uop::{Op, UOp};

    #[test]
    fn pipeline_test() {
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new());

        // UOpグラフ: a[i] + b[i]
        let buf_a = UOp::var("a", DType::F32);
        let buf_b = UOp::var("b", DType::F32);
        let loop_idx = UOp::var("i", DType::U64);
        let load_a = UOp::new(Op::Load, DType::F32, vec![buf_a, loop_idx.clone()]);
        let load_b = UOp::new(Op::Load, DType::F32, vec![buf_b, loop_idx]);
        let add_op = UOp::new(Op::Add, DType::F32, vec![load_a, load_b]);

        // 実行に必要なVariableを作成
        let var_a = backend.alloc(10 * 4, backend.clone());
        let var_b = backend.alloc(10 * 4, backend.clone());
        let var_out = backend.alloc(10 * 4, backend.clone());
        
        let args = vec![&var_a, &var_b, &var_out];
        backend.compile_and_exec(&add_op, &args);

        println!("Pipeline test completed successfully!");
    }
}
