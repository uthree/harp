use crate::backend::Backend;
use crate::uop::UOp;
use std::rc::Rc;
use std::sync::Arc;

// Variableの定義をここに移動
pub struct Variable_ {
    pub id: usize,
    pub size: usize,
    pub backend: Arc<dyn Backend>,
}

impl Drop for Variable_ {
    fn drop(&mut self) {
        // self.backend.free(self.id);
    }
}

#[derive(Clone)]
pub struct Variable(pub Rc<Variable_>);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::dtype::DType;
    use crate::uop::Op;

    #[test]
    fn pipeline_test() {
        // 1. Backendの準備
        let backend = CpuBackend::new();

        // 2. UOpグラフの作成 (a + b)
        let a = UOp::new(Op::Var("a".to_string()), DType::F32, vec![]);
        let b = UOp::new(Op::Var("b".to_string()), DType::F32, vec![]);
        let add_op = UOp::new(Op::Add, DType::F32, vec![a, b]);

        // 3. コンパイルと実行
        // このテストは、現在の実装では引数を必要としないため、空のVecを渡す
        let dummy_args: Vec<&Variable> = vec![];
        backend.compile_and_exec(&add_op, &dummy_args);

        // パニックせずにここまで到達すれば、パイプラインの基本は通っている
        println!("Pipeline test completed successfully!");
    }
}
