use harp::backends::{Backend, CpuBackend};
use harp::dtype::DType;
use harp::uop::{Op, UOp};
use std::sync::Arc;

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
