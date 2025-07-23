use harp::backends::{Backend, GccBackend};
use harp::dtype::DType;
use harp::lower;
use harp::uop::{Op, UOp};
use std::sync::Arc;

#[test]
fn pipeline_test() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend: Arc<dyn Backend> = Arc::new(GccBackend::new());

    // UOpグラフ: a[i] + b[i]
    let buf_a = UOp::var("a", DType::F32);
    let buf_b = UOp::var("b", DType::F32);
    let loop_idx = UOp::var("i", DType::U64);
    let load_a = UOp::new(Op::Load, DType::F32, vec![buf_a.clone(), loop_idx.clone()]);
    let load_b = UOp::new(Op::Load, DType::F32, vec![buf_b.clone(), loop_idx.clone()]);
    let add_op = UOp::new(Op::Add, DType::F32, vec![load_a, load_b]);

    // Store the result
    let buf_out = UOp::var("out", DType::F32);
    let store_op = UOp::new(Op::Store, DType::Unit, vec![buf_out, loop_idx, add_op]);

    // Loop over the operation
    let loop_op = UOp::new(Op::Loop, DType::Unit, vec![10u64.into(), store_op]);

    // Lower the UOp graph to an AST
    let ast = lower::lower(&loop_op);

    // 実行に必要なVariableを作成
    let var_a = backend.alloc(10 * 4, backend.clone());
    let var_b = backend.alloc(10 * 4, backend.clone());
    let var_out = backend.alloc(10 * 4, backend.clone());

    let args = vec![&var_a, &var_b, &var_out];
    backend.compile_and_exec(&ast, &args);

    println!("Pipeline test completed successfully!");
}
