use harp::backends::{Backend, ClangBackend};
use harp::dtype::DType;
use harp::linearizer::Linearizer;
use harp::uop::{Op, UOp};
use std::rc::Rc;

#[test]
fn pipeline_test() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = Rc::new(ClangBackend::new().unwrap());

    // Configure the compiler to use -O3 optimization
    backend.compiler_options_mut().optimization_level = 3;

    let backend: Rc<dyn Backend> = backend;

    // UOpグラフ: a[i] + b[i]
    let buf_a = UOp::var("a", DType::F32);
    let buf_b = UOp::var("b", DType::F32);
    let loop_idx = UOp::var("i", DType::U64);
    let load_a = UOp::new(Op::Load, DType::F32, vec![buf_a.clone(), loop_idx.clone()]);
    let load_b = UOp::new(Op::Load, DType::F32, vec![buf_b.clone(), loop_idx.clone()]);
    let add_op = UOp::new(Op::Add, DType::F32, vec![load_a, load_b]);

    // Store the result
    let buf_out = UOp::var("out", DType::F32);
    let store_op = UOp::new(
        Op::Store,
        DType::Unit,
        vec![buf_out.clone(), loop_idx, add_op],
    );

    // Linearize the UOp graph to a kernel
    let mut linearizer = Linearizer::new();
    let kernel = linearizer.linearize(&store_op, &[10]);

    // Create the necessary variables for execution
    let var_a = backend.alloc(10 * 4, backend.clone());
    let var_b = backend.alloc(10 * 4, backend.clone());
    let var_out = backend.alloc(10 * 4, backend.clone());

    let args = vec![&var_a, &var_b, &var_out];
    backend.compile_and_exec(&kernel, &args, &[]);

    println!("Pipeline test completed successfully!");
}
