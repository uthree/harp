use harp::autotuner::BackendOptions;
use harp::backends::clang::compiler::ClangCompileOptions;
use harp::backends::{Backend, ClangBackend};
use harp::dtype::DType;
use harp::linearizer::Linearizer;
use harp::uop::{Op, UOp};
use std::rc::Rc;

#[test]
fn pipeline_test() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = ClangBackend::new();
    let backend = Rc::new(backend);

    let backend: Rc<dyn Backend> = backend;

    // UOp graph: a[i] + b[i]
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
    let use_counts = store_op.get_use_counts();
    let mut linearizer = Linearizer::new(&use_counts);
    let kernel = linearizer.linearize(&store_op, &[10]);

    // Create the necessary variables for execution
    let var_a = backend.alloc(10 * 4, backend.clone());
    let var_b = backend.alloc(10 * 4, backend.clone());
    let var_out = backend.alloc(10 * 4, backend.clone());

    // Configure the compiler to use -O3 optimization
    let mut clang_options = ClangCompileOptions::default();
    clang_options.optimization_level = 3;
    let options = BackendOptions::Clang(clang_options);

    let args = vec![&var_a, &var_b, &var_out];
    let exec_time = backend.compile_and_exec(&kernel, &args, &[], &options);

    println!("Pipeline test completed successfully in {:?}!", exec_time);
}
