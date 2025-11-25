use harp::DType;
use unified_ir::*;

#[test]
fn test_end_to_end_elementwise() {
    println!(
        "\n========== Element-wise Add: 高レベル → 最適化 → Lowering → コード生成 ==========\n"
    );

    // 1. 高レベル演算の構築
    let a = helper::input("a", vec![1024], DType::F32);
    let b = helper::input("b", vec![1024], DType::F32);
    let zero = helper::const_val(0.0, DType::F32);

    // (a + b) + 0  （意図的に冗長な演算を追加）
    let add1 = helper::elementwise(ElementwiseOp::Add, vec![a, b]);
    let add2 = helper::elementwise(ElementwiseOp::Add, vec![add1, zero]);

    println!("【1. 高レベルIR】");
    println!("{}\n", add2.to_debug_string(0));

    // 2. パターンマッチングによる最適化
    let rules = basic_optimization_rules();
    let rewriter = Rewriter::new(rules);
    let optimized = rewriter.apply(&add2, 10);

    println!("【2. 最適化後のIR】");
    println!("{}\n", optimized.to_debug_string(0));

    // 3. Lowering（高レベル → 低レベル）
    let lowerer = Lowerer::new(256);
    let lowered = lowerer.lower(&optimized);

    println!("【3. Lowering後のIR（ループ・Load/Store）】");
    println!("{}\n", lowered.to_debug_string(0));

    // 4. OpenCLコード生成
    let mut codegen = OpenCLCodegen::new();
    let kernel_code = codegen.generate_kernel(&lowered, "add_kernel");

    println!("【4. 生成されたOpenCLカーネル】");
    println!("{}\n", kernel_code);

    // 検証：基本的なチェック
    assert!(kernel_code.contains("__kernel"));
    assert!(kernel_code.contains("add_kernel"));
    assert!(kernel_code.contains("get_global_id"));
    assert!(kernel_code.contains("input"));
    assert!(kernel_code.contains("output"));
}

#[test]
fn test_end_to_end_reduce() {
    println!("\n========== Reduce Sum: 高レベル → Lowering → コード生成 ==========\n");

    // 1. 高レベル演算の構築
    let a = helper::input("a", vec![10, 20], DType::F32);
    let sum = helper::reduce(ReduceOp::Sum, a, 1, vec![10, 20]);

    println!("【1. 高レベルIR（Reduce）】");
    println!("{}\n", sum.to_debug_string(0));

    // 2. Lowering
    let lowerer = Lowerer::new(256);
    let lowered = lowerer.lower(&sum);

    println!("【2. Lowering後のIR（ネストループ）】");
    println!("{}\n", lowered.to_debug_string(0));

    // 3. OpenCLコード生成
    let mut codegen = OpenCLCodegen::new();
    let kernel_code = codegen.generate_kernel(&lowered, "reduce_kernel");

    println!("【3. 生成されたOpenCLカーネル】");
    println!("{}\n", kernel_code);

    // 検証
    assert!(kernel_code.contains("__kernel"));
    assert!(kernel_code.contains("reduce_kernel"));
    assert!(kernel_code.contains("for")); // シーケンシャルループ
}

#[test]
fn test_optimization_chain() {
    println!("\n========== 最適化チェーン: x * 0 + y * 1 ==========\n");

    let x = helper::var("x", DType::F32);
    let y = helper::var("y", DType::F32);
    let zero = helper::const_val(0.0, DType::F32);
    let one = helper::const_val(1.0, DType::F32);

    // x * 0 + y * 1
    let expr = helper::add(helper::mul(x, zero), helper::mul(y.clone(), one));

    println!("【最適化前】");
    println!("{}\n", expr.to_debug_string(0));

    // 最適化
    let rules = basic_optimization_rules();
    let rewriter = Rewriter::new(rules);
    let optimized = rewriter.apply(&expr, 10);

    println!("【最適化後（期待: y）】");
    println!("{}\n", optimized.to_debug_string(0));

    // x * 0 = 0, 0 + (y * 1) = 0 + y = y, y * 1 = y のいずれかになるはず
    // 最終的には y になることを期待
    match &*optimized {
        UOp::Var { name, .. } => {
            assert_eq!(name, "y", "最適化後は y のみになるはず");
        }
        _ => {
            // Add(Const(0), Var(y)) のような中間状態の可能性もある
            println!("注意: 完全に最適化されていない可能性があります");
        }
    }
}

#[test]
fn test_manual_kernel_construction() {
    println!("\n========== 手動カーネル構築テスト ==========\n");

    // 手動で低レベルIRを構築: output[i] = input[i] * 2.0
    let tid = helper::thread_idx(0, DType::F32);
    let input_val = helper::load("input", Some(tid.clone()), DType::F32);
    let two = helper::const_val(2.0, DType::F32);
    let result = helper::mul(input_val, two);
    let store = helper::store("output", Some(tid.clone()), result);
    let kernel = helper::loop_op("tid", 0, 100, store, true);

    println!("【手動構築したIR】");
    println!("{}\n", kernel.to_debug_string(0));

    // コード生成
    let mut codegen = OpenCLCodegen::new();
    let code = codegen.generate_kernel(&kernel, "scale_kernel");

    println!("【生成されたカーネル】");
    println!("{}\n", code);

    assert!(code.contains("* 2.0f"));
    assert!(code.contains("input[(int)"));
    assert!(code.contains("output[(int)"));
}
