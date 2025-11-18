use super::super::*;
use crate::ast::DType as AstDType;
use crate::graph::DType as GraphDType;

#[test]
fn test_lower_simple_add() {
    let _ = env_logger::builder().is_test(true).try_init();

    // a + b のグラフをカーネルに変換
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let b = graph
        .input("b")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let result = a + b;

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0, input1, output (shapeは定数なのでパラメータ不要)
    use crate::ast::AstNode;
    if let AstNode::Function {
        params,
        return_type,
        ..
    } = &function
    {
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].name, "input0");
        assert_eq!(params[1].name, "input1");
        assert_eq!(params[2].name, "output");

        // 返り値の型はunit型
        assert_eq!(*return_type, AstDType::Tuple(vec![]));
    } else {
        panic!("Expected AstNode::Function");
    }

    // 生成されたコードを表示（テスト実行時に確認用）
    use crate::backend::c_like::CLikeRenderer;
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_lower_simple_add ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_simple_mul() {
    // a * b のグラフをカーネルに変換
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![20])
        .build();
    let b = graph
        .input("b")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![20])
        .build();
    let result = a * b;

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0, input1, output (shapeは定数なのでパラメータ不要)
    use crate::ast::AstNode;
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].name, "input0");
        assert_eq!(params[1].name, "input1");
        assert_eq!(params[2].name, "output");
    } else {
        panic!("Expected AstNode::Function");
    }
}

#[test]
fn test_lower_neg() {
    // -a のグラフをカーネルに変換
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let result = -a;

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0, output (shapeは定数なのでパラメータ不要)
    use crate::ast::AstNode;
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].name, "input0");
        assert_eq!(params[1].name, "output");
    } else {
        panic!("Expected AstNode::Function");
    }
}

#[test]
fn test_lower_with_permute() {
    use crate::graph::ops::GraphOp;

    // 転置されたテンソルの加算
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![3, 4])
        .build();
    let _b = graph
        .input("b")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![3, 4])
        .build();

    // aを転置: (3, 4) -> (4, 3)
    let a_transposed = GraphNode::new(
        GraphDType::F32,
        GraphOp::View(a.view.clone().permute(vec![1, 0])),
        vec![a.clone()],
        a.view.clone().permute(vec![1, 0]),
    );

    // 転置されたaと同じshapeのbの加算は失敗するはず
    // （ここでは単純に転置されたViewの動作をテスト）
    let mut lowerer = Lowerer::new();

    // 転置されたテンソルのloweringをテスト
    let function = lowerer.lower_node_to_kernel(&a_transposed, 0);

    // Viewノードは直接lowering対象ではないのでエラーになる
    assert!(function.is_err());
}

#[test]
fn test_lower_with_flipped_view() {
    let _ = env_logger::builder().is_test(true).try_init();

    // flipされたテンソルの否定演算
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();

    // aをflip
    let flipped_view = a.view.clone().flip(0);

    // flipされたaの否定演算
    // Viewの変更を直接Elementwise演算のsrcに含める
    let a_flipped = GraphNode::new(GraphDType::F32, a.op.clone(), a.src.clone(), flipped_view);

    let result = -a_flipped;

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    // View変換が実装されたので成功するはず
    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0, output (shapeは定数なのでパラメータ不要)
    use crate::ast::AstNode;
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 2);
    } else {
        panic!("Expected AstNode::Function");
    }

    // 生成されたコードを表示（テスト実行時に確認用）
    use crate::backend::c_like::CLikeRenderer;
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_lower_with_flipped_view ===\n{}\n",
        code
    );
}

#[test]
#[cfg(target_os = "macos")]
#[serial_test::serial]
fn test_end_to_end_execution() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 手動でMetalカーネルを作成（lowererの出力を参考に）
    // 後でlowererと統合する予定
    let source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void test_add(
    device const float* input0 [[buffer(0)]],
    device const float* input1 [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = input0[tid] + input1[tid];
}
"#;

    eprintln!("\n=== Metal Kernel ===\n{}\n", source);

    // Metal compilerで実行
    use crate::backend::Compiler;
    use crate::backend::metal::{MetalCode, MetalCompiler};
    if let Some(mut compiler) = MetalCompiler::with_default_device() {
        let code = MetalCode::new(source.to_string());
        let mut kernel = compiler.compile(&code);

        // バッファを作成
        let mut input0_buffer = compiler.create_buffer(vec![10], 4);
        let mut input1_buffer = compiler.create_buffer(vec![10], 4);
        let output_buffer = compiler.create_buffer(vec![10], 4);

        // 入力データを設定
        let input0_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let input1_data: Vec<f32> = (0..10).map(|i| (i * 2) as f32).collect();

        input0_buffer.write_data(&input0_data);
        input1_buffer.write_data(&input1_data);

        // グリッドサイズを設定
        kernel.set_grid_size(10, 1, 1);

        // カーネルを実行
        kernel
            .dispatch(&[&input0_buffer, &input1_buffer, &output_buffer])
            .unwrap();

        // 結果を読み出し
        let mut output_data = vec![0.0f32; 10];
        output_buffer.read_data(&mut output_data);

        // 確認
        let expected: Vec<f32> = input0_data
            .iter()
            .zip(input1_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();

        eprintln!("Input 0: {:?}", input0_data);
        eprintln!("Input 1: {:?}", input1_data);
        eprintln!("Output:  {:?}", output_data);
        eprintln!("Expected: {:?}", expected);

        assert_eq!(output_data, expected);
        eprintln!("\n✅ End-to-end execution successful!\n");
    } else {
        eprintln!("⚠️ Metal not available, skipping test");
    }
}

#[test]
fn test_lower_add_with_simd() {
    let _ = env_logger::builder().is_test(true).try_init();

    use crate::graph::{
        ElementwiseStrategy, Expr, GraphNode, View,
        ops::{ElementwiseOp, GraphOp},
    };

    // a + b のグラフをSIMD化してカーネルに変換
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![16])
        .build();
    let b = graph
        .input("b")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![16])
        .build();

    // SIMD幅を4に設定したAdd演算ノードを直接作成
    let view = View::contiguous(vec![Expr::from(16)]);
    let simd_node = GraphNode::with_elementwise_strategies(
        GraphDType::F32,
        GraphOp::Elementwise {
            op: ElementwiseOp::Add,
            elementwise_strategies: None,
        },
        vec![a, b],
        view,
        vec![ElementwiseStrategy::sequential_simd(4)],
    );

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&simd_node, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // 生成されたコードを表示（MetalRendererを使用）
    use crate::backend::c_like::CLikeRenderer;
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function_node(&function);
    println!("\nGenerated SIMD code:\n{}", code);

    // ベクトル型（float4）が含まれることを確認
    assert!(
        code.contains("float4"),
        "Generated code should contain float4 vector type"
    );
}

#[test]
fn test_lower_add_with_simd_remainder() {
    let _ = env_logger::builder().is_test(true).try_init();

    use crate::graph::{
        ElementwiseStrategy, Expr, GraphNode, View,
        ops::{ElementwiseOp, GraphOp},
    };

    // a + b のグラフをSIMD化してカーネルに変換（サイズが4で割り切れない場合）
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![17]) // 4で割り切れない
        .build();
    let b = graph
        .input("b")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![17])
        .build();

    // SIMD幅を4に設定したAdd演算ノードを直接作成
    let view = View::contiguous(vec![Expr::from(17)]);
    let simd_node = GraphNode::with_elementwise_strategies(
        GraphDType::F32,
        GraphOp::Elementwise {
            op: ElementwiseOp::Add,
            elementwise_strategies: None,
        },
        vec![a, b],
        view,
        vec![ElementwiseStrategy::sequential_simd(4)],
    );

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&simd_node, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // 生成されたコードを表示（MetalRendererを使用）
    use crate::backend::c_like::CLikeRenderer;
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function_node(&function);
    println!("\nGenerated SIMD code with remainder:\n{}", code);

    // ベクトル型（float4）とスカラー型（float）の両方が含まれることを確認
    assert!(
        code.contains("float4"),
        "Generated code should contain float4 vector type"
    );

    // 2つのループがあることを確認（メインループと残りループ）
    let loop_count = code.matches("for (").count();
    assert_eq!(
        loop_count, 2,
        "Generated code should have 2 loops (main + remainder)"
    );
}

#[test]
fn test_lower_add_with_simd_openmp() {
    let _ = env_logger::builder().is_test(true).try_init();

    use crate::graph::{
        ElementwiseStrategy, Expr, GraphNode, View,
        ops::{ElementwiseOp, GraphOp},
    };

    // a + b のグラフをSIMD化してカーネルに変換（OpenMP用）
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![16])
        .build();
    let b = graph
        .input("b")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![16])
        .build();

    // SIMD幅を4に設定したAdd演算ノードを直接作成
    let view = View::contiguous(vec![Expr::from(16)]);
    let simd_node = GraphNode::with_elementwise_strategies(
        GraphDType::F32,
        GraphOp::Elementwise {
            op: ElementwiseOp::Add,
            elementwise_strategies: None,
        },
        vec![a, b],
        view,
        vec![ElementwiseStrategy::sequential_simd(4)],
    );

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&simd_node, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // 生成されたコードを表示（OpenMP CRendererを使用）
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function);
    println!("\nGenerated OpenMP SIMD code:\n{}", code);

    // ベクトル型（float4）が含まれることを確認
    assert!(
        code.contains("float4"),
        "Generated code should contain float4 vector type"
    );

    // C言語のキャスト構文が使われていることを確認
    assert!(
        code.contains("*(float4*)"),
        "Generated code should use C-style cast for vector load"
    );
}
