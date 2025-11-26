use super::super::*;
use crate::ast::AstNode;
use crate::graph::DType as GraphDType;

#[test]
fn test_lower_complex_add() {
    let _ = env_logger::builder().is_test(true).try_init();

    // (a+bi) + (c+di) の複素数加算
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::Complex)
        .with_shape(vec![10])
        .build();
    let b = graph
        .input("b")
        .with_dtype(GraphDType::Complex)
        .with_shape(vec![10])
        .build();
    let result = &a + &b;

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック:
    // input0_re, input0_im, input1_re, input1_im, output_re, output_im
    if let AstNode::Function { params, .. } = &function {
        // 2入力 × 2 (re/im) + 1出力 × 2 (re/im) = 6
        assert_eq!(params.len(), 6, "Expected 6 parameters for complex add");
        assert_eq!(params[0].name, "input0_re");
        assert_eq!(params[1].name, "input0_im");
        assert_eq!(params[2].name, "input1_re");
        assert_eq!(params[3].name, "input1_im");
        assert_eq!(params[4].name, "output_re");
        assert_eq!(params[5].name, "output_im");
    } else {
        panic!("Expected AstNode::Function");
    }

    // 生成されたコードを表示（テスト実行時に確認用）
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_lower_complex_add ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_complex_mul() {
    let _ = env_logger::builder().is_test(true).try_init();

    // (a+bi) * (c+di) の複素数乗算
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::Complex)
        .with_shape(vec![10])
        .build();
    let b = graph
        .input("b")
        .with_dtype(GraphDType::Complex)
        .with_shape(vec![10])
        .build();
    let result = &a * &b;

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 6, "Expected 6 parameters for complex mul");
        assert_eq!(params[0].name, "input0_re");
        assert_eq!(params[1].name, "input0_im");
        assert_eq!(params[2].name, "input1_re");
        assert_eq!(params[3].name, "input1_im");
        assert_eq!(params[4].name, "output_re");
        assert_eq!(params[5].name, "output_im");
    } else {
        panic!("Expected AstNode::Function");
    }

    // 生成されたコードを表示
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_lower_complex_mul ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_complex_neg() {
    let _ = env_logger::builder().is_test(true).try_init();

    // -(a+bi) の複素数否定
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::Complex)
        .with_shape(vec![10])
        .build();
    let result = -&a;

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック:
    // input0_re, input0_im, output_re, output_im
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 4, "Expected 4 parameters for complex neg");
        assert_eq!(params[0].name, "input0_re");
        assert_eq!(params[1].name, "input0_im");
        assert_eq!(params[2].name, "output_re");
        assert_eq!(params[3].name, "output_im");
    } else {
        panic!("Expected AstNode::Function");
    }

    // 生成されたコードを表示
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_lower_complex_neg ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_complex_recip() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 1/(a+bi) の複素数逆数
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::Complex)
        .with_shape(vec![10])
        .build();
    let result = a.recip();

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック:
    // input0_re, input0_im, output_re, output_im
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 4, "Expected 4 parameters for complex recip");
        assert_eq!(params[0].name, "input0_re");
        assert_eq!(params[1].name, "input0_im");
        assert_eq!(params[2].name, "output_re");
        assert_eq!(params[3].name, "output_im");
    } else {
        panic!("Expected AstNode::Function");
    }

    // 生成されたコードを表示
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_lower_complex_recip ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_complex_with_constant() {
    let _ = env_logger::builder().is_test(true).try_init();

    // (a+bi) + (1+2i) 複素数定数との加算
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::Complex)
        .with_shape(vec![10])
        .build();
    let constant = GraphNode::complex_constant(1.0, 2.0);
    let result = &a + &constant;

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック:
    // 入力は1つの複素数バッファのみ (定数はインライン化される)
    // input0_re, input0_im, output_re, output_im
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(
            params.len(),
            4,
            "Expected 4 parameters for complex add with constant"
        );
        assert_eq!(params[0].name, "input0_re");
        assert_eq!(params[1].name, "input0_im");
        assert_eq!(params[2].name, "output_re");
        assert_eq!(params[3].name, "output_im");
    } else {
        panic!("Expected AstNode::Function");
    }

    // 生成されたコードを表示
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_lower_complex_with_constant ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_complex_scalar() {
    let _ = env_logger::builder().is_test(true).try_init();

    // スカラー複素数の演算
    let a = GraphNode::complex_constant(1.0, 2.0);
    let b = GraphNode::complex_constant(3.0, 4.0);
    let result = &a * &b;

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // 出力のみのパラメータ（定数はインライン化）
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 2, "Expected 2 parameters for scalar complex");
        assert_eq!(params[0].name, "output_re");
        assert_eq!(params[1].name, "output_im");
    } else {
        panic!("Expected AstNode::Function");
    }

    // 生成されたコードを表示
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_lower_complex_scalar ===\n{}\n",
        code
    );
}
