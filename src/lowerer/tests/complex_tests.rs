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
    // インターリーブレイアウト: input0, input1, output（それぞれ1つのF32*バッファ）
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(
            params.len(),
            3,
            "Expected 3 parameters for complex add (interleaved layout)"
        );
        assert_eq!(params[0].name, "input0");
        assert_eq!(params[1].name, "input1");
        assert_eq!(params[2].name, "output");
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
        assert_eq!(
            params.len(),
            3,
            "Expected 3 parameters for complex mul (interleaved layout)"
        );
        assert_eq!(params[0].name, "input0");
        assert_eq!(params[1].name, "input1");
        assert_eq!(params[2].name, "output");
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
    // input0, output
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(
            params.len(),
            2,
            "Expected 2 parameters for complex neg (interleaved layout)"
        );
        assert_eq!(params[0].name, "input0");
        assert_eq!(params[1].name, "output");
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
    // input0, output
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(
            params.len(),
            2,
            "Expected 2 parameters for complex recip (interleaved layout)"
        );
        assert_eq!(params[0].name, "input0");
        assert_eq!(params[1].name, "output");
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
    // input0, output
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(
            params.len(),
            2,
            "Expected 2 parameters for complex add with constant (interleaved layout)"
        );
        assert_eq!(params[0].name, "input0");
        assert_eq!(params[1].name, "output");
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
        assert_eq!(
            params.len(),
            1,
            "Expected 1 parameter for scalar complex (interleaved layout)"
        );
        assert_eq!(params[0].name, "output");
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

#[test]
fn test_complex_interleaved_layout() {
    let _ = env_logger::builder().is_test(true).try_init();

    // インターリーブレイアウトの検証
    // 複素数配列 [z0, z1, z2] はメモリ上で [re0, im0, re1, im1, re2, im2] となる
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::Complex)
        .with_shape(vec![4])
        .build();
    let result = -&a;

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0).unwrap();

    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function);

    // インデックスが * 2 と * 2 + 1 になっていることを確認
    assert!(code.contains("* 2)"), "Should use * 2 for real part offset");
    assert!(
        code.contains("* 2) + 1)"),
        "Should use * 2 + 1 for imaginary part offset"
    );

    eprintln!(
        "\n=== Generated Code for test_complex_interleaved_layout ===\n{}\n",
        code
    );
}
