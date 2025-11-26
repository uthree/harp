use super::super::*;
use crate::ast::AstNode;
use crate::graph::DType;

#[test]
fn test_lower_arange_basic() {
    let _ = env_logger::builder().is_test(true).try_init();

    // arange(5) -> [0, 1, 2, 3, 4] (I32)
    let result = GraphNode::arange(5);

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: output のみ（shapeは定数）
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 1, "Expected 1 parameter for arange");
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
        "\n=== Generated Code for test_lower_arange_basic ===\n{}\n",
        code
    );

    // 出力がint*型であることを確認
    assert!(
        code.contains("int* output"),
        "Should have int* output: {}",
        code
    );
    // インデックスをそのまま使用していることを確認
    assert!(
        code.contains("ridx0"),
        "Should use ridx0 directly: {}",
        code
    );
}

#[test]
fn test_lower_arange_with_shape_var() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 動的サイズのarange
    use crate::graph::shape::Expr;
    let n = Expr::Var("n".to_string());
    let result = GraphNode::arange(n);

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: output, n
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(
            params.len(),
            2,
            "Expected 2 parameters for arange with shape var"
        );
        assert_eq!(params[0].name, "output");
        assert_eq!(params[1].name, "n");
    } else {
        panic!("Expected AstNode::Function");
    }

    // 生成されたコードを表示
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_lower_arange_with_shape_var ===\n{}\n",
        code
    );

    // コードにnが含まれることを確認
    assert!(code.contains("n"), "Should use shape variable n: {}", code);
}

#[test]
fn test_arange_cast_to_float() {
    let _ = env_logger::builder().is_test(true).try_init();

    // arange(5).cast(F32) -> [0.0, 1.0, 2.0, 3.0, 4.0]
    let result = GraphNode::arange(5).cast(DType::F32);

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok(), "Failed to lower cast: {:?}", function);

    // 生成されたコードを表示
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function.unwrap());
    eprintln!(
        "\n=== Generated Code for test_arange_cast_to_float ===\n{}\n",
        code
    );

    // floatへのキャストが含まれることを確認
    assert!(
        code.contains("float(") || code.contains("(float)"),
        "Should cast to float: {}",
        code
    );
}

#[test]
fn test_arange_with_offset() {
    let _ = env_logger::builder().is_test(true).try_init();

    // arange(5).cast(F32) + 10.0 -> [10.0, 11.0, 12.0, 13.0, 14.0]
    let result = GraphNode::arange(5).cast(DType::F32) + 10.0f32;

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok(), "Failed to lower: {:?}", function);

    // 生成されたコードを表示
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function.unwrap());
    eprintln!(
        "\n=== Generated Code for test_arange_with_offset ===\n{}\n",
        code
    );
}

#[test]
fn test_arange_with_scale() {
    let _ = env_logger::builder().is_test(true).try_init();

    // arange(5).cast(F32) * 0.5 -> [0.0, 0.5, 1.0, 1.5, 2.0]
    let result = GraphNode::arange(5).cast(DType::F32) * 0.5f32;

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok(), "Failed to lower: {:?}", function);

    // 生成されたコードを表示
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function.unwrap());
    eprintln!(
        "\n=== Generated Code for test_arange_with_scale ===\n{}\n",
        code
    );
}

#[test]
fn test_arange_combined() {
    let _ = env_logger::builder().is_test(true).try_init();

    // arange(5).cast(F32) * 2.0 + 10.0 -> [10.0, 12.0, 14.0, 16.0, 18.0]
    let result = GraphNode::arange(5).cast(DType::F32) * 2.0f32 + 10.0f32;

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok(), "Failed to lower: {:?}", function);

    // 生成されたコードを表示
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function.unwrap());
    eprintln!(
        "\n=== Generated Code for test_arange_combined ===\n{}\n",
        code
    );
}

#[test]
fn test_arange_int_arithmetic() {
    let _ = env_logger::builder().is_test(true).try_init();

    // arange(5) + 10 (I32演算) -> [10, 11, 12, 13, 14]
    let result = GraphNode::arange(5) + GraphNode::constant(10isize);

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(
        function.is_ok(),
        "Failed to lower I32 arithmetic: {:?}",
        function
    );

    // 生成されたコードを表示
    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function.unwrap());
    eprintln!(
        "\n=== Generated Code for test_arange_int_arithmetic ===\n{}\n",
        code
    );
}
