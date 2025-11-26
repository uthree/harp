use super::super::*;
use crate::ast::AstNode;

#[test]
fn test_lower_arange_basic() {
    let _ = env_logger::builder().is_test(true).try_init();

    // arange(5) -> [0, 1, 2, 3, 4]
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

    // コードにキャストが含まれることを確認（C++スタイル: float(x)）
    assert!(
        code.contains("float(ridx0)"),
        "Should cast index to float: {}",
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
fn test_arange_with_offset() {
    let _ = env_logger::builder().is_test(true).try_init();

    // arange(5) + 10 -> [10, 11, 12, 13, 14]
    let result = GraphNode::arange(5) + 10.0f32;

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());

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

    // arange(5) * 0.5 -> [0.0, 0.5, 1.0, 1.5, 2.0]
    let result = GraphNode::arange(5) * 0.5f32;

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());

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

    // arange(5) * 2.0 + 10.0 -> [10.0, 12.0, 14.0, 16.0, 18.0]
    let result = GraphNode::arange(5) * 2.0f32 + 10.0f32;

    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());

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
