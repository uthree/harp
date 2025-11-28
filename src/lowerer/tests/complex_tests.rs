use super::super::*;
use crate::ast::AstNode;
use crate::graph::DType as GraphDType;

#[test]
fn test_lower_complex_add() {
    let _ = env_logger::builder().is_test(true).try_init();

    // (a+bi) + (c+di) の複素数加算
    let mut graph = Graph::new();
    let a = graph.input("a", GraphDType::Complex, vec![10]);
    let b = graph.input("b", GraphDType::Complex, vec![10]);
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
    let a = graph.input("a", GraphDType::Complex, vec![10]);
    let b = graph.input("b", GraphDType::Complex, vec![10]);
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
    let a = graph.input("a", GraphDType::Complex, vec![10]);
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
    let a = graph.input("a", GraphDType::Complex, vec![10]);
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
    let a = graph.input("a", GraphDType::Complex, vec![10]);
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
    let a = graph.input("a", GraphDType::Complex, vec![4]);
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

#[test]
fn test_lower_complex_real() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 複素数テンソルから実部を取り出す
    let mut graph = Graph::new();
    let z = graph.input("z", GraphDType::Complex, vec![10]);
    let result = z.real();

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0 (Complex F32*), output (F32*)
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 2, "Expected 2 parameters for real extraction");
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

    // 実部はoffset * 2でアクセス
    assert!(code.contains("* 2)"), "Should use * 2 for real part offset");

    eprintln!(
        "\n=== Generated Code for test_lower_complex_real ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_complex_imag() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 複素数テンソルから虚部を取り出す
    let mut graph = Graph::new();
    let z = graph.input("z", GraphDType::Complex, vec![10]);
    let result = z.imag();

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0 (Complex F32*), output (F32*)
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(params.len(), 2, "Expected 2 parameters for imag extraction");
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

    // 虚部はoffset * 2 + 1でアクセス
    assert!(
        code.contains("* 2) + 1)"),
        "Should use * 2 + 1 for imaginary part offset"
    );

    eprintln!(
        "\n=== Generated Code for test_lower_complex_imag ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_complex_from_parts() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 実部と虚部のF32テンソルから複素数テンソルを構築
    let mut graph = Graph::new();
    let re = graph.input("re", GraphDType::F32, vec![10]);
    let im = graph.input("im", GraphDType::F32, vec![10]);
    let result = GraphNode::complex_from_parts(re, im);

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0 (F32*), input1 (F32*), output (Complex F32*)
    if let AstNode::Function { params, .. } = &function {
        assert_eq!(
            params.len(),
            3,
            "Expected 3 parameters for complex_from_parts"
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

    // 出力はインターリーブレイアウト: real at * 2, imag at * 2 + 1
    assert!(
        code.contains("* 2)"),
        "Should use * 2 for real part output offset"
    );
    assert!(
        code.contains("* 2) + 1)"),
        "Should use * 2 + 1 for imaginary part output offset"
    );

    eprintln!(
        "\n=== Generated Code for test_lower_complex_from_parts ===\n{}\n",
        code
    );
}

#[test]
fn test_complex_roundtrip() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 複素数 -> 実部/虚部 -> 複素数の往復変換
    // z = complex_from_parts(z.real(), z.imag()) であることを構造的に確認
    let mut graph = Graph::new();
    let z = graph.input("z", GraphDType::Complex, vec![10]);

    let re = z.real();
    let im = z.imag();
    let reconstructed = GraphNode::complex_from_parts(re, im);

    // 型の確認
    assert_eq!(z.dtype, GraphDType::Complex);
    assert_eq!(reconstructed.dtype, GraphDType::Complex);

    // shapeの確認
    assert_eq!(z.view.shape(), reconstructed.view.shape());

    // 各中間ノードの型確認
    // z.real() -> F32
    let z_real = z.real();
    assert_eq!(z_real.dtype, GraphDType::F32);
    assert_eq!(z_real.view.shape(), z.view.shape());

    // z.imag() -> F32
    let z_imag = z.imag();
    assert_eq!(z_imag.dtype, GraphDType::F32);
    assert_eq!(z_imag.view.shape(), z.view.shape());
}

#[test]
fn test_real_imag_multidimensional() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 多次元複素数テンソルのreal/imag
    let mut graph = Graph::new();
    let z = graph.input("z", GraphDType::Complex, vec![4, 5, 6]);

    let re = z.real();
    let im = z.imag();

    // 型の確認
    assert_eq!(re.dtype, GraphDType::F32);
    assert_eq!(im.dtype, GraphDType::F32);

    // shapeの確認（複素数のshapeと同じ）
    assert_eq!(
        re.view.shape(),
        &[
            crate::graph::Expr::from(4),
            crate::graph::Expr::from(5),
            crate::graph::Expr::from(6)
        ]
    );
    assert_eq!(re.view.shape(), im.view.shape());

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&re, 0).unwrap();

    if let AstNode::Function { params, .. } = &function {
        // 静的shapeの場合: input0, output のみ
        // 動的shapeの場合: input0, output, shape0, shape1, shape2
        assert!(
            params.len() >= 2,
            "Expected at least 2 parameters (input0, output)"
        );
    }

    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_real_imag_multidimensional ===\n{}\n",
        code
    );
}

#[test]
fn test_complex_from_parts_multidimensional() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 多次元テンソルからの複素数構築
    let mut graph = Graph::new();
    let re = graph.input("re", GraphDType::F32, vec![4, 5, 6]);
    let im = graph.input("im", GraphDType::F32, vec![4, 5, 6]);
    let z = GraphNode::complex_from_parts(re, im);

    // 型の確認
    assert_eq!(z.dtype, GraphDType::Complex);

    // shapeの確認
    assert_eq!(
        z.view.shape(),
        &[
            crate::graph::Expr::from(4),
            crate::graph::Expr::from(5),
            crate::graph::Expr::from(6)
        ]
    );

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&z, 0).unwrap();

    use crate::backend::c::CRenderer;
    use crate::backend::c_like::CLikeRenderer;
    let mut renderer = CRenderer::new();
    let code = renderer.render_function_node(&function);
    eprintln!(
        "\n=== Generated Code for test_complex_from_parts_multidimensional ===\n{}\n",
        code
    );
}

#[test]
#[should_panic(expected = "real() can only be applied to Complex tensors")]
fn test_real_on_non_complex_panics() {
    let mut graph = Graph::new();
    let f = graph.input("f", GraphDType::F32, vec![10]);
    let _ = f.real(); // should panic
}

#[test]
#[should_panic(expected = "imag() can only be applied to Complex tensors")]
fn test_imag_on_non_complex_panics() {
    let mut graph = Graph::new();
    let f = graph.input("f", GraphDType::F32, vec![10]);
    let _ = f.imag(); // should panic
}

#[test]
#[should_panic(expected = "real part must be F32")]
fn test_complex_from_parts_wrong_real_type_panics() {
    let mut graph = Graph::new();
    let re = graph.input("re", GraphDType::I32, vec![10]);
    let im = graph.input("im", GraphDType::F32, vec![10]);
    let _ = GraphNode::complex_from_parts(re, im); // should panic
}

#[test]
#[should_panic(expected = "imag part must be F32")]
fn test_complex_from_parts_wrong_imag_type_panics() {
    let mut graph = Graph::new();
    let re = graph.input("re", GraphDType::F32, vec![10]);
    let im = graph.input("im", GraphDType::I32, vec![10]);
    let _ = GraphNode::complex_from_parts(re, im); // should panic
}

#[test]
#[should_panic(expected = "real and imag must have the same shape")]
fn test_complex_from_parts_shape_mismatch_panics() {
    let mut graph = Graph::new();
    let re = graph.input("re", GraphDType::F32, vec![10]);
    let im = graph.input("im", GraphDType::F32, vec![20]);
    let _ = GraphNode::complex_from_parts(re, im); // should panic
}
