use super::super::*;
use crate::ast::DType as AstDType;
use crate::graph::DType as GraphDType;

#[test]
fn test_lower_reduce_sum_1d() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 1次元テンソルの合計（スカラーに縮約）
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let result = a.reduce_sum(0);

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0, output, shape0
    assert_eq!(function.params.len(), 3);
    assert_eq!(function.params[0].name, "input0");
    assert_eq!(function.params[1].name, "output");
    assert_eq!(function.params[2].name, "shape0");

    // 返り値の型はunit型
    assert_eq!(function.return_type, AstDType::Tuple(vec![]));

    // 生成されたコードを表示（テスト実行時に確認用）
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function("reduce_sum_kernel", &function);
    eprintln!(
        "\n=== Generated Code for test_lower_reduce_sum_1d ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_reduce_sum_2d() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 2次元テンソルの軸1方向の合計
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![3, 4])
        .build();
    let result = a.reduce_sum(1); // (3, 4) -> (3,)

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0, output, shape0, shape1
    assert_eq!(function.params.len(), 4);
    assert_eq!(function.params[0].name, "input0");
    assert_eq!(function.params[1].name, "output");
    assert_eq!(function.params[2].name, "shape0");
    assert_eq!(function.params[3].name, "shape1");

    // 生成されたコードを表示
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function("reduce_sum_2d_kernel", &function);
    eprintln!(
        "\n=== Generated Code for test_lower_reduce_sum_2d ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_reduce_sum_axis0() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 2次元テンソルの軸0方向の合計
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![3, 4])
        .build();
    let result = a.reduce_sum(0); // (3, 4) -> (4,)

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック
    assert_eq!(function.params.len(), 4);

    // 生成されたコードを表示
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function("reduce_sum_axis0_kernel", &function);
    eprintln!(
        "\n=== Generated Code for test_lower_reduce_sum_axis0 ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_reduce_max() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 1次元テンソルの最大値（スカラーに縮約）
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![10])
        .build();
    let result = a.reduce_max(0);

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック
    assert_eq!(function.params.len(), 3);
    assert_eq!(function.params[0].name, "input0");
    assert_eq!(function.params[1].name, "output");

    // 生成されたコードを表示
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function("reduce_max_kernel", &function);
    eprintln!(
        "\n=== Generated Code for test_lower_reduce_max ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_reduce_mul() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 2次元テンソルの積縮約
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![5, 6])
        .build();
    let result = a.reduce_mul(1); // (5, 6) -> (5,)

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック
    assert_eq!(function.params.len(), 4);

    // 生成されたコードを表示
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function("reduce_mul_kernel", &function);
    eprintln!(
        "\n=== Generated Code for test_lower_reduce_mul ===\n{}\n",
        code
    );
}

#[test]
fn test_lower_reduce_3d() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 3次元テンソルの縮約
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(GraphDType::F32)
        .with_shape(vec![2, 3, 4])
        .build();
    let result = a.reduce_sum(1); // (2, 3, 4) -> (2, 4)

    // カーネル関数を生成
    let mut lowerer = Lowerer::new();
    let function = lowerer.lower_node_to_kernel(&result, 0);

    assert!(function.is_ok());
    let function = function.unwrap();

    // パラメータをチェック: input0, output, shape0, shape1, shape2
    assert_eq!(function.params.len(), 5);

    // 生成されたコードを表示
    use crate::backend::metal::MetalRenderer;
    let mut renderer = MetalRenderer::new();
    let code = renderer.render_function("reduce_3d_kernel", &function);
    eprintln!(
        "\n=== Generated Code for test_lower_reduce_3d ===\n{}\n",
        code
    );
}
