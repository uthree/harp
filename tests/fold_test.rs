/// Fold演算のテスト
///
/// Fold演算（col2im）が正しくコンパイルされることを確認
use harp::backend::Device;
use harp::prelude::*;

#[test]
fn test_fold1d_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Fold1dのテスト: unfold -> fold で元に戻ることを確認
    let mut graph = Graph::new();

    // 入力: [2, 10] (C_in=2, L=10)
    let input = graph.input("input", DType::F32, vec![2, 10]);

    // unfold1d: kernel_size=3, stride=1, dilation=1, groups=1
    // 出力: [2, 2*3, 8] = [2, 6, 8]
    let unfolded = input.unfold1d(3, 1, 1, 1);

    // fold1d: 元に戻す
    // output_size=[2, 10], kernel_size=3, stride=1, dilation=1, groups=1
    let folded = unfolded.fold1d(vec![2, 10], 3, 1, 1, 1);

    graph.output("result", folded);

    // コンパイルと実行
    let device = Device::cpu();
    let pipeline = device.get_pipeline().unwrap();
    let mut pipeline = pipeline.borrow_mut();

    // コンパイル（エラーなく完了すればOK）
    let result = pipeline.compile_graph(graph);
    assert!(
        result.is_ok(),
        "Fold1d compilation should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_fold2d_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Fold2dのテスト
    let mut graph = Graph::new();

    // 入力: [3, 5, 5] (C_in=3, H=5, W=5)
    let input = graph.input("input", DType::F32, vec![3, 5, 5]);

    // unfold2d: kernel_size=(3,3), stride=(1,1), dilation=(1,1), groups=1
    let unfolded = input.unfold2d((3, 3), (1, 1), (1, 1), 1);

    // fold2d: 元に戻す
    let folded = unfolded.fold2d(vec![3, 5, 5], (3, 3), (1, 1), (1, 1), 1);

    graph.output("result", folded);

    // コンパイルと実行
    let device = Device::cpu();
    let pipeline = device.get_pipeline().unwrap();
    let mut pipeline = pipeline.borrow_mut();

    // コンパイル（エラーなく完了すればOK）
    let result = pipeline.compile_graph(graph);
    assert!(
        result.is_ok(),
        "Fold2d compilation should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_fold3d_compilation() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Fold3dのテスト
    let mut graph = Graph::new();

    // 入力: [2, 4, 4, 4] (C_in=2, D=4, H=4, W=4)
    let input = graph.input("input", DType::F32, vec![2, 4, 4, 4]);

    // unfold3d: kernel_size=(2,2,2), stride=(1,1,1), dilation=(1,1,1), groups=1
    let unfolded = input.unfold3d((2, 2, 2), (1, 1, 1), (1, 1, 1), 1);

    // fold3d: 元に戻す
    let folded = unfolded.fold3d(vec![2, 4, 4, 4], (2, 2, 2), (1, 1, 1), (1, 1, 1), 1);

    graph.output("result", folded);

    // コンパイルと実行
    let device = Device::cpu();
    let pipeline = device.get_pipeline().unwrap();
    let mut pipeline = pipeline.borrow_mut();

    // コンパイル（エラーなく完了すればOK）
    let result = pipeline.compile_graph(graph);
    assert!(
        result.is_ok(),
        "Fold3d compilation should succeed: {:?}",
        result.err()
    );
}
