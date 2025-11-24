//! 自動微分のテスト

use super::Tensor;
use crate::graph::{DType, Graph};

#[test]
fn test_tensor_creation() {
    let mut graph = Graph::new();
    let node = graph
        .input("x")
        .with_dtype(DType::F32)
        .with_shape([2, 3])
        .build();

    let tensor = Tensor::from_graph_node(node, true);
    assert!(tensor.requires_grad());
    assert!(tensor.grad().is_none());
}

#[test]
fn test_simple_add() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let c = &a + &b;
    assert!(c.requires_grad());
}

#[test]
fn test_simple_mul() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let c = &a * &b;
    assert!(c.requires_grad());
}

#[test]
fn test_backward_add() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    // c = a + b
    let c = &a + &b;

    // loss = sum(c)
    let loss = c.sum(0);

    // backward
    loss.backward();

    // 勾配が計算されているか確認
    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
}

#[test]
fn test_backward_mul() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    // c = a * b
    let c = &a * &b;

    // loss = sum(c)
    let loss = c.sum(0);

    // backward
    loss.backward();

    // 勾配が計算されているか確認
    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
}

#[test]
fn test_backward_complex() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([4])
            .build(),
        true,
    );

    // y = 2 * x + 1
    let y = 2.0 * &x + 1.0;

    // loss = sum(y)
    let loss = y.sum(0);

    // backward
    loss.backward();

    // xの勾配が計算されているか確認
    assert!(x.grad().is_some());
}

#[test]
fn test_zero_grad() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = &x * 2.0;
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());

    x.zero_grad();
    assert!(x.grad().is_none());
}

#[test]
fn test_no_grad() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        false, // requires_grad=false
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let c = &a + &b;
    assert!(c.requires_grad()); // bがrequires_grad=trueなので

    let loss = c.sum(0);
    loss.backward();

    // aは勾配計算されない
    assert!(a.grad().is_none());
    // bは勾配計算される
    assert!(b.grad().is_some());
}

#[test]
fn test_recip() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = x.recip();
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_div() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    // c = a / b
    let c = &a / &b;
    let loss = c.sum(0);
    loss.backward();

    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
}

// === 高レベル演算のテスト ===

#[test]
fn test_square() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = x.square();
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_powi() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = x.powi(3);
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_min() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let c = a.min(&b);
    let loss = c.sum(0);
    loss.backward();

    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
}

#[test]
fn test_clamp() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );
    let min_val = Tensor::from_graph_node(
        graph
            .input("min")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        false,
    );
    let max_val = Tensor::from_graph_node(
        graph
            .input("max")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        false,
    );

    let y = x.clamp(&min_val, &max_val);
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_mean() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3, 4])
            .build(),
        true,
    );

    let y = x.mean(1);
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

// === 数学関数のテスト ===

#[test]
fn test_log2() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = x.log2();
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_exp2() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = x.exp2();
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_log() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = x.log();
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_exp() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = x.exp();
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_sin() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = x.sin();
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_cos() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = x.cos();
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_sqrt() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = x.sqrt();
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_rsqrt() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = x.rsqrt();
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

// === Pad/Slice演算のテスト ===

#[test]
fn test_pad() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    // パディングを追加
    let padded = x.pad(vec![(1, 1)], 0.0);
    let loss = padded.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_slice() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([10])
            .build(),
        true,
    );

    // スライスを取得
    let sliced = x.slice(vec![(2, 7)]);
    let loss = sliced.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_slice_pad_roundtrip() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([10])
            .build(),
        true,
    );

    // Slice -> Pad のラウンドトリップ
    // こちらはsliceの入力が定数shapeなので動作する
    let sliced = x.slice(vec![(2, 7)]);
    let padded = sliced.pad(vec![(1, 1)], 0.0);

    let loss = padded.sum(0);
    loss.backward();

    // 勾配が計算されているか確認
    assert!(x.grad().is_some());
}

#[test]
fn test_variance() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3, 4])
            .build(),
        true,
    );

    // 軸1の分散を計算
    let var = x.variance(1);
    let loss = var.sum(0);
    loss.backward();

    // xの勾配が計算されているか確認
    assert!(x.grad().is_some());
}

#[test]
fn test_abs_square() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    // abs_squareはsquareのエイリアスなので、squareと同じ挙動
    let y = x.abs_square();
    let loss = y.sum(0);
    loss.backward();

    // xの勾配が計算されているか確認
    assert!(x.grad().is_some());
}

#[test]
fn test_zeros() {
    let zeros = Tensor::zeros(vec![2, 3]);

    // 形状が正しいか確認
    assert_eq!(zeros.data.view.shape().len(), 2);
    // requires_gradはfalse（デフォルト）
    assert!(!zeros.requires_grad());
}

#[test]
fn test_ones() {
    let ones = Tensor::ones(vec![3, 4]);

    // 形状が正しいか確認
    assert_eq!(ones.data.view.shape().len(), 2);
    // requires_gradはfalse（デフォルト）
    assert!(!ones.requires_grad());
}

#[test]
fn test_full() {
    let tensor = Tensor::full(vec![2, 2], 5.0);

    // 形状が正しいか確認
    assert_eq!(tensor.data.view.shape().len(), 2);
    // requires_gradはfalse（デフォルト）
    assert!(!tensor.requires_grad());
}

#[test]
fn test_zeros_with_operations() {
    let zeros = Tensor::zeros(vec![3]);
    let ones = Tensor::ones(vec![3]);

    // ゼロテンソルと1テンソルの加算
    let result = &zeros + &ones;

    // 計算グラフが構築されているか確認
    assert_eq!(result.data.view.shape().len(), 1);
}

#[cfg(feature = "ndarray")]
#[test]
fn test_from_ndarray_shape() {
    use ndarray::Array2;

    let array = Array2::<f32>::zeros((2, 3));
    let tensor = Tensor::from_ndarray_shape(&array.into_dyn());

    // 形状が正しくコピーされているか確認
    assert_eq!(tensor.data.view.shape().len(), 2);
}

#[cfg(feature = "ndarray")]
#[test]
fn test_from_trait() {
    use ndarray::Array2;

    let array = Array2::<f32>::zeros((3, 4));
    let tensor: Tensor = array.into();

    // 形状が正しくコピーされているか確認
    assert_eq!(tensor.data.view.shape().len(), 2);
}

#[test]
fn test_tensor_device() {
    use crate::backend::Device;

    // デフォルトデバイスで作成
    let tensor = Tensor::zeros(vec![2, 3]);
    let device = tensor.device();
    assert!(device.is_available());
}

#[test]
fn test_tensor_to_device() {
    use crate::backend::Device;

    let tensor = Tensor::ones(vec![3, 4]);
    let original_device = tensor.device();

    // 別のデバイスに変更
    let new_device = Device::cpu();
    let tensor_cpu = tensor.to(new_device);

    // デバイスが変更されていることを確認
    assert_eq!(tensor_cpu.device(), new_device);

    // 元のテンソルは変更されていない
    assert_eq!(tensor.device(), original_device);
}

#[test]
fn test_operations_preserve_device() {
    use crate::backend::Device;

    let device = Device::cpu();
    let a = Tensor::ones_on(vec![3], device);
    let b = Tensor::zeros_on(vec![3], device);

    // 演算結果のデバイスは入力と同じ
    let c = &a + &b;
    assert_eq!(c.device(), device);

    let d = &a * &b;
    assert_eq!(d.device(), device);
}

#[test]
fn test_device_in_debug() {
    use crate::backend::Device;

    let tensor = Tensor::zeros_on(vec![2, 3], Device::cpu());
    let debug_str = format!("{:?}", tensor);

    // Debug出力にdeviceフィールドが含まれていることを確認
    assert!(debug_str.contains("device"));
}
