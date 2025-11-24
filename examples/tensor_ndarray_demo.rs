//! Tensor ⇔ ndarray 変換デモ
//!
//! ndarray feature を有効にした場合の使用例
//! 実行方法: cargo run --example tensor_ndarray_demo --features ndarray

#[cfg(feature = "ndarray")]
fn main() {
    use harp::autograd::Tensor;
    use ndarray::{Array1, Array2, arr1, arr2};

    env_logger::init();

    println!("=== Tensor ⇔ ndarray Conversion Demo ===\n");

    // 1. ndarrayからTensorへの変換（From trait）
    println!("1. ndarray → Tensor conversion using From trait:");
    let array1d = arr1(&[1.0, 2.0, 3.0, 4.0]);
    let tensor1d: Tensor = array1d.into();
    println!("  Array1 shape: [4]");
    println!("  Tensor shape: {:?}", tensor1d.data.view.shape());
    println!();

    let array2d = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let tensor2d: Tensor = array2d.into();
    println!("  Array2 shape: [2, 3]");
    println!("  Tensor shape: {:?}", tensor2d.data.view.shape());
    println!();

    // 2. from_ndarray_shape() を使った変換
    println!("2. Using from_ndarray_shape():");
    let array = Array2::<f32>::zeros((3, 4));
    let tensor = Tensor::from_ndarray_shape(&array.into_dyn());
    println!("  Source array shape: [3, 4]");
    println!("  Tensor shape: {:?}", tensor.data.view.shape());
    println!();

    // 3. from_ndarray() を使った変換
    println!("3. Using from_ndarray():");
    let array = Array1::<f32>::from(vec![10.0, 20.0, 30.0]);
    let tensor = Tensor::from_ndarray(array.into_dyn());
    println!("  Source array: [10.0, 20.0, 30.0]");
    println!("  Tensor shape: {:?}", tensor.data.view.shape());
    println!();

    // 4. 演算の例
    println!("4. Operations with converted tensors:");
    let array_a = arr1(&[1.0, 2.0, 3.0]);
    let array_b = arr1(&[4.0, 5.0, 6.0]);

    let t1: Tensor = array_a.into();
    let t2: Tensor = array_b.into();
    let t3 = &t1 + &t2;

    println!("  array_a: [1.0, 2.0, 3.0]");
    println!("  array_b: [4.0, 5.0, 6.0]");
    println!("  tensor1 + tensor2");
    println!("  Result shape: {:?}", t3.data.view.shape());
    println!();

    // 5. 注意事項
    println!("5. Important notes:");
    println!("  ⚠ Current implementation only copies shape information");
    println!("  ⚠ Data values are NOT transferred (tensors are zero-initialized)");
    println!("  ⚠ For actual data transfer, use realize() with appropriate inputs");
    println!();

    println!("=== Demo Complete ===");
}

#[cfg(not(feature = "ndarray"))]
fn main() {
    println!("This example requires the 'ndarray' feature to be enabled.");
    println!("Run with: cargo run --example tensor_ndarray_demo --features ndarray");
}
