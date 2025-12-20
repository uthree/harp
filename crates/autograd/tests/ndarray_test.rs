use autograd::Variable;
use ndarray::{Array1, Array2, array};

// ============================================================================
// 演算子の組み合わせテスト（値/参照の混合）
// ============================================================================

#[test]
fn test_ref_ref() {
    // &x + &y (両方参照)
    let x = Variable::new(array![1.0, 2.0]);
    let y = Variable::new(array![3.0, 4.0]);
    let z = &x + &y;
    assert_eq!(z.value(), array![4.0, 6.0]);
    // x, y はまだ使える
    assert_eq!(x.value(), array![1.0, 2.0]);
    assert_eq!(y.value(), array![3.0, 4.0]);
}

#[test]
fn test_val_ref() {
    // x + &y (左が値、右が参照)
    let x = Variable::new(array![1.0, 2.0]);
    let y = Variable::new(array![3.0, 4.0]);
    let z = x.clone() + &y;
    assert_eq!(z.value(), array![4.0, 6.0]);
    // y はまだ使える
    assert_eq!(y.value(), array![3.0, 4.0]);
}

#[test]
fn test_ref_val() {
    // &x + y (左が参照、右が値)
    let x = Variable::new(array![1.0, 2.0]);
    let y = Variable::new(array![3.0, 4.0]);
    let z = &x + y.clone();
    assert_eq!(z.value(), array![4.0, 6.0]);
    // x はまだ使える
    assert_eq!(x.value(), array![1.0, 2.0]);
}

#[test]
fn test_val_val() {
    // x + y (両方値、所有権を消費)
    let x = Variable::new(array![1.0, 2.0]);
    let y = Variable::new(array![3.0, 4.0]);
    let z = x.clone() + y.clone();
    assert_eq!(z.value(), array![4.0, 6.0]);
}

#[test]
fn test_mixed_chain() {
    // 複合演算で異なる組み合わせを使用
    let x: Variable<Array1<f64>> = Variable::new(array![1.0, 2.0]);
    let y: Variable<Array1<f64>> = Variable::new(array![3.0, 4.0]);

    // (x + y) * x - y
    // 参照と値を自由に混ぜられる
    let sum = &x + &y; // ref + ref
    let prod = sum * &x; // val + ref
    let result = &prod - y.clone(); // ref - val

    // (1+3)*1 - 3 = 1, (2+4)*2 - 4 = 8
    assert_eq!(result.value(), array![1.0, 8.0]);

    // 勾配計算も正常に動作
    result.backward_with(Variable::new(array![1.0, 1.0]));
    assert!(x.grad().is_some());
}

// ============================================================================
// 順伝播のテスト（1次元配列）
// ============================================================================

#[test]
fn test_array1_add() {
    let x = Variable::new(array![1.0, 2.0, 3.0]);
    let y = Variable::new(array![4.0, 5.0, 6.0]);
    let z = &x + &y;
    assert_eq!(z.value(), array![5.0, 7.0, 9.0]);
}

#[test]
fn test_array1_mul() {
    let x = Variable::new(array![1.0, 2.0, 3.0]);
    let y = Variable::new(array![4.0, 5.0, 6.0]);
    let z = &x * &y;
    assert_eq!(z.value(), array![4.0, 10.0, 18.0]);
}

#[test]
fn test_array1_neg() {
    let x = Variable::new(array![1.0, -2.0, 3.0]);
    let z = -&x;
    assert_eq!(z.value(), array![-1.0, 2.0, -3.0]);
}

#[test]
fn test_array1_sub() {
    let x = Variable::new(array![5.0, 6.0, 7.0]);
    let y = Variable::new(array![1.0, 2.0, 3.0]);
    let z = &x - &y;
    assert_eq!(z.value(), array![4.0, 4.0, 4.0]);
}

// ============================================================================
// 順伝播のテスト（2次元配列）
// ============================================================================

#[test]
fn test_array2_add() {
    let x = Variable::new(array![[1.0, 2.0], [3.0, 4.0]]);
    let y = Variable::new(array![[5.0, 6.0], [7.0, 8.0]]);
    let z = &x + &y;
    assert_eq!(z.value(), array![[6.0, 8.0], [10.0, 12.0]]);
}

#[test]
fn test_array2_mul() {
    let x = Variable::new(array![[1.0, 2.0], [3.0, 4.0]]);
    let y = Variable::new(array![[2.0, 3.0], [4.0, 5.0]]);
    let z = &x * &y;
    assert_eq!(z.value(), array![[2.0, 6.0], [12.0, 20.0]]);
}

// ============================================================================
// 逆伝播のテスト（1次元配列）
// ============================================================================

#[test]
fn test_array1_add_backward() {
    // z = x + y
    // ∂z/∂x = 1, ∂z/∂y = 1 (element-wise)
    let x: Variable<Array1<f64>> = Variable::new(array![1.0, 2.0, 3.0]);
    let y: Variable<Array1<f64>> = Variable::new(array![4.0, 5.0, 6.0]);
    let z = &x + &y;

    // 初期勾配を全て1.0に設定
    let grad = Variable::new(array![1.0, 1.0, 1.0]);
    z.backward_with(grad);

    assert_eq!(x.grad().unwrap().value(), array![1.0, 1.0, 1.0]);
    assert_eq!(y.grad().unwrap().value(), array![1.0, 1.0, 1.0]);
}

#[test]
fn test_array1_mul_backward() {
    // z = x * y (element-wise)
    // ∂z/∂x = y, ∂z/∂y = x
    let x: Variable<Array1<f64>> = Variable::new(array![1.0, 2.0, 3.0]);
    let y: Variable<Array1<f64>> = Variable::new(array![4.0, 5.0, 6.0]);
    let z = &x * &y;

    let grad = Variable::new(array![1.0, 1.0, 1.0]);
    z.backward_with(grad);

    // ∂z/∂x = y
    assert_eq!(x.grad().unwrap().value(), array![4.0, 5.0, 6.0]);
    // ∂z/∂y = x
    assert_eq!(y.grad().unwrap().value(), array![1.0, 2.0, 3.0]);
}

#[test]
fn test_array1_neg_backward() {
    // z = -x
    // ∂z/∂x = -1
    let x: Variable<Array1<f64>> = Variable::new(array![1.0, 2.0, 3.0]);
    let z = -&x;

    let grad = Variable::new(array![1.0, 1.0, 1.0]);
    z.backward_with(grad);

    assert_eq!(x.grad().unwrap().value(), array![-1.0, -1.0, -1.0]);
}

#[test]
fn test_array1_sub_backward() {
    // z = x - y = x + (-y)
    // ∂z/∂x = 1, ∂z/∂y = -1
    let x: Variable<Array1<f64>> = Variable::new(array![5.0, 6.0, 7.0]);
    let y: Variable<Array1<f64>> = Variable::new(array![1.0, 2.0, 3.0]);
    let z = &x - &y;

    let grad = Variable::new(array![1.0, 1.0, 1.0]);
    z.backward_with(grad);

    assert_eq!(x.grad().unwrap().value(), array![1.0, 1.0, 1.0]);
    assert_eq!(y.grad().unwrap().value(), array![-1.0, -1.0, -1.0]);
}

// ============================================================================
// 複合演算のテスト
// ============================================================================

#[test]
fn test_array1_square() {
    // z = x * x (element-wise square)
    // ∂z/∂x = 2x
    let x: Variable<Array1<f64>> = Variable::new(array![1.0, 2.0, 3.0]);
    let z = &x * &x;

    assert_eq!(z.value(), array![1.0, 4.0, 9.0]);

    let grad = Variable::new(array![1.0, 1.0, 1.0]);
    z.backward_with(grad);

    // ∂z/∂x = 2x
    assert_eq!(x.grad().unwrap().value(), array![2.0, 4.0, 6.0]);
}

#[test]
fn test_array1_chain() {
    // z = (x + y) * y
    // ∂z/∂x = y, ∂z/∂y = x + 2y
    let x: Variable<Array1<f64>> = Variable::new(array![1.0, 2.0]);
    let y: Variable<Array1<f64>> = Variable::new(array![3.0, 4.0]);

    let sum = &x + &y; // [4.0, 6.0]
    let z = &sum * &y; // [12.0, 24.0]

    assert_eq!(z.value(), array![12.0, 24.0]);

    let grad = Variable::new(array![1.0, 1.0]);
    z.backward_with(grad);

    // ∂z/∂sum = y = [3.0, 4.0]
    // ∂sum/∂x = 1, ∂sum/∂y = 1
    // ∂z/∂x = ∂z/∂sum * ∂sum/∂x = y = [3.0, 4.0]
    assert_eq!(x.grad().unwrap().value(), array![3.0, 4.0]);

    // ∂z/∂y = ∂z/∂sum * ∂sum/∂y + ∂z/∂y_direct
    //       = y * 1 + sum = y + (x + y) = x + 2y = [7.0, 10.0]
    assert_eq!(y.grad().unwrap().value(), array![7.0, 10.0]);
}

// ============================================================================
// 2次元配列の逆伝播テスト
// ============================================================================

#[test]
fn test_array2_add_backward() {
    let x: Variable<Array2<f64>> = Variable::new(array![[1.0, 2.0], [3.0, 4.0]]);
    let y: Variable<Array2<f64>> = Variable::new(array![[5.0, 6.0], [7.0, 8.0]]);
    let z = &x + &y;

    let grad = Variable::new(array![[1.0, 1.0], [1.0, 1.0]]);
    z.backward_with(grad);

    assert_eq!(x.grad().unwrap().value(), array![[1.0, 1.0], [1.0, 1.0]]);
    assert_eq!(y.grad().unwrap().value(), array![[1.0, 1.0], [1.0, 1.0]]);
}

#[test]
fn test_array2_mul_backward() {
    let x: Variable<Array2<f64>> = Variable::new(array![[1.0, 2.0], [3.0, 4.0]]);
    let y: Variable<Array2<f64>> = Variable::new(array![[2.0, 3.0], [4.0, 5.0]]);
    let z = &x * &y;

    let grad = Variable::new(array![[1.0, 1.0], [1.0, 1.0]]);
    z.backward_with(grad);

    // ∂z/∂x = y
    assert_eq!(x.grad().unwrap().value(), array![[2.0, 3.0], [4.0, 5.0]]);
    // ∂z/∂y = x
    assert_eq!(y.grad().unwrap().value(), array![[1.0, 2.0], [3.0, 4.0]]);
}

// ============================================================================
// 勾配累積のテスト
// ============================================================================

#[test]
fn test_array1_gradient_accumulation() {
    let x: Variable<Array1<f64>> = Variable::new(array![1.0, 2.0, 3.0]);
    let y: Variable<Array1<f64>> = Variable::new(array![4.0, 5.0, 6.0]);

    let z1 = &x + &y;
    z1.backward_with(Variable::new(array![1.0, 1.0, 1.0]));

    assert_eq!(x.grad().unwrap().value(), array![1.0, 1.0, 1.0]);

    // 再度計算（勾配が累積される）
    let z2 = &x + &y;
    z2.backward_with(Variable::new(array![1.0, 1.0, 1.0]));

    assert_eq!(x.grad().unwrap().value(), array![2.0, 2.0, 2.0]);
}

#[test]
fn test_array1_zero_grad() {
    let x: Variable<Array1<f64>> = Variable::new(array![1.0, 2.0, 3.0]);
    let y: Variable<Array1<f64>> = Variable::new(array![4.0, 5.0, 6.0]);

    let z = &x + &y;
    z.backward_with(Variable::new(array![1.0, 1.0, 1.0]));

    assert!(x.grad().is_some());

    x.zero_grad();
    assert!(x.grad().is_none());
}
