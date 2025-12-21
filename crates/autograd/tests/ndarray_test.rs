use autograd::{Expand, Max, Prod, Sum, Variable};
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

// ============================================================================
// Reduce/Expand トレイトの直接テスト（順伝播のみ）
// ============================================================================

#[test]
fn test_sum_trait_array2_axis0() {
    // 2x3 の配列を axis=0 で sum
    // [[1, 2, 3], [4, 5, 6]] -> [[5, 7, 9]]
    let arr = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let (result, size) = Sum::sum(&arr, 0);

    assert_eq!(size, 2);
    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(result, array![[5.0, 7.0, 9.0]]);
}

#[test]
fn test_sum_trait_array2_axis1() {
    // 2x3 の配列を axis=1 で sum
    // [[1, 2, 3], [4, 5, 6]] -> [[6], [15]]
    let arr = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let (result, size) = Sum::sum(&arr, 1);

    assert_eq!(size, 3);
    assert_eq!(result.shape(), &[2, 1]);
    assert_eq!(result, array![[6.0], [15.0]]);
}

#[test]
fn test_prod_trait_array2_axis0() {
    // 2x3 の配列を axis=0 で prod
    // [[1, 2, 3], [4, 5, 6]] -> [[4, 10, 18]]
    let arr = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let (result, size) = Prod::prod(&arr, 0);

    assert_eq!(size, 2);
    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(result, array![[4.0, 10.0, 18.0]]);
}

#[test]
fn test_max_trait_array2_axis0() {
    // 2x3 の配列を axis=0 で max
    // [[1, 5, 3], [4, 2, 6]] -> [[4, 5, 6]]
    let arr = array![[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]];
    let (result, size) = Max::max(&arr, 0);

    assert_eq!(size, 2);
    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(result, array![[4.0, 5.0, 6.0]]);
}

#[test]
fn test_expand_trait_array2() {
    // 1x3 の配列を axis=0 で 2 倍に拡張
    // [[1, 2, 3]] -> [[1, 2, 3], [1, 2, 3]]
    let arr = array![[1.0, 2.0, 3.0]];
    let result = Expand::expand(&arr, 0, 2);

    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result, array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
}

#[test]
fn test_sum_expand_roundtrip() {
    // sum して expand すると値は保持されるが、形状は元に戻る
    let arr = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let (summed, size) = Sum::sum(&arr, 0);
    let expanded = Expand::expand(&summed, 0, size);

    assert_eq!(expanded.shape(), arr.shape());
    // 合計値が各行にコピーされる
    assert_eq!(expanded, array![[5.0, 7.0, 9.0], [5.0, 7.0, 9.0]]);
}

// ============================================================================
// Variable<Array2> での reduce 演算テスト（順伝播）
// ============================================================================

#[test]
fn test_variable_sum_forward() {
    // Variable<Array2> の sum 順伝播
    let x: Variable<Array2<f64>> = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let y = x.sum(0);

    assert_eq!(y.value().shape(), &[1, 3]);
    assert_eq!(y.value(), array![[5.0, 7.0, 9.0]]);
}

#[test]
fn test_variable_prod_forward() {
    // Variable<Array2> の prod 順伝播
    let x: Variable<Array2<f64>> = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let y = x.prod(0);

    assert_eq!(y.value().shape(), &[1, 3]);
    assert_eq!(y.value(), array![[4.0, 10.0, 18.0]]);
}

#[test]
fn test_variable_max_forward() {
    // Variable<Array2> の max 順伝播
    let x: Variable<Array2<f64>> = Variable::new(array![[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]);
    let y = x.max(0);

    assert_eq!(y.value().shape(), &[1, 3]);
    assert_eq!(y.value(), array![[4.0, 5.0, 6.0]]);
}

#[test]
fn test_variable_expand_forward() {
    // Variable<Array2> の expand 順伝播
    let x: Variable<Array2<f64>> = Variable::new(array![[1.0, 2.0, 3.0]]);
    let y = x.expand(0, 2);

    assert_eq!(y.value().shape(), &[2, 3]);
    assert_eq!(y.value(), array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
}

// ============================================================================
// Variable<Array2> での reduce 演算テスト（逆伝播）
// ============================================================================

#[test]
fn test_variable_sum_backward() {
    // y = sum(x, axis=0)
    // x: 2x3, y: 1x3
    // ∂L/∂x = expand(∂L/∂y, axis=0)
    let x: Variable<Array2<f64>> = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let y = x.sum(0);

    // 勾配 [1, 1, 1] を渡す
    let grad = Variable::new(array![[1.0, 1.0, 1.0]]);
    y.backward_with(grad);

    // sum の勾配: 各要素に同じ勾配が伝播
    let expected_grad = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
    assert_eq!(x.grad().unwrap().value(), expected_grad);
}

#[test]
fn test_variable_expand_backward() {
    // y = expand(x, axis=0, size=2)
    // x: 1x3, y: 2x3
    // ∂L/∂x = sum(∂L/∂y, axis=0)
    let x: Variable<Array2<f64>> = Variable::new(array![[1.0, 2.0, 3.0]]);
    let y = x.expand(0, 2);

    // 勾配を渡す
    let grad = Variable::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    y.backward_with(grad);

    // expand の勾配: 軸方向に合計
    let expected_grad = array![[5.0, 7.0, 9.0]];
    assert_eq!(x.grad().unwrap().value(), expected_grad);
}

#[test]
fn test_variable_prod_backward() {
    // y = prod(x, axis=0)
    // x: [[2, 3], [4, 5]], y: [[8, 15]]
    // ∂L/∂x_ij = ∂L/∂y_j * y_j / x_ij
    let x: Variable<Array2<f64>> = Variable::new(array![[2.0, 3.0], [4.0, 5.0]]);
    let y = x.prod(0);

    assert_eq!(y.value(), array![[8.0, 15.0]]);

    // 勾配 [1, 1] を渡す
    let grad = Variable::new(array![[1.0, 1.0]]);
    y.backward_with(grad);

    // ∂L/∂x_ij = y_j / x_ij
    // [[8/2, 15/3], [8/4, 15/5]] = [[4, 5], [2, 3]]
    let expected_grad = array![[4.0, 5.0], [2.0, 3.0]];
    assert_eq!(x.grad().unwrap().value(), expected_grad);
}

#[test]
fn test_variable_max_backward() {
    // y = max(x, axis=0)
    // x: [[1, 5], [4, 2]], y: [[4, 5]]
    // ∂L/∂x_ij = ∂L/∂y_j if x_ij == max else 0
    let x: Variable<Array2<f64>> = Variable::new(array![[1.0, 5.0], [4.0, 2.0]]);
    let y = x.max(0);

    assert_eq!(y.value(), array![[4.0, 5.0]]);

    // 勾配 [1, 1] を渡す
    let grad = Variable::new(array![[1.0, 1.0]]);
    y.backward_with(grad);

    // 最大値の位置: (1, 0) と (0, 1)
    // [[0, 1], [1, 0]]
    let expected_grad = array![[0.0, 1.0], [1.0, 0.0]];
    assert_eq!(x.grad().unwrap().value(), expected_grad);
}

// ============================================================================
// 複合演算のテスト (reduce + arithmetic)
// ============================================================================

#[test]
fn test_sum_then_mul() {
    // z = sum(x, axis=0) * y
    // x: 2x2, y: 1x2, z: 1x2
    let x: Variable<Array2<f64>> = Variable::new(array![[1.0, 2.0], [3.0, 4.0]]);
    let y: Variable<Array2<f64>> = Variable::new(array![[2.0, 3.0]]);

    let sum_x = x.sum(0); // [[4, 6]]
    let z = &sum_x * &y; // [[8, 18]]

    assert_eq!(z.value(), array![[8.0, 18.0]]);

    let grad = Variable::new(array![[1.0, 1.0]]);
    z.backward_with(grad);

    // ∂z/∂sum_x = y = [2, 3]
    // ∂sum_x/∂x = expand([2, 3]) = [[2, 3], [2, 3]]
    assert_eq!(x.grad().unwrap().value(), array![[2.0, 3.0], [2.0, 3.0]]);

    // ∂z/∂y = sum_x = [4, 6]
    assert_eq!(y.grad().unwrap().value(), array![[4.0, 6.0]]);
}

#[test]
fn test_mul_then_sum() {
    // z = sum(x * y, axis=0)
    // x: 2x2, y: 2x2, z: 1x2
    let x: Variable<Array2<f64>> = Variable::new(array![[1.0, 2.0], [3.0, 4.0]]);
    let y: Variable<Array2<f64>> = Variable::new(array![[2.0, 3.0], [4.0, 5.0]]);

    let prod = &x * &y; // [[2, 6], [12, 20]]
    let z = prod.sum(0); // [[14, 26]]

    assert_eq!(z.value(), array![[14.0, 26.0]]);

    let grad = Variable::new(array![[1.0, 1.0]]);
    z.backward_with(grad);

    // ∂z/∂prod = expand([[1, 1]]) = [[1, 1], [1, 1]]
    // ∂prod/∂x = y
    // ∂z/∂x = y * [[1, 1], [1, 1]] = y
    assert_eq!(x.grad().unwrap().value(), array![[2.0, 3.0], [4.0, 5.0]]);
    assert_eq!(y.grad().unwrap().value(), array![[1.0, 2.0], [3.0, 4.0]]);
}
