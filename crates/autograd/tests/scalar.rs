use autograd::{Recip, Variable};

// ============================================================================
// 順伝播のテスト
// ============================================================================

#[test]
fn test_variable_new_and_value() {
    let x = Variable::new(3.0_f64);
    assert_eq!(x.value(), 3.0);
}

#[test]
fn test_add() {
    let x = Variable::new(2.0_f64);
    let y = Variable::new(3.0_f64);
    let z = &x + &y;
    assert_eq!(z.value(), 5.0);
}

#[test]
fn test_mul() {
    let x = Variable::new(2.0_f64);
    let y = Variable::new(3.0_f64);
    let z = &x * &y;
    assert_eq!(z.value(), 6.0);
}

#[test]
fn test_sub() {
    let x = Variable::new(5.0_f64);
    let y = Variable::new(3.0_f64);
    let z = &x - &y;
    assert_eq!(z.value(), 2.0);
}

#[test]
fn test_div() {
    let x = Variable::new(6.0_f64);
    let y = Variable::new(2.0_f64);
    let z = &x / &y;
    assert_eq!(z.value(), 3.0);
}

#[test]
fn test_neg() {
    let x = Variable::new(3.0_f64);
    let z = -&x;
    assert_eq!(z.value(), -3.0);
}

// ============================================================================
// 勾配伝播のテスト
// ============================================================================

#[test]
fn test_add_backward() {
    // z = x + y
    // ∂z/∂x = 1, ∂z/∂y = 1
    let x = Variable::new(2.0_f64);
    let y = Variable::new(3.0_f64);
    let z = &x + &y;

    z.backward();

    assert_eq!(x.grad().unwrap().value(), 1.0);
    assert_eq!(y.grad().unwrap().value(), 1.0);
}

#[test]
fn test_mul_backward() {
    // z = x * y
    // ∂z/∂x = y, ∂z/∂y = x
    let x = Variable::new(2.0_f64);
    let y = Variable::new(3.0_f64);
    let z = &x * &y;

    z.backward();

    assert_eq!(x.grad().unwrap().value(), 3.0); // ∂z/∂x = y = 3
    assert_eq!(y.grad().unwrap().value(), 2.0); // ∂z/∂y = x = 2
}

#[test]
fn test_neg_backward() {
    // z = -x
    // ∂z/∂x = -1
    let x = Variable::new(3.0_f64);
    let z = -&x;

    z.backward();

    assert_eq!(x.grad().unwrap().value(), -1.0);
}

#[test]
fn test_recip_backward() {
    // z = 1/x
    // ∂z/∂x = -1/x²
    let x = Variable::new(2.0_f64);
    let z = Variable::with_grad_fn(1.0 / x.value(), Box::new(Recip::new(x.clone())));

    z.backward();

    // ∂z/∂x = -1/x² = -1/4 = -0.25
    assert_eq!(x.grad().unwrap().value(), -0.25);
}

// ============================================================================
// 複合演算のテスト
// ============================================================================

#[test]
fn test_chain_rule() {
    // f(x) = (x + 1) * 2
    // df/dx = 2
    let x = Variable::new(3.0_f64);
    let one = Variable::new(1.0_f64);
    let two = Variable::new(2.0_f64);

    // x + 1
    let sum = &x + &one;

    // (x + 1) * 2
    // 演算結果同士の演算も Variable なので直接可能
    let sum_var = Variable::new(sum.value());
    let result = &sum_var * &two;

    result.backward();

    // sum_var は新しいリーフ変数なので、x には勾配が伝播しない
    assert_eq!(sum_var.grad().unwrap().value(), 2.0);
}

#[test]
fn test_multiple_paths() {
    // f(x) = x * x = x²
    // df/dx = 2x
    let x = Variable::new(3.0_f64);
    let z = &x * &x;

    z.backward();

    // ∂z/∂x = 2x = 6
    assert_eq!(x.grad().unwrap().value(), 6.0);
}

// ============================================================================
// 勾配累積のテスト
// ============================================================================

#[test]
fn test_gradient_accumulation() {
    // 複数回 backward を呼んだ場合、勾配は累積される
    let x = Variable::new(2.0_f64);
    let y = Variable::new(3.0_f64);

    let z1 = &x + &y;
    z1.backward();

    assert_eq!(x.grad().unwrap().value(), 1.0);

    // 再度 backward (勾配が累積される)
    let z2 = &x + &y;
    z2.backward();

    // 1.0 + 1.0 = 2.0
    assert_eq!(x.grad().unwrap().value(), 2.0);
}

#[test]
fn test_zero_grad() {
    let x = Variable::new(2.0_f64);
    let y = Variable::new(3.0_f64);

    let z = &x + &y;
    z.backward();

    assert!(x.grad().is_some());

    // 勾配をリセット
    x.zero_grad();
    assert!(x.grad().is_none());
}

#[test]
fn test_detach() {
    // detach すると勾配の伝播が止まる
    let x = Variable::new(2.0_f64);
    let y = Variable::new(3.0_f64);

    let z = &x + &y;

    // detach で grad_fn を削除
    z.detach();

    z.backward();

    // z 自身には勾配が設定されるが、x, y には伝播しない
    assert!(x.grad().is_none());
    assert!(y.grad().is_none());
}

// ============================================================================
// 複雑な計算グラフのテスト
// ============================================================================

#[test]
fn test_simple_chain() {
    // f(x) = 2 * (x + 1)
    // f'(x) = 2
    let x = Variable::new(3.0_f64);
    let one = Variable::new(1.0_f64);
    let two = Variable::new(2.0_f64);

    // x + 1 = 4
    let sum = &x + &one;

    let sum_var = Variable::new(sum.value());
    let result = &two * &sum_var;

    assert_eq!(result.value(), 8.0);

    result.backward();

    // sum_var は新しいリーフ変数なので、x には勾配が伝播しない
    assert_eq!(sum_var.grad().unwrap().value(), 2.0);
}

#[test]
fn test_division_chain() {
    // f(x) = 1 / (x + 1)
    // f'(x) = -1 / (x + 1)²
    // f'(1) = -1 / 4 = -0.25
    let x = Variable::new(1.0_f64);
    let one = Variable::new(1.0_f64);

    // x + 1 = 2
    let sum = &x + &one;

    let sum_var = Variable::new(sum.value());
    let result =
        Variable::with_grad_fn(1.0 / sum_var.value(), Box::new(Recip::new(sum_var.clone())));

    assert_eq!(result.value(), 0.5);

    result.backward();

    // ∂(1/sum_var)/∂sum_var = -1/sum_var² = -1/4 = -0.25
    assert_eq!(sum_var.grad().unwrap().value(), -0.25);
}
