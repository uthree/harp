use autograd::{Add, Mul, Neg, Recip, Variable};

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

    // grad_fn を設定
    z.set_grad_fn(Box::new(Add::new(x.clone(), y.clone())));

    // backward
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

    // grad_fn を設定
    z.set_grad_fn(Box::new(Mul::new(x.clone(), y.clone())));

    // backward
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

    // grad_fn を設定
    z.set_grad_fn(Box::new(Neg::new(x.clone())));

    // backward
    z.backward();

    assert_eq!(x.grad().unwrap().value(), -1.0);
}

#[test]
fn test_recip_backward() {
    // z = 1/x
    // ∂z/∂x = -1/x²
    let x = Variable::new(2.0_f64);
    let z = Variable::new(1.0 / x.value());

    // grad_fn を設定
    z.set_grad_fn(Box::new(Recip::new(x.clone())));

    // backward
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
    sum.set_grad_fn(Box::new(Add::new(x.clone(), one.clone())));

    // (x + 1) * 2
    let result = &sum * &two;
    result.set_grad_fn(Box::new(Mul::new(sum.clone(), two.clone())));

    // backward
    result.backward();

    // ∂result/∂sum = 2, ∂sum/∂x = 1
    // => ∂result/∂x = 2 * 1 = 2
    assert_eq!(x.grad().unwrap().value(), 2.0);
}

#[test]
fn test_multiple_paths() {
    // f(x) = x * x = x²
    // df/dx = 2x
    let x = Variable::new(3.0_f64);
    let z = &x * &x;

    // grad_fn を設定（x を両方の入力として使用）
    z.set_grad_fn(Box::new(Mul::new(x.clone(), x.clone())));

    // backward
    z.backward();

    // ∂z/∂x = 2x = 6
    // (勾配は累積されるので、x に対して 3 + 3 = 6)
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

    // z1 = x + y
    let z1 = &x + &y;
    z1.set_grad_fn(Box::new(Add::new(x.clone(), y.clone())));
    z1.backward();

    assert_eq!(x.grad().unwrap().value(), 1.0);

    // 再度 backward (勾配が累積される)
    let z2 = &x + &y;
    z2.set_grad_fn(Box::new(Add::new(x.clone(), y.clone())));
    z2.backward();

    // 1.0 + 1.0 = 2.0
    assert_eq!(x.grad().unwrap().value(), 2.0);
}

#[test]
fn test_zero_grad() {
    let x = Variable::new(2.0_f64);
    let y = Variable::new(3.0_f64);

    let z = &x + &y;
    z.set_grad_fn(Box::new(Add::new(x.clone(), y.clone())));
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
    z.set_grad_fn(Box::new(Add::new(x.clone(), y.clone())));

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
    sum.set_grad_fn(Box::new(Add::new(x.clone(), one.clone())));

    // 2 * (x + 1) = 8
    let result = &two * &sum;
    result.set_grad_fn(Box::new(Mul::new(two.clone(), sum.clone())));

    assert_eq!(result.value(), 8.0);

    result.backward();

    // ∂result/∂sum = 2, ∂sum/∂x = 1
    // => ∂result/∂x = 2
    assert_eq!(x.grad().unwrap().value(), 2.0);
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
    sum.set_grad_fn(Box::new(Add::new(x.clone(), one.clone())));

    // 1 / (x + 1) = 0.5
    let result = Variable::new(1.0 / sum.value());
    result.set_grad_fn(Box::new(Recip::new(sum.clone())));

    assert_eq!(result.value(), 0.5);

    result.backward();

    // f'(1) = -0.25
    assert_eq!(x.grad().unwrap().value(), -0.25);
}
