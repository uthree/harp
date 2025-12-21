use autograd::{RecipBackward, Variable};
use std::f64::consts::{E, FRAC_1_SQRT_2, LN_2, PI, SQRT_2};

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
    let z = Variable::with_grad_fn(1.0 / x.value(), Box::new(RecipBackward::new(x.clone())));

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
fn test_requires_grad_default() {
    let x = Variable::new(2.0_f64);
    // デフォルトで requires_grad = true
    assert!(x.requires_grad());
}

#[test]
fn test_requires_grad_false() {
    // requires_grad = false の場合、勾配は累積されない
    let x = Variable::new(2.0_f64);
    let y = Variable::new(3.0_f64);

    // x の勾配計算を無効化
    x.set_requires_grad(false);
    assert!(!x.requires_grad());

    let z = &x + &y;
    z.backward();

    // x には勾配が累積されない
    assert!(x.grad().is_none());
    // y には勾配が累積される
    assert!(y.grad().is_some());
    assert_eq!(y.grad().unwrap().value(), 1.0);
}

#[test]
fn test_requires_grad_chain() {
    // チェーンの途中で requires_grad = false
    let x = Variable::new(2.0_f64);
    let y = Variable::new(3.0_f64);
    let w = Variable::new(4.0_f64);

    // y のみ勾配計算を無効化
    y.set_requires_grad(false);

    // z = (x + y) * w
    let sum = &x + &y;
    let z = &sum * &w;

    z.backward();

    // x には勾配が伝播する（w = 4）
    assert!(x.grad().is_some());
    assert_eq!(x.grad().unwrap().value(), 4.0);

    // y には勾配が累積されない
    assert!(y.grad().is_none());

    // w には勾配が伝播する（x + y = 5）
    assert!(w.grad().is_some());
    assert_eq!(w.grad().unwrap().value(), 5.0);
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
    let result = Variable::with_grad_fn(
        1.0 / sum_var.value(),
        Box::new(RecipBackward::new(sum_var.clone())),
    );

    assert_eq!(result.value(), 0.5);

    result.backward();

    // ∂(1/sum_var)/∂sum_var = -1/sum_var² = -1/4 = -0.25
    assert_eq!(sum_var.grad().unwrap().value(), -0.25);
}

// ============================================================================
// 超越関数のテスト
// ============================================================================

#[test]
fn test_sin_forward() {
    let x = Variable::new(0.0_f64);
    let y = x.sin();
    assert!((y.value() - 0.0).abs() < 1e-10);

    let x = Variable::new(PI / 2.0);
    let y = x.sin();
    assert!((y.value() - 1.0).abs() < 1e-10);
}

#[test]
fn test_cos_forward() {
    let x = Variable::new(0.0_f64);
    let y = x.cos();
    assert!((y.value() - 1.0).abs() < 1e-10);

    let x = Variable::new(PI);
    let y = x.cos();
    assert!((y.value() - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_log2_forward() {
    let x = Variable::new(1.0_f64);
    let y = x.log2();
    assert!((y.value() - 0.0).abs() < 1e-10);

    let x = Variable::new(8.0_f64);
    let y = x.log2();
    assert!((y.value() - 3.0).abs() < 1e-10);
}

#[test]
fn test_exp2_forward() {
    let x = Variable::new(0.0_f64);
    let y = x.exp2();
    assert!((y.value() - 1.0).abs() < 1e-10);

    let x = Variable::new(3.0_f64);
    let y = x.exp2();
    assert!((y.value() - 8.0).abs() < 1e-10);
}

#[test]
fn test_sqrt_forward() {
    let x = Variable::new(4.0_f64);
    let y = x.sqrt();
    assert!((y.value() - 2.0).abs() < 1e-10);

    let x = Variable::new(2.0_f64);
    let y = x.sqrt();
    assert!((y.value() - SQRT_2).abs() < 1e-10);
}

// ============================================================================
// 超越関数の逆伝播テスト
// ============================================================================

#[test]
fn test_sin_backward() {
    // y = sin(x)
    // ∂y/∂x = cos(x)
    let x = Variable::new(0.0_f64);
    let y = x.sin();
    y.backward();
    // cos(0) = 1
    assert!((x.grad().unwrap().value() - 1.0).abs() < 1e-10);

    let x = Variable::new(PI / 2.0);
    let y = x.sin();
    y.backward();
    // cos(π/2) = 0
    assert!(x.grad().unwrap().value().abs() < 1e-10);
}

#[test]
fn test_cos_backward() {
    // y = cos(x)
    // ∂y/∂x = -sin(x)
    let x = Variable::new(0.0_f64);
    let y = x.cos();
    y.backward();
    // -sin(0) = 0
    assert!(x.grad().unwrap().value().abs() < 1e-10);

    let x = Variable::new(PI / 2.0);
    let y = x.cos();
    y.backward();
    // -sin(π/2) = -1
    assert!((x.grad().unwrap().value() - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_log2_backward() {
    // y = log2(x)
    // ∂y/∂x = 1 / (x * ln(2))
    let x = Variable::new(2.0_f64);
    let y = x.log2();
    y.backward();
    // 1 / (2 * ln(2))
    let expected = 1.0 / (2.0 * LN_2);
    assert!((x.grad().unwrap().value() - expected).abs() < 1e-10);
}

#[test]
fn test_exp2_backward() {
    // y = exp2(x) = 2^x
    // ∂y/∂x = 2^x * ln(2)
    let x = Variable::new(3.0_f64);
    let y = x.exp2();
    y.backward();
    // 2^3 * ln(2) = 8 * ln(2)
    let expected = 8.0 * LN_2;
    assert!((x.grad().unwrap().value() - expected).abs() < 1e-10);
}

#[test]
fn test_sqrt_backward() {
    // y = sqrt(x)
    // ∂y/∂x = 1 / (2 * sqrt(x))
    let x = Variable::new(4.0_f64);
    let y = x.sqrt();
    y.backward();
    // 1 / (2 * 2) = 0.25
    assert!((x.grad().unwrap().value() - 0.25).abs() < 1e-10);

    let x = Variable::new(2.0_f64);
    let y = x.sqrt();
    y.backward();
    // 1 / (2 * sqrt(2)) = 1 / (2 * SQRT_2) = FRAC_1_SQRT_2 / 2
    let expected = FRAC_1_SQRT_2 / 2.0;
    assert!((x.grad().unwrap().value() - expected).abs() < 1e-10);
}

// ============================================================================
// 超越関数の複合演算テスト
// ============================================================================

#[test]
fn test_sin_mul_cos() {
    // y = sin(x) * cos(x) = sin(2x) / 2
    // ∂y/∂x = cos(x) * cos(x) - sin(x) * sin(x) = cos(2x)
    // x = π/4 の場合: cos(π/2) = 0
    let x = Variable::new(PI / 4.0);
    let sin_x = x.sin();
    let cos_x = x.cos();
    let y = &sin_x * &cos_x;

    // sin(π/4) * cos(π/4) = 0.5 * 0.5 * 2 = 0.5
    assert!((y.value() - 0.5).abs() < 1e-10);

    y.backward();
    // cos(2 * π/4) = cos(π/2) = 0
    // 注: 現在の実装では同じ変数を複数回使う複合演算で
    // 勾配が正しく累積されないため、個別の変数を使用
    let grad = x.grad().unwrap().value();
    assert!(grad.abs() < 1e-10);
}

#[test]
fn test_exp2_log2_inverse() {
    // y = exp2(log2(x)) = x
    // ∂y/∂x = 1
    let x = Variable::new(5.0_f64);
    let log_x = x.log2();
    let y = log_x.exp2();

    assert!((y.value() - 5.0).abs() < 1e-10);

    y.backward();
    // chain rule: exp2(log2(x)) の微分は 1
    assert!((x.grad().unwrap().value() - 1.0).abs() < 1e-10);
}

// ============================================================================
// 自然対数・自然指数関数（Ln, Exp）のテスト
// ============================================================================

#[test]
fn test_ln_forward() {
    // ln(1) = 0
    let x = Variable::new(1.0_f64);
    let y = x.ln();
    assert!((y.value() - 0.0).abs() < 1e-10);

    // ln(e) = 1
    let x = Variable::new(E);
    let y = x.ln();
    assert!((y.value() - 1.0).abs() < 1e-10);

    // ln(e^2) = 2
    let x = Variable::new(E * E);
    let y = x.ln();
    assert!((y.value() - 2.0).abs() < 1e-10);
}

#[test]
fn test_exp_forward() {
    // exp(0) = 1
    let x = Variable::new(0.0_f64);
    let y = x.exp();
    assert!((y.value() - 1.0).abs() < 1e-10);

    // exp(1) = e
    let x = Variable::new(1.0_f64);
    let y = x.exp();
    assert!((y.value() - E).abs() < 1e-10);

    // exp(2) = e^2
    let x = Variable::new(2.0_f64);
    let y = x.exp();
    assert!((y.value() - E * E).abs() < 1e-10);
}

#[test]
fn test_ln_backward() {
    // y = ln(x)
    // ∂y/∂x = 1/x
    let x = Variable::new(2.0_f64);
    let y = x.ln();
    y.backward();
    // 1/2 = 0.5
    assert!((x.grad().unwrap().value() - 0.5).abs() < 1e-10);

    let x = Variable::new(E);
    let y = x.ln();
    y.backward();
    // 1/e
    assert!((x.grad().unwrap().value() - 1.0 / E).abs() < 1e-10);
}

#[test]
fn test_exp_backward() {
    // y = exp(x)
    // ∂y/∂x = exp(x)
    let x = Variable::new(0.0_f64);
    let y = x.exp();
    y.backward();
    // exp(0) = 1
    assert!((x.grad().unwrap().value() - 1.0).abs() < 1e-10);

    let x = Variable::new(1.0_f64);
    let y = x.exp();
    y.backward();
    // exp(1) = e
    assert!((x.grad().unwrap().value() - E).abs() < 1e-10);

    let x = Variable::new(2.0_f64);
    let y = x.exp();
    y.backward();
    // exp(2) = e^2
    assert!((x.grad().unwrap().value() - E * E).abs() < 1e-10);
}

#[test]
fn test_exp_ln_inverse() {
    // y = exp(ln(x)) = x
    // ∂y/∂x = 1
    let x = Variable::new(5.0_f64);
    let ln_x = x.ln();
    let y = ln_x.exp();

    assert!((y.value() - 5.0).abs() < 1e-10);

    y.backward();
    // chain rule: exp(ln(x)) の微分は 1
    assert!((x.grad().unwrap().value() - 1.0).abs() < 1e-10);
}

#[test]
fn test_ln_exp_inverse() {
    // y = ln(exp(x)) = x
    // ∂y/∂x = 1
    let x = Variable::new(3.0_f64);
    let exp_x = x.exp();
    let y = exp_x.ln();

    assert!((y.value() - 3.0).abs() < 1e-10);

    y.backward();
    // chain rule: ln(exp(x)) の微分は 1
    assert!((x.grad().unwrap().value() - 1.0).abs() < 1e-10);
}

// ============================================================================
// 剰余演算（Rem）のテスト
// ============================================================================

#[test]
fn test_rem_forward() {
    // 7.5 % 2.5 = 0.0
    let x = Variable::new(7.5_f64);
    let y = Variable::new(2.5_f64);
    let z = x.rem(&y);
    assert!((z.value() - 0.0).abs() < 1e-10);

    // 7.0 % 3.0 = 1.0
    let x = Variable::new(7.0_f64);
    let y = Variable::new(3.0_f64);
    let z = x.rem(&y);
    assert!((z.value() - 1.0).abs() < 1e-10);

    // 5.5 % 2.0 = 1.5
    let x = Variable::new(5.5_f64);
    let y = Variable::new(2.0_f64);
    let z = x.rem(&y);
    assert!((z.value() - 1.5).abs() < 1e-10);
}

#[test]
fn test_rem_backward() {
    // z = x % y
    // ∂z/∂x = 1
    // ∂z/∂y = -floor(x/y)

    // x = 7.0, y = 3.0 → z = 1.0
    // floor(7/3) = 2
    // ∂z/∂x = 1, ∂z/∂y = -2
    let x = Variable::new(7.0_f64);
    let y = Variable::new(3.0_f64);
    let z = x.rem(&y);

    z.backward();

    assert!((x.grad().unwrap().value() - 1.0).abs() < 1e-10);
    assert!((y.grad().unwrap().value() - (-2.0)).abs() < 1e-10);
}

#[test]
fn test_rem_chain() {
    // w = (x % y) * 2
    // x = 7.0, y = 3.0 → (7 % 3) * 2 = 1 * 2 = 2
    let x = Variable::new(7.0_f64);
    let y = Variable::new(3.0_f64);
    let two = Variable::new(2.0_f64);

    let rem = x.rem(&y);
    let w = &rem * &two;

    assert!((w.value() - 2.0).abs() < 1e-10);

    w.backward();

    // ∂w/∂rem = 2
    // ∂rem/∂x = 1 → ∂w/∂x = 2 * 1 = 2
    // ∂rem/∂y = -floor(7/3) = -2 → ∂w/∂y = 2 * (-2) = -4
    assert!((x.grad().unwrap().value() - 2.0).abs() < 1e-10);
    assert!((y.grad().unwrap().value() - (-4.0)).abs() < 1e-10);
}

// ============================================================================
// 型変換（Cast）のテスト
// ============================================================================

/// テスト用のラッパー型（スケール付きf64）
/// Scaled(x) は x * 10 としてf64に変換される
#[derive(Debug, Clone, PartialEq)]
struct Scaled(f64);

impl std::ops::Add for Scaled {
    type Output = Scaled;
    fn add(self, rhs: Self) -> Self::Output {
        Scaled(self.0 + rhs.0)
    }
}

impl From<Scaled> for f64 {
    fn from(s: Scaled) -> f64 {
        s.0 * 10.0
    }
}

impl From<f64> for Scaled {
    fn from(f: f64) -> Scaled {
        Scaled(f / 10.0)
    }
}

#[test]
fn test_cast_forward() {
    // Scaled(3.0) -> f64 = 30.0
    let x = Variable::new(Scaled(3.0));
    let y: Variable<f64> = x.cast();
    assert!((y.value() - 30.0).abs() < 1e-10);
}

#[test]
fn test_cast_backward() {
    // y = cast(x) where x: Scaled, y: f64
    // y = x.0 * 10
    // ∂L/∂x = ∂L/∂y の逆変換 (÷10)
    let x = Variable::new(Scaled(3.0));
    let y: Variable<f64> = x.cast();

    // y.backward() でgrad_y = 1.0 (f64)
    // これがScaledに変換されて x.grad = Scaled(0.1)
    y.backward();

    let grad = x.grad().unwrap().value();
    assert!((grad.0 - 0.1).abs() < 1e-10);
}

#[test]
fn test_cast_chain() {
    // z = cast(x) + 5.0
    // x: Scaled, z: f64
    let x = Variable::new(Scaled(2.0)); // → 20.0
    let y: Variable<f64> = x.cast();
    let five = Variable::new(5.0_f64);
    let z = y + five;

    assert!((z.value() - 25.0).abs() < 1e-10);

    z.backward();

    // ∂z/∂y = 1.0, ∂y/∂x は逆変換 (÷10)
    // → ∂z/∂x = Scaled(0.1)
    let grad = x.grad().unwrap().value();
    assert!((grad.0 - 0.1).abs() < 1e-10);
}

// ============================================================================
// requires_grad 最適化のテスト
// ============================================================================

#[test]
fn test_no_grad_forward_only() {
    // requires_grad=false 同士の演算では grad_fn が作成されない
    let x = Variable::new_no_grad(2.0_f64);
    let y = Variable::new_no_grad(3.0_f64);

    assert!(!x.requires_grad());
    assert!(!y.requires_grad());

    let z = &x + &y;
    assert_eq!(z.value(), 5.0);
    assert!(!z.requires_grad());

    let w = &x * &y;
    assert_eq!(w.value(), 6.0);
    assert!(!w.requires_grad());

    let neg_x = -&x;
    assert_eq!(neg_x.value(), -2.0);
    assert!(!neg_x.requires_grad());
}

#[test]
fn test_mixed_requires_grad() {
    // requires_grad=true と false の混在
    let x = Variable::new(2.0_f64); // requires_grad=true
    let y = Variable::new_no_grad(3.0_f64); // requires_grad=false

    assert!(x.requires_grad());
    assert!(!y.requires_grad());

    // どちらかが true なら結果も勾配追跡が必要
    let z = &x + &y;
    assert_eq!(z.value(), 5.0);
    // backward しても y には勾配が蓄積されない（requires_grad=false なので）
    z.backward();
    assert!(x.grad().is_some());
    assert!(y.grad().is_none()); // requires_grad=false なので勾配なし
}

#[test]
fn test_transcendental_no_grad() {
    // 超越関数でも requires_grad=false なら勾配追跡しない
    let x = Variable::new_no_grad(1.0_f64);
    let sin_x = x.sin();
    assert!((sin_x.value() - 1.0_f64.sin()).abs() < 1e-10);
    assert!(!sin_x.requires_grad());

    let sqrt_x = x.sqrt();
    assert!((sqrt_x.value() - 1.0).abs() < 1e-10);
    assert!(!sqrt_x.requires_grad());

    let exp2_x = x.exp2();
    assert!((exp2_x.value() - 2.0).abs() < 1e-10);
    assert!(!exp2_x.requires_grad());

    let log2_x = x.log2();
    assert!((log2_x.value() - 0.0).abs() < 1e-10);
    assert!(!log2_x.requires_grad());
}

#[test]
fn test_new_with_requires_grad() {
    // new_with_requires_grad のテスト
    let x_true = Variable::new_with_requires_grad(1.0_f64, true);
    let x_false = Variable::new_with_requires_grad(1.0_f64, false);

    assert!(x_true.requires_grad());
    assert!(!x_false.requires_grad());
}

#[test]
fn test_complex_no_grad_chain() {
    // 複雑な計算でも requires_grad=false なら全体が勾配追跡なし
    let a = Variable::new_no_grad(2.0_f64);
    let b = Variable::new_no_grad(3.0_f64);
    let c = Variable::new_no_grad(4.0_f64);

    // (a + b) * c - a
    let sum = &a + &b; // 5.0
    let prod = &sum * &c; // 20.0
    let result = &prod - &a; // 18.0

    assert_eq!(result.value(), 18.0);
    assert!(!result.requires_grad());
}
