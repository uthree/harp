//! 勾配降下法による曲線フィッティングのデモ
//!
//! ノイズ付きの sin カーブを多項式でフィッティングします。
//! プログレスバーとターミナルグラフで結果を表示します。
//!
//! 実行:
//! ```
//! cargo run --example curve_fitting -p autograd
//! ```

use autograd::Variable;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use textplots::{Chart, Plot, Shape};

/// 多項式モデル: y = a*x^3 + b*x^2 + c*x + d
fn polynomial(
    x: f64,
    a: &Variable<f64>,
    b: &Variable<f64>,
    c: &Variable<f64>,
    d: &Variable<f64>,
) -> Variable<f64> {
    let x_var = Variable::new_no_grad(x);
    let x2 = Variable::new_no_grad(x * x);
    let x3 = Variable::new_no_grad(x * x * x);

    // a*x^3 + b*x^2 + c*x + d
    &(&(a * &x3) + &(b * &x2)) + &(&(c * &x_var) + d)
}

/// 平均二乗誤差を計算
fn mse_loss(
    xs: &[f64],
    ys: &[f64],
    a: &Variable<f64>,
    b: &Variable<f64>,
    c: &Variable<f64>,
    d: &Variable<f64>,
) -> Variable<f64> {
    let n = xs.len() as f64;
    let mut total_loss = Variable::new(0.0);

    for (&x, &y_target) in xs.iter().zip(ys.iter()) {
        let y_pred = polynomial(x, a, b, c, d);
        let y_target_var = Variable::new_no_grad(y_target);
        let diff = &y_pred - &y_target_var;
        let squared = &diff * &diff;
        total_loss = &total_loss + &squared;
    }

    let n_var = Variable::new_no_grad(n);
    &total_loss / &n_var
}

/// 勾配をゼロにリセット
fn zero_grad(params: &[&Variable<f64>]) {
    for p in params {
        p.zero_grad();
    }
}

/// 勾配降下ステップ
fn sgd_step(params: &mut [Variable<f64>], lr: f64) {
    for p in params.iter_mut() {
        if let Some(grad) = p.grad() {
            let new_val = p.value() - lr * grad.value();
            *p = Variable::new(new_val);
        }
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║       勾配降下法による曲線フィッティング デモ              ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // ============================================================
    // 1. ノイズ付きデータの生成
    // ============================================================
    println!("📊 データ生成中...");

    let mut rng = rand::thread_rng();
    let n_samples = 50;
    let noise_scale = 0.3;

    // sin カーブにノイズを加える
    let xs: Vec<f64> = (0..n_samples)
        .map(|i| (i as f64 / n_samples as f64) * 4.0 - 2.0) // [-2, 2]
        .collect();

    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| {
            let noise: f64 = rng.gen_range(-noise_scale..noise_scale);
            x.sin() + noise
        })
        .collect();

    println!("   サンプル数: {}", n_samples);
    println!(
        "   入力範囲: [{:.1}, {:.1}]",
        xs.first().unwrap(),
        xs.last().unwrap()
    );
    println!();

    // ============================================================
    // 2. パラメータの初期化
    // ============================================================
    println!("🔧 パラメータ初期化...");

    let mut a = Variable::new(rng.gen_range(-0.5..0.5));
    let mut b = Variable::new(rng.gen_range(-0.5..0.5));
    let mut c = Variable::new(rng.gen_range(-0.5..0.5));
    let mut d = Variable::new(rng.gen_range(-0.5..0.5));

    println!(
        "   初期値: a={:.4}, b={:.4}, c={:.4}, d={:.4}",
        a.value(),
        b.value(),
        c.value(),
        d.value()
    );
    println!();

    // ============================================================
    // 3. 勾配降下法による最適化
    // ============================================================
    let epochs = 10000;
    let lr = 0.01;

    println!("🚀 最適化開始 (epochs={}, lr={})", epochs, lr);
    println!();

    let pb = ProgressBar::new(epochs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} (loss: {msg})")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "),
    );

    let mut loss_history: Vec<f64> = Vec::new();

    for epoch in 0..epochs {
        // 勾配をゼロにリセット
        zero_grad(&[&a, &b, &c, &d]);

        // 損失を計算
        let loss = mse_loss(&xs, &ys, &a, &b, &c, &d);
        let loss_val = loss.value();
        loss_history.push(loss_val);

        // 逆伝播
        loss.backward();

        // パラメータ更新
        let mut params = [a.clone(), b.clone(), c.clone(), d.clone()];
        sgd_step(&mut params, lr);
        a = params[0].clone();
        b = params[1].clone();
        c = params[2].clone();
        d = params[3].clone();

        // プログレスバー更新
        if epoch % 10 == 0 || epoch == epochs - 1 {
            pb.set_message(format!("{:.6}", loss_val));
        }
        pb.inc(1);
    }

    pb.finish_with_message(format!("{:.6}", loss_history.last().unwrap()));
    println!();

    // ============================================================
    // 4. 結果表示
    // ============================================================
    println!("✅ 最適化完了!");
    println!();
    println!("📈 学習済みパラメータ:");
    println!("   a = {:.6} (x³の係数)", a.value());
    println!("   b = {:.6} (x²の係数)", b.value());
    println!("   c = {:.6} (x の係数)", c.value());
    println!("   d = {:.6} (定数項)", d.value());
    println!();
    println!("   最終損失: {:.6}", loss_history.last().unwrap());
    println!();

    // ============================================================
    // 5. ターミナルグラフで可視化
    // ============================================================
    println!("📉 損失の推移:");
    let loss_points: Vec<(f32, f32)> = loss_history
        .iter()
        .enumerate()
        .step_by(10)
        .map(|(i, &l)| (i as f32, l as f32))
        .collect();

    Chart::new(120, 40, 0.0, epochs as f32)
        .lineplot(&Shape::Lines(&loss_points))
        .nice();
    println!();

    // フィッティング結果のグラフ
    println!("📊 フィッティング結果 (○: データ, ─: 予測):");

    // データ点
    let data_points: Vec<(f32, f32)> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| (x as f32, y as f32))
        .collect();

    // 予測曲線
    let pred_points: Vec<(f32, f32)> = (-200..=200)
        .map(|i| {
            let x = i as f64 / 100.0;
            let y = a.value() * x.powi(3) + b.value() * x.powi(2) + c.value() * x + d.value();
            (x as f32, y as f32)
        })
        .collect();

    // 真の sin カーブ
    let true_points: Vec<(f32, f32)> = (-200..=200)
        .map(|i| {
            let x = i as f64 / 100.0;
            (x as f32, x.sin() as f32)
        })
        .collect();

    Chart::new(120, 60, -2.0, 2.0)
        .lineplot(&Shape::Points(&data_points))
        .lineplot(&Shape::Lines(&pred_points))
        .lineplot(&Shape::Lines(&true_points))
        .nice();

    println!();
    println!("凡例: 散布点=観測データ, 実線=多項式予測, 点線=真のsin曲線");
}
