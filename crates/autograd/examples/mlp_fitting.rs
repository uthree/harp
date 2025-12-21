//! 2層パーセプトロン（MLP）による多次元関数フィッティングのデモ
//!
//! 2次元入力 (x, y) から複雑な関数 z = f(x, y) を学習します。
//!
//! ターゲット関数:
//!   z = sin(πx) * cos(πy) + 0.3x² - 0.2y² + 0.1xy
//!
//! ネットワーク構造:
//!   入力層 (2) → 隠れ層 (32) → 出力層 (1)
//!
//! 実行:
//! ```
//! cargo run --example mlp_fitting -p autograd --features ndarray
//! ```

use autograd::Variable;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use rand::Rng;

// ============================================================================
// ターゲット関数
// ============================================================================

/// 学習対象の関数: z = sin(πx) * cos(πy) + 0.3x² - 0.2y² + 0.1xy
fn target_function(x: f64, y: f64) -> f64 {
    let pi = std::f64::consts::PI;
    (pi * x).sin() * (pi * y).cos() + 0.3 * x * x - 0.2 * y * y + 0.1 * x * y
}

// ============================================================================
// ReLU 活性化関数
// ============================================================================

/// ReLU: max(x, 0)
fn relu(x: &Variable<Array2<f64>>) -> Variable<Array2<f64>> {
    x.maximum(&x.zeros_like())
}

// ============================================================================
// 2層 MLP
// ============================================================================

/// 2層パーセプトロン（バイアスなし、ReLU活性化）
struct Mlp {
    // 第1層: [入力, 隠れ]
    w1: Variable<Array2<f64>>,
    // 第2層: [隠れ, 出力]
    w2: Variable<Array2<f64>>,
}

impl Mlp {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        // He 初期化 (ReLU 向け)
        let scale1 = (2.0 / input_dim as f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();

        let w1_data: Vec<f64> = (0..input_dim * hidden_dim)
            .map(|_| rng.gen_range(-scale1..scale1))
            .collect();
        let w2_data: Vec<f64> = (0..hidden_dim * output_dim)
            .map(|_| rng.gen_range(-scale2..scale2))
            .collect();

        let w1 = Array2::from_shape_vec((input_dim, hidden_dim), w1_data).unwrap();
        let w2 = Array2::from_shape_vec((hidden_dim, output_dim), w2_data).unwrap();

        Self {
            w1: Variable::new(w1),
            w2: Variable::new(w2),
        }
    }

    /// 順伝播: x [batch, input] → y [batch, output]
    fn forward(&self, x: &Variable<Array2<f64>>) -> Variable<Array2<f64>> {
        // 第1層: z1 = x @ W1
        let z1 = x.matmul(&self.w1);

        // ReLU 活性化: h = max(z1, 0)
        let h = relu(&z1);

        // 第2層: y = h @ W2
        h.matmul(&self.w2)
    }

    /// 勾配をゼロに初期化
    fn zero_grad(&self) {
        self.w1.zero_grad();
        self.w2.zero_grad();
    }

    /// 勾配降下法でパラメータを更新
    fn step(&mut self, lr: f64) {
        if let Some(grad) = self.w1.grad() {
            let new_w1 = self.w1.value() - &(grad.value() * lr);
            self.w1 = Variable::new(new_w1);
        }
        if let Some(grad) = self.w2.grad() {
            let new_w2 = self.w2.value() - &(grad.value() * lr);
            self.w2 = Variable::new(new_w2);
        }
    }
}

// ============================================================================
// メイン
// ============================================================================

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     2層パーセプトロンによる多次元関数フィッティング デモ       ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // ============================================================
    // 1. データ生成
    // ============================================================
    println!("📊 データ生成中...");

    let mut rng = rand::thread_rng();
    let n_samples = 500;
    let noise_scale = 0.05;

    // [-1, 1] × [-1, 1] の範囲でサンプリング
    let mut x_data: Vec<f64> = Vec::with_capacity(n_samples * 2);
    let mut y_data: Vec<f64> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x = rng.gen_range(-1.0..1.0);
        let y = rng.gen_range(-1.0..1.0);
        let z = target_function(x, y) + rng.gen_range(-noise_scale..noise_scale);

        x_data.push(x);
        x_data.push(y);
        y_data.push(z);
    }

    let x_train = Array2::from_shape_vec((n_samples, 2), x_data).unwrap();
    let y_train = Array2::from_shape_vec((n_samples, 1), y_data).unwrap();

    println!("   サンプル数: {}", n_samples);
    println!("   入力次元: 2 (x, y)");
    println!("   出力次元: 1 (z)");
    println!("   ターゲット関数: z = sin(πx)cos(πy) + 0.3x² - 0.2y² + 0.1xy");
    println!();

    // ============================================================
    // 2. MLP 初期化
    // ============================================================
    println!("🔧 ネットワーク初期化...");

    let input_dim = 2;
    let hidden_dim = 32;
    let output_dim = 1;

    let mut mlp = Mlp::new(input_dim, hidden_dim, output_dim);

    println!(
        "   構造: {} → {} → {} (バイアスなし)",
        input_dim, hidden_dim, output_dim
    );
    println!(
        "   パラメータ数: {} (W1: {}×{} + W2: {}×{})",
        input_dim * hidden_dim + hidden_dim * output_dim,
        input_dim,
        hidden_dim,
        hidden_dim,
        output_dim
    );
    println!();

    // ============================================================
    // 3. 学習（自動微分による勾配降下法）
    // ============================================================
    let epochs = 1000;
    let lr = 0.01;
    let batch_size = 50;
    let n_batches = n_samples / batch_size;

    println!(
        "🚀 学習開始 (epochs={}, lr={}, batch_size={})",
        epochs, lr, batch_size
    );
    println!("   活性化関数: ReLU (max(x, 0))");
    println!("   勾配計算: 自動微分 (バックプロパゲーション)");
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
        let mut epoch_loss = 0.0;

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            let x_batch = x_train.slice(ndarray::s![start..end, ..]).to_owned();
            let y_batch = y_train.slice(ndarray::s![start..end, ..]).to_owned();

            // 勾配をゼロに初期化
            mlp.zero_grad();

            // 順伝播
            let x_var = Variable::new(x_batch);
            let y_var = Variable::new(y_batch);
            let pred = mlp.forward(&x_var);

            // MSE損失: L = mean((pred - target)²)
            let diff = &pred - &y_var;
            let squared = &diff * &diff;
            let loss = squared.sum(0).sum(1); // スカラーに縮約

            // 損失値を記録
            let loss_val = loss.value()[[0, 0]] / (batch_size as f64);
            epoch_loss += loss_val;

            // 逆伝播
            // 勾配スケール: 1/batch_size (MSEの平均化)
            let grad_scale = 1.0 / (batch_size as f64);
            let grad = Variable::new(Array2::from_elem((1, 1), grad_scale));
            loss.backward_with(grad);

            // パラメータ更新
            mlp.step(lr);
        }

        epoch_loss /= n_batches as f64;
        loss_history.push(epoch_loss);

        if epoch % 10 == 0 || epoch == epochs - 1 {
            pb.set_message(format!("{:.6}", epoch_loss));
        }
        pb.inc(1);
    }

    pb.finish_with_message(format!("{:.6}", loss_history.last().unwrap()));
    println!();

    // ============================================================
    // 4. 結果表示
    // ============================================================
    println!("✅ 学習完了!");
    println!();
    println!("📈 最終結果:");
    println!("   最終損失: {:.6}", loss_history.last().unwrap());
    println!();

    // テスト: いくつかの点で予測と真値を比較
    println!("📊 予測 vs 真値 (サンプル):");
    println!(
        "   {:>8} {:>8} │ {:>10} {:>10} │ {:>8}",
        "x", "y", "予測", "真値", "誤差"
    );
    println!("   ─────────────────┼───────────────────────┼─────────");

    let test_points = [
        (0.0, 0.0),
        (0.5, 0.5),
        (-0.5, 0.5),
        (0.3, -0.7),
        (-0.8, -0.3),
    ];

    for (x, y) in test_points {
        let input = Array2::from_shape_vec((1, 2), vec![x, y]).unwrap();
        let x_var = Variable::new(input);
        let pred = mlp.forward(&x_var);
        let pred_val = pred.value()[[0, 0]];
        let true_val = target_function(x, y);
        let error = (pred_val - true_val).abs();

        println!(
            "   {:>8.3} {:>8.3} │ {:>10.4} {:>10.4} │ {:>8.4}",
            x, y, pred_val, true_val, error
        );
    }
    println!();

    // 損失推移グラフ
    println!("📉 損失の推移:");
    use textplots::{Chart, Plot, Shape};

    let loss_points: Vec<(f32, f32)> = loss_history
        .iter()
        .enumerate()
        .step_by(10)
        .map(|(i, &l)| (i as f32, l as f32))
        .collect();

    Chart::new(100, 30, 0.0, epochs as f32)
        .lineplot(&Shape::Lines(&loss_points))
        .nice();

    println!();
    println!("このデモでは ReLU 活性化関数と自動微分を使用しています。");
}
