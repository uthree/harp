//! コスト推定のユーティリティ関数
//!
//! コストは対数スケール（log(CPUサイクル数)）で表現されます。
//! これにより、大きな値を扱う際の数値的安定性が向上し、
//! 乗算が加算に変換されるため計算が簡潔になります。

/// log-sum-exp: log(exp(a) + exp(b)) を数値的に安定な方法で計算
///
/// # 引数
/// * `log_a` - log(a)
/// * `log_b` - log(b)
///
/// # 戻り値
/// log(a + b) = log(exp(log_a) + exp(log_b))
///
/// # 実装
/// max = max(log_a, log_b) として、
/// log(exp(log_a) + exp(log_b)) = max + log(exp(log_a - max) + exp(log_b - max))
/// これにより、オーバーフローを防ぎます。
pub fn log_sum_exp(log_a: f32, log_b: f32) -> f32 {
    // どちらかが -∞ (つまり元の値が0) の場合の処理
    if log_a.is_infinite() && log_a.is_sign_negative() {
        return log_b;
    }
    if log_b.is_infinite() && log_b.is_sign_negative() {
        return log_a;
    }

    let max = log_a.max(log_b);
    max + ((log_a - max).exp() + (log_b - max).exp()).ln()
}

/// 複数の対数値の log-sum-exp
///
/// # 引数
/// * `log_values` - log(x_i) の配列
///
/// # 戻り値
/// log(Σ x_i) = log(Σ exp(log(x_i)))
pub fn log_sum_exp_slice(log_values: &[f32]) -> f32 {
    if log_values.is_empty() {
        return f32::NEG_INFINITY; // log(0)
    }

    // 全て -∞ の場合
    let max = log_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max.is_infinite() && max.is_sign_negative() {
        return f32::NEG_INFINITY;
    }

    max + log_values
        .iter()
        .map(|&x| (x - max).exp())
        .sum::<f32>()
        .ln()
}

/// イテレータから log-sum-exp を計算
pub fn log_sum_exp_iter<I>(log_values: I) -> f32
where
    I: IntoIterator<Item = f32>,
{
    let values: Vec<f32> = log_values.into_iter().collect();
    log_sum_exp_slice(&values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sum_exp_basic() {
        // log(exp(1) + exp(2)) = log(e + e^2) ≈ 2.313
        let result = log_sum_exp(1.0, 2.0);
        let expected = (1.0_f32.exp() + 2.0_f32.exp()).ln();
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_log_sum_exp_with_neg_inf() {
        // log(exp(-∞) + exp(2)) = log(0 + e^2) = 2
        let result = log_sum_exp(f32::NEG_INFINITY, 2.0);
        assert_eq!(result, 2.0);

        let result = log_sum_exp(2.0, f32::NEG_INFINITY);
        assert_eq!(result, 2.0);
    }

    #[test]
    fn test_log_sum_exp_slice_basic() {
        // log(e^1 + e^2 + e^3)
        let result = log_sum_exp_slice(&[1.0, 2.0, 3.0]);
        let expected = (1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp()).ln();
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_log_sum_exp_slice_empty() {
        let result = log_sum_exp_slice(&[]);
        assert!(result.is_infinite() && result.is_sign_negative());
    }

    #[test]
    fn test_log_sum_exp_iter() {
        let result = log_sum_exp_iter(vec![1.0, 2.0, 3.0]);
        let expected = (1.0_f32.exp() + 2.0_f32.exp() + 3.0_f32.exp()).ln();
        assert!((result - expected).abs() < 1e-6);
    }
}
