// Softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
program {
    graph softmax(x: f32[32, 64]) -> f32[32, 64] {
        let x_max = max(x, axis=1);
        let x_shifted = x - expand(x_max, [32, 64]);
        let exp_x = exp(x_shifted);
        let sum_exp = sum(exp_x, axis=1);
        return exp_x / expand(sum_exp, [32, 64]);
    }
}
