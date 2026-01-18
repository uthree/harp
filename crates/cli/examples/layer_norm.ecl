// Layer Normalization
program {
    graph layer_norm(x: f32[32, 512], gamma: f32[512], beta: f32[512]) -> f32[32, 512] {
        let mean = sum(x, axis=1) / 512.0;
        let x_centered = x - expand(mean, [32, 512]);
        let var = sum(x_centered * x_centered, axis=1) / 512.0;
        let std = sqrt(var + 1e-5);
        let x_norm = x_centered / expand(std, [32, 512]);
        return x_norm * gamma + beta;
    }
}
