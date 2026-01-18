// Simple element-wise addition
program {
    graph add(a: f32[16, 256, 256], b: f32[16, 256, 256]) -> f32[16, 256, 256] {
        return a + b;
    }
}
