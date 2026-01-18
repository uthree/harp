// Dynamic shape example
// Use: eclat-transpile dynamic.ecl -o - -b c -D batch=32 -D hidden=64
program {
    graph relu(x: f32[batch, hidden]) -> f32[batch, hidden] {
        return where(x > 0.0, x, 0.0);
    }
}
