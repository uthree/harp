// Dynamic Shape Matrix Multiplication
// Usage: eclat-transpile dynamic_matmul.ecl -o - -b c -D M=64 -D K=128 -D N=32
//
// Computes C = A @ B where:
//   A: [M, K]
//   B: [K, N]
//   C: [M, N]
//
// Implementation:
//   1. Expand A from [M, K] to [M, K, N]
//   2. Expand B from [K, N] to [M, K, N]
//   3. Element-wise multiply
//   4. Sum along K axis

program {
    // Matrix multiplication with dynamic dimensions
    graph<M, K, N> matmul(a: f32[M, K], b: f32[K, N]) -> f32[M, N] {
        // Expand A: [M, K] -> [M, K, 1] -> [M, K, N]
        let a_expanded = expand(unsqueeze(a, axis=2), [M, K, N]);
        
        // Expand B: [K, N] -> [1, K, N] -> [M, K, N]
        let b_expanded = expand(unsqueeze(b, axis=0), [M, K, N]);
        
        // Element-wise multiply and sum along K axis
        let product = a_expanded * b_expanded;
        let result = sum(product, axis=1);
        
        return squeeze(result, axis=1);
    }
}
