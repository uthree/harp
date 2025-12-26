//! Common test utilities for backend integration tests

pub const EPSILON: f32 = 1e-5;

pub fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

pub fn vec_approx_eq(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y))
}
