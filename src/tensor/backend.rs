/// Validate that a backend name is supported.
///
/// # Examples
///
/// ```
/// use harp::tensor::backend::validate_backend;
///
/// validate_backend("c"); // OK
/// ```
#[cfg(feature = "backend-c")]
pub fn validate_backend(name: &str) {
    match name {
        "c" => {}
        _ => panic!("Unknown backend: {}", name),
    }
}

#[cfg(not(feature = "backend-c"))]
pub fn validate_backend(_name: &str) {
    panic!("No backends are enabled. Enable at least one backend feature.");
}

#[cfg(all(test, feature = "backend-c"))]
mod tests {
    use super::*;

    #[test]
    fn test_validate_backend() {
        validate_backend("c"); // Should not panic
    }

    #[test]
    #[should_panic(expected = "Unknown backend")]
    fn test_unknown_backend() {
        validate_backend("unknown");
    }
}
