//! Test that all example .harp files compile successfully

use std::fs;
use std::path::Path;

/// Helper function to compile a .harp file
fn compile_file(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

    harp_dsl::compile(&source).map_err(|e| format!("Compilation failed: {}", e))?;

    Ok(())
}

macro_rules! example_test {
    ($name:ident, $file:expr) => {
        #[test]
        fn $name() {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("examples")
                .join($file);
            if let Err(e) = compile_file(&path) {
                panic!("Failed to compile {}: {}", $file, e);
            }
        }
    };
}

example_test!(test_add, "add.harp");
example_test!(test_matmul, "matmul.harp");
example_test!(test_softmax, "softmax.harp");
example_test!(test_layer_norm, "layer_norm.harp");
example_test!(test_fused_ops, "fused_ops.harp");
example_test!(test_mlp, "mlp.harp");
example_test!(test_reduce_ops, "reduce_ops.harp");
example_test!(test_view_ops, "view_ops.harp");
example_test!(test_activations, "activations.harp");

/// Test that all .harp files in examples directory compile
#[test]
fn test_all_examples_compile() {
    let examples_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");

    let mut failures = Vec::new();

    for entry in fs::read_dir(&examples_dir).expect("Failed to read examples directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();

        if path.extension().map(|e| e == "harp").unwrap_or(false) {
            if let Err(e) = compile_file(&path) {
                failures.push(format!("{}: {}", path.display(), e));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "Failed to compile {} example(s):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
