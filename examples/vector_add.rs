use harp::backend::c::{CCompiler, CRenderer};
use harp::backend::codegen::CodeGenerator;
use harp::backend::renderer::Renderer;
use harp::backend::Compiler;
use harp::node::{constant, Node};
use harp::op::{Load, Loop, LoopVariable, Store};
use std::fs::File;
use std::io::Write;
use std::process::Command;
use tempfile::tempdir;

fn main() -> std::io::Result<()> {
    env_logger::init();
    // --- 1. Define the computation graph for vector addition ---
    // for i in 0..10 { c[i] = a[i] + b[i] }
    let i = Node::new(LoopVariable, vec![]);
    let a_i = Node::new(Load("a".to_string(), 10), vec![i.clone()]);
    let b_i = Node::new(Load("b".to_string(), 10), vec![i.clone()]);
    let add_result = a_i + b_i;
    let store_node = Node::new(Store("c".to_string(), 10), vec![i, add_result]);

    let n = 10;
    let count = constant(n);
    let graph = Node::new(
        Loop {
            count: count.clone(),
            body: store_node,
        },
        vec![count],
    );

    // --- 2. Generate C code from the graph ---
    println!("--- Generating C code for vector addition ---");
    let renderer = CRenderer;
    let mut codegen = CodeGenerator::new(&renderer);
    let instructions = codegen.generate(&graph);
    let kernel_code = renderer.render_function(
        "vector_add",
        &[
            ("const float*", "a"),
            ("const float*", "b"),
            ("float*", "c"),
        ],
        &instructions,
        "void",
    );
    println!("{}", kernel_code);

    // --- 3. Create a main.c to test the generated kernel ---
    let main_c_code = format!(
        r#"
#include <stdio.h>
#include <stdlib.h>

// Forward declaration of the generated kernel
{kernel_code}

int main() {{
    int n = {n};
    float a[{n}], b[{n}], c[{n}];

    // Initialize input arrays
    for (int i = 0; i < n; ++i) {{
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }}

    // Call the generated kernel
    vector_add(a, b, c);

    // Verify the result
    printf("--- Verifying results ---\n");
    int errors = 0;
    for (int i = 0; i < n; ++i) {{
        float expected = a[i] + b[i];
        if (c[i] != expected) {{
            printf("Error at index %d: got %f, expected %f\n", i, c[i], expected);
            errors++;
        }}
    }}

    if (errors == 0) {{
        printf("Verification successful!\n");
        printf("c[0] = %f, c[n-1] = %f\n", c[0], c[n-1]);
    }} else {{
        printf("Verification failed with %d errors.\n", errors);
        return 1;
    }}

    return 0;
}}
"#
    );

    // --- 4. Compile and run the C code ---
    println!("\n--- Compiling and running the generated code ---");
    let compiler = CCompiler;
    if !compiler.is_available() {
        eprintln!("Skipping execution: gcc not found.");
        return Ok(());
    }

    let temp_dir = tempdir()?;
    let main_path = temp_dir.path().join("main.c");
    let exe_path = temp_dir.path().join("main");

    let mut main_file = File::create(&main_path)?;
    main_file.write_all(main_c_code.as_bytes())?;

    let output = Command::new("gcc")
        .arg("-o")
        .arg(&exe_path)
        .arg(&main_path)
        .output()?;

    if !output.status.success() {
        eprintln!(
            "gcc compilation failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return Ok(());
    }

    let run_output = Command::new(&exe_path).output()?;
    println!("{}", String::from_utf8_lossy(&run_output.stdout));
    if !run_output.status.success() {
        eprintln!(
            "Execution failed:\n{}",
            String::from_utf8_lossy(&run_output.stderr)
        );
    }

    Ok(())
}
