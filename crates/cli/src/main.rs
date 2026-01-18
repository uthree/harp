//! eclat-transpile: DSL to backend code transpiler CLI
//!
//! Usage:
//!   eclat-transpile [OPTIONS] <INPUT> -o <OUTPUT>
//!
//! Examples:
//!   eclat-transpile softmax.ecl -o softmax.c -b c
//!   eclat-transpile model.ecl -o model.cu -b cuda
//!   eclat-transpile model.ecl -o model.c -D batch=32

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};

use clap::{Parser, ValueEnum};

use eclat::ast::AstNode;
use eclat::backend::renderer::{GenericRenderer, Renderer};
use eclat::lowerer::Lowerer;
use eclat::opt::ast::{AstOptimizer, RuleBaseOptimizer};
use eclat::opt::ast::rules::all_algebraic_rules;
use eclat_dsl::{GraphBuilder, parse_program};

/// Eclat DSL Transpiler
///
/// Transpiles Eclat DSL to various backend code (C, CUDA, Metal, etc.)
#[derive(Parser, Debug)]
#[command(name = "eclat-transpile")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input file (use "-" for stdin)
    #[arg(value_name = "INPUT")]
    input: String,

    /// Output file (use "-" for stdout)
    #[arg(short, long, value_name = "FILE")]
    output: String,

    /// Target backend
    #[arg(short, long, default_value = "c")]
    backend: Backend,

    /// Define dynamic dimension values (e.g., -D batch=32)
    #[arg(short = 'D', long = "define", value_name = "NAME=VALUE")]
    defines: Vec<String>,

    /// Optimization level (0-3)
    #[arg(short = 'O', long = "opt-level", default_value = "1", value_parser = clap::value_parser!(u8).range(0..=3))]
    opt_level: u8,

    /// Emit intermediate representation
    #[arg(long, value_name = "STAGE")]
    emit: Option<EmitStage>,

    /// Dump computation graph
    #[arg(long)]
    dump_graph: bool,

    /// Dump AST
    #[arg(long)]
    dump_ast: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Target backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Backend {
    /// C backend
    C,
    /// CUDA backend
    Cuda,
    /// Metal backend (macOS only)
    Metal,
    /// OpenCL backend
    Opencl,
    /// Rust backend
    Rust,
    /// OpenMP backend
    Openmp,
}

impl Backend {
    #[allow(dead_code)]
    fn file_extension(&self) -> &'static str {
        match self {
            Backend::C => "c",
            Backend::Cuda => "cu",
            Backend::Metal => "metal",
            Backend::Opencl => "cl",
            Backend::Rust => "rs",
            Backend::Openmp => "c",
        }
    }
}

/// Emit stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EmitStage {
    /// Emit AST
    Ast,
    /// Emit computation graph
    Graph,
    /// Emit generated code
    Code,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Read input
    let source = if args.input == "-" {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf)?;
        buf
    } else {
        fs::read_to_string(&args.input)?
    };

    if args.verbose {
        eprintln!("Parsing DSL...");
    }

    // Parse DSL
    let program = parse_program(&source).map_err(|e| format!("Parse error: {}", e))?;

    if args.verbose {
        eprintln!("Found {} graph(s)", program.graphs.len());
        for graph in &program.graphs {
            eprintln!("  - {}", graph.name);
        }
    }

    // Parse dynamic dimension definitions
    let dynamic_dims = parse_defines(&args.defines)?;

    if args.verbose && !dynamic_dims.is_empty() {
        eprintln!("Dynamic dimensions:");
        for (name, value) in &dynamic_dims {
            eprintln!("  {} = {}", name, value);
        }
    }

    // Build computation graphs
    if args.verbose {
        eprintln!("Building computation graphs...");
    }

    let builder = GraphBuilder::with_dynamic_dims(dynamic_dims);
    let built_graphs = builder
        .build_program(&program)
        .map_err(|e| format!("Graph build error: {}", e))?;

    if args.dump_graph {
        eprintln!("\n=== Computation Graphs ===");
        for graph in &built_graphs {
            eprintln!("\nGraph: {}", graph.name);
            eprintln!("  Inputs:");
            for (name, node) in &graph.inputs {
                eprintln!("    {}: {:?}", name, node.shape());
            }
            eprintln!("  Output: {:?}", graph.output.shape());
        }
        eprintln!();
    }

    // Emit graph stage
    if args.emit == Some(EmitStage::Graph) {
        let output = format!("{:#?}", built_graphs);
        write_output(&args.output, &output)?;
        return Ok(());
    }

    // Lower to AST
    if args.verbose {
        eprintln!("Lowering to AST...");
    }

    let mut all_asts = Vec::new();
    for graph in &built_graphs {
        let mut lowerer = Lowerer::new();
        let ast = lowerer.lower(&[graph.output.clone()]);
        all_asts.push((graph.name.clone(), ast));
    }

    // Apply optimization if opt_level > 0
    if args.opt_level > 0 {
        if args.verbose {
            eprintln!("Optimizing AST (level {})...", args.opt_level);
        }

        let rules = all_algebraic_rules();
        let max_iterations = match args.opt_level {
            1 => 10,
            2 => 50,
            3 => 100,
            _ => 10,
        };
        let mut optimizer = RuleBaseOptimizer::new(rules).with_max_iterations(max_iterations);

        all_asts = all_asts
            .into_iter()
            .map(|(name, ast)| {
                let optimized = optimizer.optimize(ast);
                (name, optimized)
            })
            .collect();
    }

    if args.dump_ast {
        eprintln!("\n=== AST ===");
        for (name, ast) in &all_asts {
            eprintln!("\nGraph: {}", name);
            eprintln!("{:#?}", ast);
        }
        eprintln!();
    }

    // Emit AST stage
    if args.emit == Some(EmitStage::Ast) {
        let output = format!("{:#?}", all_asts);
        write_output(&args.output, &output)?;
        return Ok(());
    }

    // Render to backend code
    if args.verbose {
        eprintln!("Rendering to {} code...", format!("{:?}", args.backend).to_lowercase());
    }

    let code = render_to_backend(&all_asts, args.backend)?;

    // Write output
    write_output(&args.output, &code)?;

    if args.verbose {
        eprintln!("Done! Output written to {}", args.output);
    }

    Ok(())
}

/// Parse -D definitions into a HashMap
fn parse_defines(defines: &[String]) -> Result<HashMap<String, i64>, String> {
    let mut map = HashMap::new();
    for def in defines {
        let parts: Vec<&str> = def.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(format!(
                "Invalid definition '{}': expected NAME=VALUE",
                def
            ));
        }
        let name = parts[0].trim().to_string();
        let value: i64 = parts[1]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid value for '{}': expected integer", name))?;
        map.insert(name, value);
    }
    Ok(map)
}

/// Render AST to backend code
fn render_to_backend(asts: &[(String, AstNode)], backend: Backend) -> Result<String, String> {
    let mut output = String::new();

    // Add header comment
    output.push_str(&format!(
        "// Generated by eclat-transpile\n// Backend: {:?}\n\n",
        backend
    ));

    match backend {
        Backend::C | Backend::Openmp => {
            // Use GenericRenderer for C-like code
            let renderer = GenericRenderer::new();
            for (name, ast) in asts {
                output.push_str(&format!("// Graph: {}\n", name));
                let code = renderer.render(ast);
                output.push_str(&code);
                output.push_str("\n\n");
            }
        }
        Backend::Cuda => {
            // CUDA uses similar rendering with different headers
            output.push_str("#include <cuda_runtime.h>\n");
            output.push_str("#include <math.h>\n\n");
            let renderer = GenericRenderer::new();
            for (name, ast) in asts {
                output.push_str(&format!("// Graph: {}\n", name));
                let code = renderer.render(ast);
                output.push_str(&code);
                output.push_str("\n\n");
            }
        }
        Backend::Metal => {
            output.push_str("#include <metal_stdlib>\n");
            output.push_str("using namespace metal;\n\n");
            let renderer = GenericRenderer::new();
            for (name, ast) in asts {
                output.push_str(&format!("// Graph: {}\n", name));
                let code = renderer.render(ast);
                output.push_str(&code);
                output.push_str("\n\n");
            }
        }
        Backend::Opencl => {
            output.push_str("// OpenCL kernel\n\n");
            let renderer = GenericRenderer::new();
            for (name, ast) in asts {
                output.push_str(&format!("// Graph: {}\n", name));
                let code = renderer.render(ast);
                output.push_str(&code);
                output.push_str("\n\n");
            }
        }
        Backend::Rust => {
            output.push_str("// Rust code\n\n");
            let renderer = GenericRenderer::new();
            for (name, ast) in asts {
                output.push_str(&format!("// Graph: {}\n", name));
                let code = renderer.render(ast);
                // Note: GenericRenderer produces C-like code, not idiomatic Rust
                // A proper Rust renderer would need different implementation
                output.push_str(&code);
                output.push_str("\n\n");
            }
        }
    }

    Ok(output)
}

/// Write output to file or stdout
fn write_output(path: &str, content: &str) -> io::Result<()> {
    if path == "-" {
        io::stdout().write_all(content.as_bytes())?;
    } else {
        fs::write(path, content)?;
    }
    Ok(())
}
