//! eclat-transpile: DSL to backend code transpiler CLI
//!
//! Usage:
//!   eclat-transpile [OPTIONS] <INPUT> -o <OUTPUT>
//!
//! Examples:
//!   eclat-transpile softmax.ecl -o softmax.c -b c
//!   eclat-transpile model.ecl -o model.cu -b cuda
//!   eclat-transpile model.ecl -o model.c -D batch=32
//!   eclat-transpile model.ecl -o model.c --viz  # Interactive optimization viewer

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};

use clap::{Parser, ValueEnum};

use eclat::ast::AstNode;
use eclat::backend::renderer::Renderer;
use eclat::backend::{
    CompilationPipeline, DeviceKind, OptimizationConfig, mark_parallel_for_openmp,
};
use eclat_backend_c::CRenderer;
use eclat_backend_cuda::CudaRenderer;
#[cfg(target_os = "macos")]
use eclat_backend_metal::MetalRenderer;
use eclat_backend_opencl::OpenCLRenderer;
use eclat_backend_openmp::OpenMPRenderer;
use eclat_backend_rust::RustRenderer;
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

    /// Visualize optimization history (requires --features viz)
    #[cfg(feature = "viz")]
    #[arg(long)]
    viz: bool,
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

    fn to_device_kind(&self) -> DeviceKind {
        match self {
            Backend::C => DeviceKind::C,
            Backend::Cuda => DeviceKind::Cuda,
            Backend::Metal => DeviceKind::Metal,
            Backend::Opencl => DeviceKind::OpenCL,
            Backend::Rust => DeviceKind::Rust,
            Backend::Openmp => DeviceKind::OpenMP,
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

    let mut builder = GraphBuilder::with_dynamic_dims(dynamic_dims);
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

    // Create compilation pipeline
    let device_kind = args.backend.to_device_kind();
    let opt_config = OptimizationConfig::level(args.opt_level);
    let pipeline = CompilationPipeline::new(device_kind).with_optimization(opt_config);

    // Lower to AST
    if args.verbose {
        eprintln!("Lowering to AST...");
    }

    let mut all_asts = Vec::new();
    for graph in &built_graphs {
        let ast = pipeline
            .lower(&[graph.output.clone()])
            .map_err(|e| format!("Lowering error: {}", e))?;
        all_asts.push((graph.name.clone(), ast));
    }

    // Apply optimization if opt_level > 0
    #[cfg(feature = "viz")]
    let mut all_histories: Vec<eclat::opt::ast::history::OptimizationHistory> = Vec::new();

    if args.opt_level > 0 {
        if args.verbose {
            eprintln!("Optimizing AST (level {})...", args.opt_level);
        }

        #[cfg(feature = "viz")]
        {
            all_asts = all_asts
                .into_iter()
                .map(|(name, ast)| {
                    let (optimized, history) = pipeline.optimize_with_history(ast);
                    all_histories.push(history);
                    (name, optimized)
                })
                .collect();
        }

        #[cfg(not(feature = "viz"))]
        {
            all_asts = all_asts
                .into_iter()
                .map(|(name, ast)| {
                    let optimized = pipeline.optimize(ast);
                    (name, optimized)
                })
                .collect();
        }
    }

    // Mark outermost loops as parallel for OpenMP backend
    if args.backend == Backend::Openmp {
        if args.verbose {
            eprintln!("Marking loops for OpenMP parallelization...");
        }
        all_asts = all_asts
            .into_iter()
            .map(|(name, ast)| (name, mark_parallel_for_openmp(ast)))
            .collect();
    }

    // Launch visualization TUI if --viz flag is set
    #[cfg(feature = "viz")]
    if args.viz {
        if all_histories.is_empty() {
            eprintln!("warning: No optimization history to visualize (opt_level may be 0)");
        } else {
            // Merge all histories into one (for now, just use the first one)
            // TODO: Support multi-graph visualization
            let history = all_histories.remove(0);
            if args.verbose {
                eprintln!("Launching optimization visualizer...");
            }
            launch_viz(history, args.backend)?;
        }
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
        eprintln!(
            "Rendering to {} code...",
            format!("{:?}", args.backend).to_lowercase()
        );
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
            return Err(format!("Invalid definition '{}': expected NAME=VALUE", def));
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
        Backend::C => {
            let renderer = CRenderer::new();
            for (name, ast) in asts {
                output.push_str(&format!("// Graph: {}\n", name));
                let code = renderer.render(ast);
                output.push_str(code.as_str());
                output.push_str("\n\n");
            }
        }
        Backend::Openmp => {
            let renderer = OpenMPRenderer::new();
            for (name, ast) in asts {
                output.push_str(&format!("// Graph: {}\n", name));
                let code = renderer.render(ast);
                output.push_str(code.as_str());
                output.push_str("\n\n");
            }
        }
        Backend::Cuda => {
            let renderer = CudaRenderer::new();
            for (name, ast) in asts {
                output.push_str(&format!("// Graph: {}\n", name));
                let code = renderer.render(ast);
                output.push_str(code.as_str());
                output.push_str("\n\n");
            }
        }
        Backend::Metal => {
            #[cfg(target_os = "macos")]
            {
                let renderer = MetalRenderer::new();
                for (name, ast) in asts {
                    output.push_str(&format!("// Graph: {}\n", name));
                    let code = renderer.render(ast);
                    output.push_str(code.as_str());
                    output.push_str("\n\n");
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                return Err("Metal backend is only available on macOS".to_string());
            }
        }
        Backend::Opencl => {
            let renderer = OpenCLRenderer::new();
            for (name, ast) in asts {
                output.push_str(&format!("// Graph: {}\n", name));
                let code = renderer.render(ast);
                output.push_str(code.as_str());
                output.push_str("\n\n");
            }
        }
        Backend::Rust => {
            let renderer = RustRenderer::new();
            for (name, ast) in asts {
                output.push_str(&format!("// Graph: {}\n", name));
                let code = renderer.render(ast);
                output.push_str(code.as_str());
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

/// Launch optimization visualization TUI
#[cfg(feature = "viz")]
fn launch_viz(
    history: eclat::opt::ast::history::OptimizationHistory,
    backend: Backend,
) -> Result<(), Box<dyn std::error::Error>> {
    match backend {
        Backend::C => {
            eclat_viz::run_with_renderer(history, CRenderer::new())?;
        }
        Backend::Openmp => {
            eclat_viz::run_with_renderer(history, OpenMPRenderer::new())?;
        }
        Backend::Cuda => {
            eclat_viz::run_with_renderer(history, CudaRenderer::new())?;
        }
        Backend::Metal => {
            #[cfg(target_os = "macos")]
            {
                eclat_viz::run_with_renderer(history, MetalRenderer::new())?;
            }
            #[cfg(not(target_os = "macos"))]
            {
                return Err("Metal backend is only available on macOS".into());
            }
        }
        Backend::Opencl => {
            eclat_viz::run_with_renderer(history, OpenCLRenderer::new())?;
        }
        Backend::Rust => {
            eclat_viz::run_with_renderer(history, RustRenderer::new())?;
        }
    }

    Ok(())
}
