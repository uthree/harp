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

use eclat::ast::{AstNode, ParallelInfo, ParallelKind};
use eclat::backend::renderer::Renderer;
use eclat::lowerer::Lowerer;
#[cfg(feature = "viz")]
use eclat::opt::ast::history::OptimizationHistory;
#[cfg(not(feature = "viz"))]
use eclat::opt::ast::AstOptimizer;
use eclat::opt::ast::{
    AstSuggester, BeamSearchOptimizer, CompositeSuggester, FunctionInliningSuggester,
    GroupParallelizationSuggester, LoopFusionSuggester, LoopInliningSuggester,
    LoopInterchangeSuggester, LoopTilingSuggester, RuleBaseSuggester,
};
use eclat::opt::ast::rules::all_algebraic_rules;
use eclat_backend_c::CRenderer;
use eclat_backend_cuda::CudaRenderer;
use eclat_backend_opencl::OpenCLRenderer;
use eclat_backend_openmp::OpenMPRenderer;
use eclat_backend_rust::RustRenderer;
#[cfg(target_os = "macos")]
use eclat_backend_metal::MetalRenderer;
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
    #[cfg(feature = "viz")]
    let mut all_histories: Vec<OptimizationHistory> = Vec::new();

    if args.opt_level > 0 {
        if args.verbose {
            eprintln!("Optimizing AST (level {})...", args.opt_level);
        }

        // Configure optimization based on level and backend
        let (beam_width, max_steps) = match args.opt_level {
            1 => (4, 50),
            2 => (8, 100),
            3 => (16, 200),
            _ => (4, 50),
        };

        // Build suggesters based on backend capabilities
        let rules = all_algebraic_rules();
        let mut suggesters: Vec<Box<dyn AstSuggester>> = vec![
            // Rule-based (algebraic transformations, constant folding)
            Box::new(RuleBaseSuggester::new(rules)),
            // Loop transformations
            Box::new(LoopTilingSuggester::new()),
            Box::new(LoopInliningSuggester::new()),
            Box::new(LoopInterchangeSuggester::new()),
            Box::new(LoopFusionSuggester::new()),
            Box::new(FunctionInliningSuggester::with_default_limit()),
        ];

        // Add parallelization suggester for GPU backends
        let is_gpu_backend = matches!(
            args.backend,
            Backend::Cuda | Backend::Metal | Backend::Opencl
        );
        if is_gpu_backend {
            suggesters.push(Box::new(GroupParallelizationSuggester::new()));
        }

        let suggester = CompositeSuggester::new(suggesters);
        let mut optimizer = BeamSearchOptimizer::new(suggester)
            .with_beam_width(beam_width)
            .with_max_steps(max_steps)
            .without_progress();

        #[cfg(feature = "viz")]
        {
            all_asts = all_asts
                .into_iter()
                .map(|(name, ast)| {
                    let (optimized, history) = optimizer.optimize_with_history(ast);
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
                    let optimized = optimizer.optimize(ast);
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

/// Mark outermost loops as parallel for OpenMP
fn mark_parallel_for_openmp(ast: AstNode) -> AstNode {
    mark_parallel_recursive(ast, true)
}

fn mark_parallel_recursive(ast: AstNode, is_outermost: bool) -> AstNode {
    match ast {
        AstNode::Program { functions, execution_waves } => {
            let functions = functions
                .into_iter()
                .map(|f| mark_parallel_recursive(f, true))
                .collect();
            AstNode::Program { functions, execution_waves }
        }
        AstNode::Kernel {
            name,
            params,
            return_type,
            body,
            default_grid_size,
            default_thread_group_size,
        } => AstNode::Kernel {
            name,
            params,
            return_type,
            body: Box::new(mark_parallel_recursive(*body, true)),
            default_grid_size,
            default_thread_group_size,
        },
        AstNode::Range {
            var,
            start,
            step,
            stop,
            body,
            parallel,
        } => {
            // Mark outermost loop as parallel, inner loops stay sequential
            let new_parallel = if is_outermost {
                ParallelInfo {
                    is_parallel: true,
                    kind: ParallelKind::OpenMP,
                    reductions: parallel.reductions,
                }
            } else {
                parallel
            };
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body: Box::new(mark_parallel_recursive(*body, false)),
                parallel: new_parallel,
            }
        }
        AstNode::Block { statements, scope } => {
            let statements = statements
                .into_iter()
                .map(|s| mark_parallel_recursive(s, is_outermost))
                .collect();
            AstNode::Block { statements, scope }
        }
        AstNode::If {
            condition,
            then_body,
            else_body,
        } => AstNode::If {
            condition,
            then_body: Box::new(mark_parallel_recursive(*then_body, false)),
            else_body: else_body.map(|e| Box::new(mark_parallel_recursive(*e, false))),
        },
        // Other nodes pass through unchanged
        other => other,
    }
}

/// Launch optimization visualization TUI
#[cfg(feature = "viz")]
fn launch_viz(history: OptimizationHistory, backend: Backend) -> Result<(), Box<dyn std::error::Error>> {
    use eclat::viz;

    match backend {
        Backend::C => {
            viz::run_with_renderer(history, CRenderer::new())?;
        }
        Backend::Openmp => {
            viz::run_with_renderer(history, OpenMPRenderer::new())?;
        }
        Backend::Cuda => {
            viz::run_with_renderer(history, CudaRenderer::new())?;
        }
        Backend::Metal => {
            #[cfg(target_os = "macos")]
            {
                viz::run_with_renderer(history, MetalRenderer::new())?;
            }
            #[cfg(not(target_os = "macos"))]
            {
                return Err("Metal backend is only available on macOS".into());
            }
        }
        Backend::Opencl => {
            viz::run_with_renderer(history, OpenCLRenderer::new())?;
        }
        Backend::Rust => {
            viz::run_with_renderer(history, RustRenderer::new())?;
        }
    }

    Ok(())
}
