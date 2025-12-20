//! Harp DSL Compiler CLI

use clap::{Parser, Subcommand};
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;

use harp::backend::c_like::CLikeRenderer;
#[cfg(target_os = "macos")]
use harp::backend::metal::MetalRenderer;
use harp::backend::opencl::OpenCLRenderer;
use harp::backend::{MultiPhaseConfig, Renderer, SubgraphMode, create_multi_phase_optimizer};
use harp::opt::ast::rules::all_algebraic_rules;
use harp::opt::ast::{
    AstOptimizer, BeamSearchOptimizer as AstBeamSearchOptimizer,
    CompositeSuggester as AstCompositeSuggester, FunctionInliningSuggester, LoopFusionSuggester,
    LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester, RuleBaseOptimizer,
};
use harp::opt::graph::GraphOptimizer;
use harp_dsl::{compile, decompile, parse};

#[derive(Parser)]
#[command(name = "harpc")]
#[command(about = "Harp DSL Compiler - Compile computation graphs to optimized kernels")]
#[command(version)]
struct Cli {
    /// Input .harp file (use '-' for stdin)
    /// If provided without subcommand, compiles the file
    #[arg(value_name = "FILE")]
    input: Option<PathBuf>,

    /// Output file (default: stdout)
    #[arg(short, long, global = true)]
    output: Option<PathBuf>,

    /// Target backend
    #[arg(short, long, default_value = "opencl", global = true)]
    target: Target,

    /// Output type
    #[arg(long, default_value = "code", global = true)]
    emit: EmitType,

    /// Embed original DSL source code as comment in output
    #[arg(long, default_value = "false", global = true)]
    embed_source: bool,

    /// Subgraph handling mode
    ///
    /// How to process subgraphs in the computation graph:
    /// - inline: Expand subgraphs inline (default, single large kernel)
    /// - separate: Generate separate kernel functions for each subgraph
    /// - skip: Skip subgraph processing (for debugging)
    #[arg(long, default_value = "inline", global = true, value_enum)]
    subgraph_mode: CliSubgraphMode,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a .harp file to optimized kernel code
    Compile {
        /// Input .harp file (use '-' for stdin)
        #[arg(value_name = "FILE")]
        input: Option<PathBuf>,
    },

    /// Check a .harp file for errors without compiling
    Check {
        /// Input .harp file
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// Parse and print the AST of a .harp file
    Ast {
        /// Input .harp file
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// Format a .harp file
    Fmt {
        /// Input .harp file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Write result to file instead of stdout
        #[arg(short, long)]
        write: bool,
    },
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum Target {
    #[value(name = "opencl", alias = "cl")]
    OpenCL,
    #[cfg(target_os = "macos")]
    Metal,
}

impl Target {
    /// Get the file extension for this target
    fn extension(&self) -> &'static str {
        match self {
            Target::OpenCL => "c",
            #[cfg(target_os = "macos")]
            Target::Metal => "metal",
        }
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum EmitType {
    /// Emit generated kernel code
    Code,
    /// Emit graph structure (debug)
    Graph,
    /// Emit DSL AST (debug)
    Ast,
}

/// CLI用のサブグラフ処理モード
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum CliSubgraphMode {
    /// Expand subgraphs inline (single large kernel)
    #[default]
    Inline,
    /// Generate separate kernel functions for each subgraph
    Separate,
    /// Skip subgraph processing (for debugging)
    Skip,
}

impl From<CliSubgraphMode> for SubgraphMode {
    fn from(cli_mode: CliSubgraphMode) -> Self {
        match cli_mode {
            CliSubgraphMode::Inline => SubgraphMode::Inline,
            CliSubgraphMode::Separate => SubgraphMode::SeparateKernels,
            CliSubgraphMode::Skip => SubgraphMode::Skip,
        }
    }
}

fn main() {
    // RUST_LOG環境変数でログレベルを制御（例: RUST_LOG=debug）
    env_logger::init();

    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Read source from file or stdin
fn read_source(input: Option<&PathBuf>) -> Result<(String, String), Box<dyn std::error::Error>> {
    match input {
        Some(path) if path.to_str() == Some("-") => {
            // Read from stdin
            let mut source = String::new();
            io::stdin().read_to_string(&mut source)?;
            Ok((source, "stdin".to_string()))
        }
        Some(path) => {
            let source = fs::read_to_string(path)?;
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("graph")
                .to_string();
            Ok((source, name))
        }
        None => {
            // Check if stdin has data (pipe)
            if !io::stdin().is_terminal() {
                let mut source = String::new();
                io::stdin().read_to_string(&mut source)?;
                Ok((source, "stdin".to_string()))
            } else {
                Err("No input file specified. Use '-' to read from stdin.".into())
            }
        }
    }
}

/// Write output to file or stdout
fn write_output(output: Option<&PathBuf>, content: &str) -> Result<(), Box<dyn std::error::Error>> {
    match output {
        Some(path) => {
            fs::write(path, content)?;
        }
        None => {
            // Write to stdout without trailing newline if piping
            if io::stdout().is_terminal() {
                println!("{}", content);
            } else {
                print!("{}", content);
                io::stdout().flush()?;
            }
        }
    }
    Ok(())
}

/// Determine output filename based on input and target
/// Returns None if output should go to stdout (e.g., when input is from stdin and piping)
fn determine_output_path(
    input: Option<&PathBuf>,
    target: Target,
    emit: EmitType,
) -> Option<PathBuf> {
    // Only auto-generate output path for code emission
    if !matches!(emit, EmitType::Code) {
        return None;
    }

    let ext = target.extension();

    match input {
        Some(path) if path.to_str() != Some("-") => {
            // Use input filename with new extension
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("a");
            Some(PathBuf::from(format!("{}.{}", stem, ext)))
        }
        _ => {
            // stdin or "-": use default "a.{ext}" like C compilers use "a.out"
            // But only if stdout is a terminal (not piping)
            if io::stdout().is_terminal() {
                Some(PathBuf::from(format!("a.{}", ext)))
            } else {
                None // Piping to another command, use stdout
            }
        }
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        // Explicit compile subcommand
        Some(Commands::Compile { input }) => {
            let input_path = input.or(cli.input);
            do_compile(
                input_path.as_ref(),
                cli.output.as_ref(),
                cli.target,
                cli.emit,
                cli.embed_source,
                cli.subgraph_mode.into(),
            )
        }

        // Check subcommand
        Some(Commands::Check { input }) => {
            let source = fs::read_to_string(&input)?;
            let _graph = compile(&source)?;
            eprintln!("OK: {} compiled successfully", input.display());
            Ok(())
        }

        // AST subcommand
        Some(Commands::Ast { input }) => {
            let source = fs::read_to_string(&input)?;
            let module = parse(&source)?;
            write_output(cli.output.as_ref(), &format!("{:#?}", module))
        }

        // Format subcommand
        Some(Commands::Fmt { input, write }) => {
            let source = fs::read_to_string(&input)?;
            let graph = compile(&source)?;

            let name = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("graph");

            let _ = name; // unused for now
            let formatted = decompile(&graph);

            if write {
                fs::write(&input, &formatted)?;
                eprintln!("Formatted: {}", input.display());
            } else {
                write_output(cli.output.as_ref(), &formatted)?;
            }
            Ok(())
        }

        // No subcommand: default to compile if input is provided or stdin has data
        None => do_compile(
            cli.input.as_ref(),
            cli.output.as_ref(),
            cli.target,
            cli.emit,
            cli.embed_source,
            cli.subgraph_mode.into(),
        ),
    }
}

fn do_compile(
    input: Option<&PathBuf>,
    explicit_output: Option<&PathBuf>,
    target: Target,
    emit: EmitType,
    embed_source: bool,
    subgraph_mode: SubgraphMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let (source, _name) = read_source(input)?;
    let graph = compile(&source)?;

    let generated = match emit {
        EmitType::Code => match target {
            Target::OpenCL => compile_to_code::<OpenCLRenderer>(graph, subgraph_mode)?,
            #[cfg(target_os = "macos")]
            Target::Metal => compile_to_code::<MetalRenderer>(graph, subgraph_mode)?,
        },
        EmitType::Graph => {
            format!("{:#?}", graph)
        }
        EmitType::Ast => {
            let module = parse(&source)?;
            format!("{:#?}", module)
        }
    };

    // Optionally embed original DSL source as comment
    let result = if embed_source && matches!(emit, EmitType::Code) {
        let comment_prefix = match target {
            Target::OpenCL => "//",
            #[cfg(target_os = "macos")]
            Target::Metal => "//",
        };
        let mut output = String::new();
        output.push_str(&format!(
            "{} === Original Harp DSL Source ===\n",
            comment_prefix
        ));
        for line in source.lines() {
            output.push_str(&format!("{} {}\n", comment_prefix, line));
        }
        output.push_str(&format!("{} === End of Source ===\n\n", comment_prefix));
        output.push_str(&generated);
        output
    } else {
        generated
    };

    // Determine output path: use explicit output if provided, otherwise auto-determine
    let auto_output = determine_output_path(input, target, emit);
    let output_path = explicit_output.or(auto_output.as_ref());

    // Write output and print message if writing to file
    if let Some(path) = output_path {
        fs::write(path, &result)?;
        eprintln!("Wrote output to: {}", path.display());
        Ok(())
    } else {
        write_output(None, &result)
    }
}

fn compile_to_code<R>(
    graph: harp::graph::Graph,
    subgraph_mode: SubgraphMode,
) -> Result<String, Box<dyn std::error::Error>>
where
    R: Renderer + CLikeRenderer + Default + Clone + 'static,
{
    // Phase 1: Graph optimization
    let config = MultiPhaseConfig::new()
        .with_beam_width(4)
        .with_max_steps(5000)
        .with_progress(false)
        .with_collect_logs(false)
        .with_subgraph_mode(subgraph_mode);

    let optimizer = create_multi_phase_optimizer(config);
    let (optimized_graph, _) = optimizer.optimize_with_history(graph);

    // Phase 2: Lower to AST
    let program = harp::lowerer::extract_program(optimized_graph);

    // Phase 3: AST optimization
    let optimized_program = optimize_ast(program);

    // Phase 4: Render to code
    let mut renderer = R::default();
    let code = renderer.render_program_clike(&optimized_program);
    Ok(code)
}

fn optimize_ast(program: harp::ast::AstNode) -> harp::ast::AstNode {
    // Phase 1: Rule-based optimization
    let rule_optimizer = RuleBaseOptimizer::new(all_algebraic_rules());
    let rule_optimized = rule_optimizer.optimize(program);

    // Phase 2: Loop optimization with beam search
    let loop_suggester = AstCompositeSuggester::new(vec![
        Box::new(LoopTilingSuggester::new()),
        Box::new(LoopInliningSuggester::new()),
        Box::new(LoopInterchangeSuggester::new()),
        Box::new(LoopFusionSuggester::new()),
        Box::new(FunctionInliningSuggester::with_default_limit()),
    ]);

    let loop_optimizer = AstBeamSearchOptimizer::new(loop_suggester)
        .with_beam_width(4)
        .with_max_steps(5000)
        .with_progress(false);

    let (optimized, _) = loop_optimizer.optimize_with_history(rule_optimized);

    optimized
}
