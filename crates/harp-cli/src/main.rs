//! Harp DSL Compiler CLI

use clap::{Parser, Subcommand};
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;

use harp::backend::metal::MetalCompiler;
use harp::backend::opencl::OpenCLCompiler;
use harp::backend::{GenericPipeline, Renderer};
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
    Metal,
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

fn main() {
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

            let formatted = decompile(&graph, name);

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
        ),
    }
}

fn do_compile(
    input: Option<&PathBuf>,
    output: Option<&PathBuf>,
    target: Target,
    emit: EmitType,
) -> Result<(), Box<dyn std::error::Error>> {
    let (source, _name) = read_source(input)?;
    let graph = compile(&source)?;

    let result = match emit {
        EmitType::Code => {
            // Generate code for the target
            match target {
                Target::OpenCL => {
                    compile_to_code::<harp::backend::opencl::OpenCLRenderer, OpenCLCompiler>(graph)?
                }
                Target::Metal => {
                    #[cfg(target_os = "macos")]
                    {
                        compile_to_code::<harp::backend::metal::MetalRenderer, MetalCompiler>(
                            graph,
                        )?
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        return Err("Metal backend is only available on macOS".into());
                    }
                }
            }
        }
        EmitType::Graph => {
            format!("{:#?}", graph)
        }
        EmitType::Ast => {
            let module = parse(&source)?;
            format!("{:#?}", module)
        }
    };

    write_output(output, &result)
}

fn compile_to_code<R, C>(graph: harp::graph::Graph) -> Result<String, Box<dyn std::error::Error>>
where
    R: Renderer + Default + Clone + 'static,
    C: harp::backend::Compiler<CodeRepr = R::CodeRepr> + Default + Clone + 'static,
    C::Buffer: 'static,
{
    let mut pipeline = GenericPipeline::new(R::default(), C::default());

    let (program, _) = pipeline.optimize_graph_with_all_histories(graph)?;

    let renderer = R::default();
    let code = renderer.render(&program);
    Ok(code.into())
}
