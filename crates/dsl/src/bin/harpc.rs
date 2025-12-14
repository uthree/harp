//! Harp DSL Compiler CLI

use clap::{Parser, Subcommand};
use std::fs;
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
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a .harp file to optimized kernel code
    Compile {
        /// Input .harp file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Target backend
        #[arg(short, long, default_value = "opencl")]
        target: Target,

        /// Output type
        #[arg(long, default_value = "code")]
        emit: EmitType,
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

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Compile {
            input,
            output,
            target,
            emit,
        } => {
            let source = fs::read_to_string(&input)?;
            let graph = compile(&source)?;

            let result = match emit {
                EmitType::Code => {
                    // Generate code for the target
                    match target {
                        Target::OpenCL => compile_to_code::<
                            harp::backend::opencl::OpenCLRenderer,
                            OpenCLCompiler,
                        >(graph)?,
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

            if let Some(output_path) = output {
                fs::write(output_path, result)?;
            } else {
                println!("{}", result);
            }
        }

        Commands::Check { input } => {
            let source = fs::read_to_string(&input)?;
            let _graph = compile(&source)?;
            println!("OK: {} compiled successfully", input.display());
        }

        Commands::Ast { input } => {
            let source = fs::read_to_string(&input)?;
            let module = parse(&source)?;
            println!("{:#?}", module);
        }

        Commands::Fmt { input, write } => {
            let source = fs::read_to_string(&input)?;
            let graph = compile(&source)?;

            // Extract graph name from filename
            let name = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("graph");

            let formatted = decompile(&graph, name);

            if write {
                fs::write(&input, &formatted)?;
                println!("Formatted: {}", input.display());
            } else {
                println!("{}", formatted);
            }
        }
    }

    Ok(())
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
