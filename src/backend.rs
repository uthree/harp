use crate::compiler::{Compiler, GccCompiler, GccCompileOptions};
use crate::lower;
use crate::renderer::{CStyleRenderer, Renderer};
use crate::tensor::Variable;
use crate::uop::UOp;
use std::sync::{Arc, Mutex};

pub trait Backend {
    fn compile_and_exec(&self, uop: &UOp, args: &[&Variable]);
    fn set_optimization_level(&self, level: u8);
}

pub struct CpuBackend {
    compiler: GccCompiler,
    renderer: CStyleRenderer,
    compile_options: Mutex<GccCompileOptions>,
}

impl CpuBackend {
    pub fn new() -> Self {
        let compiler = GccCompiler;
        if !compiler.is_available() {
            panic!("gcc not found");
        }
        Self {
            compiler,
            renderer: CStyleRenderer,
            compile_options: Mutex::new(GccCompileOptions::default()),
        }
    }
}

impl Backend for CpuBackend {
    fn set_optimization_level(&self, level: u8) {
        let mut options = self.compile_options.lock().unwrap();
        options.optimization_level = level;
    }

    fn compile_and_exec(&self, uop: &UOp, args: &[&Variable]) {
        // 1. Lowering (UOpグラフ -> UOpツリー)
        let ast = lower::lower(uop);
        println!("--- Lowered AST ---");
        // TODO: A proper Debug print for UOp is needed for this to be useful
        // println!("{:#?}", ast);

        // 2. Rendering (UOpツリー -> String)
        let code = self.renderer.render(&ast);
        println!("--- Rendered Code ---");
        println!("{}", code);

        // 3. Compiling (String -> Kernel)
        let options = self.compile_options.lock().unwrap();
        let kernel = self.compiler.compile(&code, &options).unwrap();
        println!("--- Compilation Done ---");

        // 4. Executing (Kernel)
        kernel.exec(args);
        println!("--- Execution Done ---");
    }
}
