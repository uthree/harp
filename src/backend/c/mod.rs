use crate::backend::codegen::Instruction;
use crate::backend::Compiler;
use crate::backend::Kernel;
use crate::backend::renderer::{Render, Renderer};
use crate::op::*;
use libloading::{Library, Symbol};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::sync::Arc;
use tempfile::TempDir;

/// A renderer for generating C code from a computation graph.
pub struct CRenderer;

impl Renderer for CRenderer {
    fn render_op(&self, op: &dyn Operator, operands: &[String]) -> Option<String> {
        let op_any = op.as_any();
        match op_any {
            _ if op_any.is::<OpAdd>() => Some(self.render(op_any.downcast_ref::<OpAdd>().unwrap(), operands)),
            _ if op_any.is::<OpMul>() => Some(self.render(op_any.downcast_ref::<OpMul>().unwrap(), operands)),
            _ if op_any.is::<Sin>() => Some(self.render(op_any.downcast_ref::<Sin>().unwrap(), operands)),
            _ if op_any.is::<Const>() => Some(self.render(op_any.downcast_ref::<Const>().unwrap(), operands)),
            _ if op_any.is::<Variable>() => Some(self.render(op_any.downcast_ref::<Variable>().unwrap(), operands)),
            _ if op_any.is::<LoopVariable>() => Some(self.render(op_any.downcast_ref::<LoopVariable>().unwrap(), operands)),
            _ if op_any.is::<Load>() => Some(self.render(op_any.downcast_ref::<Load>().unwrap(), operands)),
            _ => None,
        }
    }

    fn render_function(
        &self,
        fn_name: &str,
        args: &[(&str, &str)],
        body: &[Instruction],
        return_type: &str,
    ) -> String {
        let rendered_body = self.render_body(body, 1);
        let arg_string = args
            .iter()
            .map(|(dtype, name)| format!("{} {}", dtype, name))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "{} {}({}) {{\n{}}}",
            return_type, fn_name, arg_string, rendered_body
        )
    }
}

impl CRenderer {
    fn render_body(&self, instructions: &[Instruction], indent_level: usize) -> String {
        let indent = "    ".repeat(indent_level);
        let mut body_str = String::new();
        for inst in instructions {
            let line = match inst {
                Instruction::DeclareVariable { name, dtype, value } => {
                    format!("{} {} = {};", dtype, name, value)
                }
                Instruction::Statement { code } => format!("{};", code),
                Instruction::Loop { count, body } => {
                    let rendered_body = self.render_body(body, indent_level + 1);
                    format!(
                        "for (int i = 0; i < {}; ++i) {{\n{}\n{}}}",
                        count, rendered_body, indent
                    )
                }
                Instruction::Return { value } => format!("return {};", value),
            };
            body_str.push_str(&indent);
            body_str.push_str(&line);
            body_str.push('\n');
        }
        body_str
    }
}

impl Render<Load> for CRenderer {
    fn render(&self, op: &Load, operands: &[String]) -> String {
        format!("{}[{}]", op.0, operands[0])
    }
}

impl Render<LoopVariable> for CRenderer {
    fn render(&self, _op: &LoopVariable, _operands: &[String]) -> String { "i".to_string() }
}
impl Render<OpAdd> for CRenderer {
    fn render(&self, _op: &OpAdd, operands: &[String]) -> String { format!("({} + {})", operands[0], operands[1]) }
}
impl Render<OpMul> for CRenderer {
    fn render(&self, _op: &OpMul, operands: &[String]) -> String { format!("({} * {})", operands[0], operands[1]) }
}
impl Render<Sin> for CRenderer {
    fn render(&self, _op: &Sin, operands: &[String]) -> String { format!("sin({})", operands[0]) }
}
impl Render<Const> for CRenderer {
    fn render(&self, op: &Const, _operands: &[String]) -> String { format!("{:?}", op.0) }
}
impl Render<Variable> for CRenderer {
    fn render(&self, op: &Variable, _operands: &[String]) -> String { op.0.clone() }
}

/// A compiled C kernel loaded from a dynamic library.
pub struct CKernel {
    _lib: Arc<Library>,
    compute: Symbol<'static, unsafe extern "C" fn() -> f32>,
}

impl Kernel for CKernel {
    fn execute(&self) -> f32 {
        unsafe { (self.compute)() }
    }
}

/// A compiler for C code using `gcc`.
pub struct CCompiler;

impl Compiler for CCompiler {
    type Kernel = CKernel;

    fn is_available(&self) -> bool {
        Command::new("gcc").arg("--version").output().is_ok()
    }

    fn compile(&self, code: &str) -> Result<Self::Kernel, Box<dyn Error>> {
        let temp_dir = TempDir::new()?;
        let source_path = temp_dir.path().join("kernel.c");
        let lib_path = temp_dir.path().join(if cfg!(target_os = "windows") {
            "kernel.dll"
        } else if cfg!(target_os = "macos") {
            "libkernel.dylib"
        } else {
            "libkernel.so"
        });

        let mut source_file = File::create(&source_path)?;
        source_file.write_all(code.as_bytes())?;

        let output = Command::new("gcc")
            .arg("-shared")
            .arg("-o")
            .arg(&lib_path)
            .arg(&source_path)
            .arg("-fPIC")
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "gcc compilation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ).into());
        }

        unsafe {
            let lib = Arc::new(Library::new(&lib_path)?);
            let compute: Symbol<unsafe extern "C" fn() -> f32> = lib.get(b"compute")?;
            
            let compute = std::mem::transmute(compute);

            Ok(CKernel {
                _lib: lib.clone(),
                compute,
            })
        }
    }
}
