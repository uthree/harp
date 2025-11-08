use std::collections::HashMap;

use crate::ast::AstNode;
pub mod c_like;
pub mod metal;

// Re-export commonly used types
pub use metal::{MetalCode, MetalRenderer};

#[cfg(target_os = "macos")]
pub use metal::{MetalBuffer, MetalCompiler, MetalKernel};

// レンダラー。
// ASTを受け取って文字列としてレンダリングする
pub trait Renderer {
    type CodeRepr;
    type Option;
    fn render(&self, ast: AstNode) -> Self::CodeRepr;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, _option: Self::Option) {} // default implementation is "do nothing".
}
pub trait Compiler {
    type CodeRepr;
    type Buffer: Buffer;
    type Kernel: Kernel<Buffer = Self::Buffer>;
    type Option;
    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, _option: Self::Option) {} // default implementation is "do nothing".
    fn compile(&mut self, code: &Self::CodeRepr) -> Self::Kernel;
    fn create_buffer(&self, shape: Vec<usize>, element_size: usize) -> Self::Buffer;
}
pub trait Buffer {
    // get buffer size
    fn shape(&self) -> Vec<usize>;
    // TODO: 初期化と（CPU上の）バイト列(u8の配列?)への相互変換
}

pub trait Kernel {
    type Buffer: Buffer;
    fn signature(&self) -> KernelSignature;

    /// QueryBuilderを作成してメソッドチェーンでクエリを構築
    fn query(&self) -> QueryBuilder<'_, Self>
    where
        Self: Sized,
    {
        QueryBuilder::new(self)
    }
}

// 一通りの演算をまとめて行うためのパイプライン
pub trait Pipeline {
    type Compiler: Compiler;
    type Renderer: Renderer<CodeRepr = <Self::Compiler as Compiler>::CodeRepr>;
    
}

// カーネルへの指示をまとめる構造体
#[derive(Debug)]
pub struct Query<'a, B: Buffer> {
    pub inputs: HashMap<String, &'a B>, // inputsは読み取り専用なので借用
    pub outputs: HashMap<String, B>,    // outputsは書き込み対象
    pub shape_vars: HashMap<String, usize>, // 動的shape変数の値
}

impl<'a, B: Buffer> Query<'a, B> {
    /// 新しいQueryを作成
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            shape_vars: HashMap::new(),
        }
    }
}

impl<'a, B: Buffer> Default for Query<'a, B> {
    fn default() -> Self {
        Self::new()
    }
}

/// 出力バッファの指定方法
enum OutputSpec<B: Buffer> {
    Provided(B), // 既存のバッファを使用
    AutoCreate {
        shape: Vec<usize>,
        element_size: usize,
    }, // 自動生成
}

/// QueryBuilderを使ってKernelの実行クエリを構築する
pub struct QueryBuilder<'a, K: Kernel> {
    kernel: &'a K,
    inputs: HashMap<String, &'a K::Buffer>,
    outputs: HashMap<String, OutputSpec<K::Buffer>>,
    shape_vars: HashMap<String, usize>,
}

impl<'a, K: Kernel> QueryBuilder<'a, K> {
    /// 新しいQueryBuilderを作成
    pub fn new(kernel: &'a K) -> Self {
        Self {
            kernel,
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            shape_vars: HashMap::new(),
        }
    }

    /// 入力バッファを追加
    pub fn input(mut self, name: impl Into<String>, buffer: &'a K::Buffer) -> Self {
        self.inputs.insert(name.into(), buffer);
        self
    }

    /// 出力バッファを追加（既存のバッファを使用）
    pub fn output(mut self, name: impl Into<String>, buffer: K::Buffer) -> Self {
        self.outputs
            .insert(name.into(), OutputSpec::Provided(buffer));
        self
    }

    /// 出力バッファを自動生成するよう指定
    pub fn auto_output(
        mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        element_size: usize,
    ) -> Self {
        self.outputs.insert(
            name.into(),
            OutputSpec::AutoCreate {
                shape,
                element_size,
            },
        );
        self
    }

    /// shape変数の値を設定
    pub fn shape_var(mut self, name: impl Into<String>, value: usize) -> Self {
        self.shape_vars.insert(name.into(), value);
        self
    }

    /// Queryを構築（検証付き）
    ///
    /// 注意: auto_output()で指定された出力バッファがある場合はエラーになります。
    /// その場合は finalize() を使用してください。
    pub fn build(self) -> Result<Query<'a, K::Buffer>, String> {
        let signature = self.kernel.signature();

        // 入力バッファの検証
        for input_sig in &signature.inputs {
            if !self.inputs.contains_key(&input_sig.name) {
                return Err(format!("Missing input buffer: {}", input_sig.name));
            }
        }

        // 出力バッファの検証と取り出し
        let mut outputs = HashMap::new();
        for output_sig in &signature.outputs {
            match self.outputs.get(&output_sig.name) {
                Some(OutputSpec::Provided(_)) => {
                    // 所有権を移動させる必要があるので、あとで処理
                }
                Some(OutputSpec::AutoCreate { .. }) => {
                    return Err(format!(
                        "Output buffer '{}' is marked for auto-creation. Use finalize() instead of build().",
                        output_sig.name
                    ));
                }
                None => {
                    return Err(format!("Missing output buffer: {}", output_sig.name));
                }
            }
        }

        // 出力バッファを取り出す
        for (name, spec) in self.outputs {
            match spec {
                OutputSpec::Provided(buffer) => {
                    outputs.insert(name, buffer);
                }
                OutputSpec::AutoCreate { .. } => unreachable!(),
            }
        }

        // shape変数の検証
        for var_name in &signature.shape_vars {
            if !self.shape_vars.contains_key(var_name) {
                return Err(format!("Missing shape variable: {}", var_name));
            }
        }

        Ok(Query {
            inputs: self.inputs,
            outputs,
            shape_vars: self.shape_vars,
        })
    }

    /// Compilerを使って出力バッファを自動生成し、Queryを構築
    pub fn finalize<C>(self, compiler: &C) -> Result<Query<'a, K::Buffer>, String>
    where
        C: Compiler<Buffer = K::Buffer>,
    {
        let signature = self.kernel.signature();

        // 入力バッファの検証
        for input_sig in &signature.inputs {
            if !self.inputs.contains_key(&input_sig.name) {
                return Err(format!("Missing input buffer: {}", input_sig.name));
            }
        }

        // 出力バッファの検証と生成
        let mut outputs = HashMap::new();
        for output_sig in &signature.outputs {
            match self.outputs.get(&output_sig.name) {
                Some(OutputSpec::Provided(_)) => {
                    // あとで所有権を移動
                }
                Some(OutputSpec::AutoCreate {
                    shape,
                    element_size,
                }) => {
                    // バッファを自動生成
                    let buffer = compiler.create_buffer(shape.clone(), *element_size);
                    outputs.insert(output_sig.name.clone(), buffer);
                }
                None => {
                    return Err(format!("Missing output buffer: {}", output_sig.name));
                }
            }
        }

        // Providedバッファを移動
        for (name, spec) in self.outputs {
            if let OutputSpec::Provided(buffer) = spec {
                outputs.insert(name, buffer);
            }
        }

        // shape変数の検証
        for var_name in &signature.shape_vars {
            if !self.shape_vars.contains_key(var_name) {
                return Err(format!("Missing shape variable: {}", var_name));
            }
        }

        Ok(Query {
            inputs: self.inputs,
            outputs,
            shape_vars: self.shape_vars,
        })
    }

    /// Queryを構築して即座に実行
    ///
    /// 注意: この関数は現在プレースホルダーです。
    /// Kernelトレイトにexecute()メソッドを追加する必要があります。
    pub fn execute(self) -> Result<(), String> {
        let _query = self.build()?;
        // TODO: kernel.execute(query)を実装
        Err("execute() is not yet implemented".to_string())
    }
}

/// カーネルのシグネチャ（入出力バッファの形状情報）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSignature {
    pub inputs: Vec<BufferSignature>,
    pub outputs: Vec<BufferSignature>,
    pub shape_vars: Vec<String>, // 動的なshape変数の名前リスト
}

impl KernelSignature {
    /// 新しいKernelSignatureを作成
    pub fn new(
        inputs: Vec<BufferSignature>,
        outputs: Vec<BufferSignature>,
        shape_vars: Vec<String>,
    ) -> Self {
        Self {
            inputs,
            outputs,
            shape_vars,
        }
    }

    /// 空のKernelSignatureを作成
    pub fn empty() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            shape_vars: Vec::new(),
        }
    }
}

/// バッファのシグネチャ（名前と形状）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferSignature {
    pub name: String,
    pub shape: Vec<crate::graph::shape::Expr>, // Exprで動的な形状を表現
}

impl BufferSignature {
    /// 新しいBufferSignatureを作成
    pub fn new(name: String, shape: Vec<crate::graph::shape::Expr>) -> Self {
        Self { name, shape }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // テスト用のダミー実装
    #[derive(Debug, Clone)]
    struct DummyBuffer {
        shape: Vec<usize>,
    }

    impl Buffer for DummyBuffer {
        fn shape(&self) -> Vec<usize> {
            self.shape.clone()
        }
    }

    struct DummyKernel {
        signature: KernelSignature,
    }

    impl Kernel for DummyKernel {
        type Buffer = DummyBuffer;

        fn signature(&self) -> KernelSignature {
            self.signature.clone()
        }
    }

    struct DummyCompiler;

    impl Compiler for DummyCompiler {
        type CodeRepr = String;
        type Buffer = DummyBuffer;
        type Kernel = DummyKernel;
        type Option = ();

        fn new() -> Self {
            Self
        }

        fn is_available(&self) -> bool {
            true
        }

        fn compile(&mut self, _code: &Self::CodeRepr) -> Self::Kernel {
            DummyKernel {
                signature: KernelSignature::empty(),
            }
        }

        fn create_buffer(&self, shape: Vec<usize>, _element_size: usize) -> Self::Buffer {
            DummyBuffer { shape }
        }
    }

    #[test]
    fn test_query_builder_basic() {
        use crate::graph::shape::Expr;

        let signature = KernelSignature::new(
            vec![BufferSignature::new(
                "input".to_string(),
                vec![Expr::from(10)],
            )],
            vec![BufferSignature::new(
                "output".to_string(),
                vec![Expr::from(10)],
            )],
            vec![],
        );

        let kernel = DummyKernel { signature };
        let input_buffer = DummyBuffer { shape: vec![10] };
        let output_buffer = DummyBuffer { shape: vec![10] };

        let query = kernel
            .query()
            .input("input", &input_buffer)
            .output("output", output_buffer)
            .build();

        assert!(query.is_ok());
        let query = query.unwrap();
        assert_eq!(query.inputs.len(), 1);
        assert_eq!(query.outputs.len(), 1);
    }

    #[test]
    fn test_query_builder_missing_input() {
        use crate::graph::shape::Expr;

        let signature = KernelSignature::new(
            vec![BufferSignature::new(
                "input".to_string(),
                vec![Expr::from(10)],
            )],
            vec![BufferSignature::new(
                "output".to_string(),
                vec![Expr::from(10)],
            )],
            vec![],
        );

        let kernel = DummyKernel { signature };
        let output_buffer = DummyBuffer { shape: vec![10] };

        let query = kernel.query().output("output", output_buffer).build();

        assert!(query.is_err());
        assert!(query.unwrap_err().contains("Missing input buffer"));
    }

    #[test]
    fn test_query_builder_auto_output() {
        use crate::graph::shape::Expr;

        let signature = KernelSignature::new(
            vec![BufferSignature::new(
                "input".to_string(),
                vec![Expr::from(10)],
            )],
            vec![BufferSignature::new(
                "output".to_string(),
                vec![Expr::from(10)],
            )],
            vec![],
        );

        let kernel = DummyKernel { signature };
        let input_buffer = DummyBuffer { shape: vec![10] };
        let compiler = DummyCompiler;

        let query = kernel
            .query()
            .input("input", &input_buffer)
            .auto_output("output", vec![10], 4)
            .finalize(&compiler);

        assert!(query.is_ok());
        let query = query.unwrap();
        assert_eq!(query.inputs.len(), 1);
        assert_eq!(query.outputs.len(), 1);
        assert_eq!(query.outputs.get("output").unwrap().shape(), vec![10]);
    }

    #[test]
    fn test_query_builder_shape_vars() {
        use crate::graph::shape::Expr;

        let signature = KernelSignature::new(
            vec![BufferSignature::new(
                "input".to_string(),
                vec![Expr::Var("N".to_string())],
            )],
            vec![BufferSignature::new(
                "output".to_string(),
                vec![Expr::Var("N".to_string())],
            )],
            vec!["N".to_string()],
        );

        let kernel = DummyKernel { signature };
        let input_buffer = DummyBuffer { shape: vec![10] };
        let output_buffer = DummyBuffer { shape: vec![10] };

        // shape_varを設定しない場合はエラー
        let query = kernel
            .query()
            .input("input", &input_buffer)
            .output("output", output_buffer.clone())
            .build();

        assert!(query.is_err());
        assert!(query.unwrap_err().contains("Missing shape variable: N"));

        // shape_varを設定する場合は成功
        let query = kernel
            .query()
            .input("input", &input_buffer)
            .output("output", output_buffer)
            .shape_var("N", 10)
            .build();

        assert!(query.is_ok());
    }
}
