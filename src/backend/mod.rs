use std::collections::HashMap;

pub mod c;
pub mod c_like;
pub mod device;
pub mod generic;
pub mod metal;
pub mod opencl;
pub mod pipeline;

// Re-export commonly used types
pub use c::{CBuffer, CCode, CCompiler, CKernel, CPipeline, CRenderer};
pub use device::{Device, DeviceBuffer, DeviceKernel, DevicePipeline, SharedPipeline};
pub use generic::{GenericPipeline, OptimizationConfig, OptimizationHistories};
pub use metal::{MetalCode, MetalRenderer};
pub use opencl::{
    OpenCLBuffer, OpenCLCode, OpenCLCompiler, OpenCLKernel, OpenCLPipeline, OpenCLRenderer,
};

#[cfg(target_os = "macos")]
pub use metal::{MetalBuffer, MetalCompiler, MetalKernel, MetalPipeline};

// レンダラー。
// Programを受け取って文字列としてレンダリングする
pub trait Renderer {
    type CodeRepr: Into<String>;
    type Option;
    fn render(&self, program: &crate::ast::AstNode) -> Self::CodeRepr;
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
    fn compile(&mut self, code: &Self::CodeRepr, signature: KernelSignature) -> Self::Kernel;
    fn create_buffer(&self, shape: Vec<usize>, element_size: usize) -> Self::Buffer;
}
pub trait Buffer {
    /// バッファの形状を取得
    fn shape(&self) -> Vec<usize>;

    /// バッファの要素の型を取得
    fn dtype(&self) -> crate::ast::DType;

    /// バッファの内容をバイト列として取得
    fn to_bytes(&self) -> Vec<u8>;

    /// バイト列からバッファに書き込み
    #[allow(clippy::wrong_self_convention)]
    fn from_bytes(&mut self, bytes: &[u8]) -> Result<(), String>;

    /// バッファの総バイト数を取得
    fn byte_len(&self) -> usize;

    /// バッファの内容を型付きベクタとして取得（デフォルト実装）
    ///
    /// # Safety
    /// Tのサイズがdtypeの要素サイズと一致している必要があります
    fn to_vec<T: Clone + 'static>(&self) -> Result<Vec<T>, String> {
        let bytes = self.to_bytes();
        let type_size = std::mem::size_of::<T>();

        if !bytes.len().is_multiple_of(type_size) {
            return Err(format!(
                "Buffer size {} is not a multiple of type size {}",
                bytes.len(),
                type_size
            ));
        }

        let len = bytes.len() / type_size;
        let mut result = Vec::with_capacity(len);

        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const T, result.as_mut_ptr(), len);
            result.set_len(len);
        }

        Ok(result)
    }

    /// 型付きスライスからバッファに書き込み（デフォルト実装）
    #[allow(clippy::wrong_self_convention)]
    #[allow(clippy::manual_slice_size_calculation)]
    fn from_vec<T>(&mut self, data: &[T]) -> Result<(), String> {
        let type_size = std::mem::size_of::<T>();
        let byte_len = data.len() * type_size;

        if byte_len != self.byte_len() {
            return Err(format!(
                "Data size {} does not match buffer size {}",
                byte_len,
                self.byte_len()
            ));
        }

        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };

        self.from_bytes(bytes)
    }

    /// バッファの内容をndarrayとして取得（デフォルト実装）
    fn to_ndarray<T: Clone + 'static>(&self) -> Result<ndarray::ArrayD<T>, String> {
        let shape = self.shape();
        let vec_data = self.to_vec::<T>()?;

        // shapeをndarray用に変換（Vec<usize> -> IxDyn）
        let ndarray_shape = ndarray::IxDyn(&shape);

        // ArrayDを作成
        ndarray::ArrayD::from_shape_vec(ndarray_shape, vec_data)
            .map_err(|e| format!("Failed to create ndarray: {}", e))
    }

    /// ndarrayからバッファに書き込み（デフォルト実装）
    #[allow(clippy::wrong_self_convention)]
    fn from_ndarray<T>(&mut self, array: &ndarray::ArrayD<T>) -> Result<(), String>
    where
        T: Clone,
    {
        // shapeのチェック
        let buffer_shape = self.shape();
        let array_shape: Vec<usize> = array.shape().to_vec();

        if buffer_shape != array_shape {
            return Err(format!(
                "Shape mismatch: buffer has shape {:?}, but array has shape {:?}",
                buffer_shape, array_shape
            ));
        }

        // as_slice()で直接スライスが取得できる場合はそれを使用
        if let Some(slice) = array.as_slice() {
            return self.from_vec(slice);
        }

        // 連続していない場合は、標準レイアウトにコピーしてから書き込み
        let owned = array.as_standard_layout();
        let slice = owned
            .as_slice()
            .ok_or_else(|| "Failed to convert array to contiguous slice".to_string())?;

        self.from_vec(slice)
    }
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
    type Error;

    // 各コンポーネントへのアクセス
    fn renderer(&self) -> &Self::Renderer;
    fn compiler(&mut self) -> &mut Self::Compiler;

    // Graph最適化
    fn optimize_graph(&self, graph: crate::graph::Graph) -> crate::graph::Graph {
        // デフォルトは最適化なし
        graph
    }

    // Lowering (Graph → Program)
    fn lower_to_program(&self, graph: crate::graph::Graph) -> crate::ast::AstNode {
        crate::lowerer::lower(graph)
    }

    // Program最適化
    fn optimize_program(&self, program: crate::ast::AstNode) -> crate::ast::AstNode {
        // デフォルトは最適化なし
        program
    }

    // Graph → Kernel の一連の流れ
    fn compile_graph(
        &mut self,
        graph: crate::graph::Graph,
    ) -> Result<<Self::Compiler as Compiler>::Kernel, Self::Error> {
        // デフォルト実装: 各ステージを順番に実行
        let signature = crate::lowerer::create_signature(&graph);
        let optimized_graph = self.optimize_graph(graph);
        let program = self.lower_to_program(optimized_graph);
        let optimized_program = self.optimize_program(program);
        let code = self.renderer().render(&optimized_program);
        Ok(self.compiler().compile(&code, signature))
    }
}

// カーネルへの指示をまとめる構造体
#[derive(Debug)]
pub struct Query<'a, B: Buffer> {
    pub inputs: HashMap<String, &'a B>, // inputsは読み取り専用なので借用
    pub outputs: HashMap<String, B>,    // outputsは書き込み対象
    pub shape_vars: HashMap<String, isize>, // 動的shape変数の値
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
    shape_vars: HashMap<String, isize>,
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
    pub fn shape_var(mut self, name: impl Into<String>, value: isize) -> Self {
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
        for var_name in signature.shape_vars.keys() {
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
        for var_name in signature.shape_vars.keys() {
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
    pub shape_vars: HashMap<String, isize>, // 動的shape変数の名前とデフォルト値
}

impl KernelSignature {
    /// 新しいKernelSignatureを作成
    pub fn new(
        inputs: Vec<BufferSignature>,
        outputs: Vec<BufferSignature>,
        shape_vars: HashMap<String, isize>,
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
            shape_vars: HashMap::new(),
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

        fn dtype(&self) -> crate::ast::DType {
            crate::ast::DType::F32
        }

        fn to_bytes(&self) -> Vec<u8> {
            vec![]
        }

        fn from_bytes(&mut self, _bytes: &[u8]) -> Result<(), String> {
            Ok(())
        }

        fn byte_len(&self) -> usize {
            0
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

        fn compile(&mut self, _code: &Self::CodeRepr, signature: KernelSignature) -> Self::Kernel {
            DummyKernel { signature }
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
            HashMap::new(),
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
            HashMap::new(),
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
            HashMap::new(),
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

        let mut shape_vars = HashMap::new();
        shape_vars.insert("N".to_string(), 10);

        let signature = KernelSignature::new(
            vec![BufferSignature::new(
                "input".to_string(),
                vec![Expr::Var("N".to_string())],
            )],
            vec![BufferSignature::new(
                "output".to_string(),
                vec![Expr::Var("N".to_string())],
            )],
            shape_vars,
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
