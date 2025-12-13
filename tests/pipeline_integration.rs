/// Pipeline機能の結合テスト
///
/// GenericPipelineを使って、Graph → Program → Code → Kernel の
/// 一連のフローが正しく動作することを確認します。
use harp::backend::{
    Buffer, Compiler, GenericPipeline, Kernel, KernelSignature, Pipeline, Renderer,
};
use harp::graph::{DType, Graph};

// テスト用のダミー実装
#[derive(Debug, Clone)]
struct TestBuffer {
    shape: Vec<usize>,
    #[allow(dead_code)]
    element_size: usize,
}

impl Buffer for TestBuffer {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn dtype(&self) -> harp::ast::DType {
        harp::ast::DType::F32
    }

    fn to_bytes(&self) -> Vec<u8> {
        vec![0; self.shape.iter().product::<usize>() * self.element_size]
    }

    fn from_bytes(&mut self, _bytes: &[u8]) -> Result<(), String> {
        Ok(())
    }

    fn byte_len(&self) -> usize {
        self.shape.iter().product::<usize>() * self.element_size
    }
}

#[derive(Debug, Clone)]
struct TestKernel {
    signature: KernelSignature,
    code: String,
}

impl Kernel for TestKernel {
    type Buffer = TestBuffer;

    fn signature(&self) -> KernelSignature {
        self.signature.clone()
    }

    unsafe fn execute(&self, _buffers: &mut [&mut Self::Buffer]) -> Result<(), String> {
        // テスト用のダミー実装
        Ok(())
    }
}

#[derive(Clone)]
struct TestRenderer;

impl Renderer for TestRenderer {
    type CodeRepr = String;
    type Option = ();

    fn render(&self, program: &harp::ast::AstNode) -> Self::CodeRepr {
        // シンプルなコード生成：エントリーポイントと関数数を文字列化
        match program {
            harp::ast::AstNode::Program {
                functions,
                entry_point,
            } => {
                format!("entry: {}, functions: {}", entry_point, functions.len())
            }
            _ => "not a program".to_string(),
        }
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[derive(Clone)]
struct TestCompiler {
    compile_count: std::cell::RefCell<usize>,
}

impl TestCompiler {
    fn new() -> Self {
        Self {
            compile_count: std::cell::RefCell::new(0),
        }
    }

    fn get_compile_count(&self) -> usize {
        *self.compile_count.borrow()
    }
}

impl Compiler for TestCompiler {
    type CodeRepr = String;
    type Buffer = TestBuffer;
    type Kernel = TestKernel;
    type Option = ();

    fn new() -> Self {
        TestCompiler::new()
    }

    fn is_available(&self) -> bool {
        true
    }

    fn compile(&mut self, code: &Self::CodeRepr, signature: KernelSignature) -> Self::Kernel {
        *self.compile_count.borrow_mut() += 1;
        TestKernel {
            signature,
            code: code.clone(),
        }
    }

    fn create_buffer(&self, shape: Vec<usize>, element_size: usize) -> Self::Buffer {
        TestBuffer {
            shape,
            element_size,
        }
    }
}

/// 単純なグラフを作成するヘルパー関数
fn create_simple_graph() -> Graph {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![10, 20]);
    let c = a + b; // 演算子オーバーロードを使用
    graph.output("out", c);
    graph
}

/// より複雑なグラフを作成するヘルパー関数
fn create_complex_graph() -> Graph {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![10, 20]);
    let c = a.clone() + b.clone(); // 演算子オーバーロードを使用
    let d = c * a;
    let e = d / b;
    graph.output("out", e);
    graph
}

#[test]
fn test_pipeline_basic_flow() {
    // Pipeline作成
    let renderer = TestRenderer;
    let compiler = TestCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);

    // グラフをコンパイル
    let graph = create_simple_graph();
    let result = pipeline.compile_graph(graph);

    assert!(result.is_ok());
    let kernel = result.unwrap();
    assert!(!kernel.code.is_empty());
    assert_eq!(pipeline.compiler().get_compile_count(), 1);
}

#[test]
fn test_pipeline_caching() {
    let renderer = TestRenderer;
    let compiler = TestCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);

    // 同じグラフを2回コンパイル（異なるキーでキャッシュ）
    let graph1 = create_simple_graph();
    let graph2 = create_simple_graph();

    pipeline
        .compile_and_cache("graph1".to_string(), graph1)
        .unwrap();
    pipeline
        .compile_and_cache("graph2".to_string(), graph2)
        .unwrap();

    // 両方キャッシュされていることを確認
    assert_eq!(pipeline.cache_size(), 2);
    assert!(pipeline.get_cached_kernel("graph1").is_some());
    assert!(pipeline.get_cached_kernel("graph2").is_some());

    // コンパイルが2回行われたことを確認
    assert_eq!(pipeline.compiler().get_compile_count(), 2);
}

#[test]
fn test_pipeline_cache_reuse() {
    let renderer = TestRenderer;
    let compiler = TestCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);

    // 最初のコンパイル
    let graph = create_simple_graph();
    pipeline
        .compile_and_cache("test_key".to_string(), graph)
        .unwrap();

    let initial_count = pipeline.compiler().get_compile_count();
    assert_eq!(initial_count, 1);

    // キャッシュから取得
    let cached_kernel = pipeline.get_cached_kernel("test_key");
    assert!(cached_kernel.is_some());

    // コンパイル回数が増えていないことを確認（キャッシュが使われた）
    assert_eq!(pipeline.compiler().get_compile_count(), initial_count);
}

#[test]
fn test_pipeline_cache_overwrite() {
    let renderer = TestRenderer;
    let compiler = TestCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);

    // 同じキーで異なるグラフをコンパイル
    let graph1 = create_simple_graph();
    let graph2 = create_complex_graph();

    pipeline
        .compile_and_cache("same_key".to_string(), graph1)
        .unwrap();
    pipeline
        .compile_and_cache("same_key".to_string(), graph2)
        .unwrap();

    // キャッシュサイズは1のまま（上書きされた）
    assert_eq!(pipeline.cache_size(), 1);

    // コンパイルは2回行われた
    assert_eq!(pipeline.compiler().get_compile_count(), 2);
}

#[test]
fn test_pipeline_cache_removal() {
    let renderer = TestRenderer;
    let compiler = TestCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);

    // 複数のグラフをキャッシュ
    for i in 0..3 {
        let graph = create_simple_graph();
        pipeline
            .compile_and_cache(format!("graph_{}", i), graph)
            .unwrap();
    }

    assert_eq!(pipeline.cache_size(), 3);

    // 1つ削除
    let removed = pipeline.remove_cached_kernel("graph_1");
    assert!(removed.is_some());
    assert_eq!(pipeline.cache_size(), 2);

    // 削除したキーは存在しない
    assert!(pipeline.get_cached_kernel("graph_1").is_none());

    // 他のキーは存在する
    assert!(pipeline.get_cached_kernel("graph_0").is_some());
    assert!(pipeline.get_cached_kernel("graph_2").is_some());
}

#[test]
fn test_pipeline_clear_cache() {
    let renderer = TestRenderer;
    let compiler = TestCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);

    // 複数のグラフをキャッシュ
    for i in 0..5 {
        let graph = create_simple_graph();
        pipeline
            .compile_and_cache(format!("graph_{}", i), graph)
            .unwrap();
    }

    assert_eq!(pipeline.cache_size(), 5);

    // 全てクリア
    pipeline.clear_cache();
    assert_eq!(pipeline.cache_size(), 0);

    // 全てのキーが存在しない
    for i in 0..5 {
        assert!(
            pipeline
                .get_cached_kernel(&format!("graph_{}", i))
                .is_none()
        );
    }
}

#[test]
fn test_pipeline_with_complex_graph() {
    let renderer = TestRenderer;
    let compiler = TestCompiler::new();
    let mut pipeline = GenericPipeline::new(renderer, compiler);

    // 複雑なグラフでもコンパイルできることを確認
    let graph = create_complex_graph();
    let result = pipeline.compile_graph(graph);

    assert!(result.is_ok());
    let kernel = result.unwrap();
    assert!(!kernel.code.is_empty());
}

// Metal実装を使った結合テスト（macOS限定）
#[cfg(target_os = "macos")]
mod metal_tests {
    use super::*;
    use harp::backend::{MetalCompiler, MetalRenderer};

    #[test]
    #[ignore] // Metalの実装が完全ではないため一旦スキップ
    fn test_metal_pipeline_basic() {
        let renderer = MetalRenderer::new();
        let compiler = MetalCompiler::new();

        // Metal実装が利用可能かチェック
        if !renderer.is_available() || !compiler.is_available() {
            eprintln!("Metal is not available, skipping test");
            return;
        }

        let mut pipeline = GenericPipeline::new(renderer, compiler);

        // シンプルなグラフを作成してコンパイル
        let graph = create_simple_graph();
        let result = pipeline.compile_graph(graph);

        // Metalが利用可能な環境ではコンパイルが成功するはず
        assert!(result.is_ok());
    }

    #[test]
    #[ignore] // Metalの実装が完全ではないため一旦スキップ
    fn test_metal_pipeline_caching() {
        let renderer = MetalRenderer::new();
        let compiler = MetalCompiler::new();

        if !renderer.is_available() || !compiler.is_available() {
            eprintln!("Metal is not available, skipping test");
            return;
        }

        let mut pipeline = GenericPipeline::new(renderer, compiler);

        // 異なるグラフをキャッシュ
        let graph1 = create_simple_graph();
        let graph2 = create_complex_graph();

        // 借用チェックエラーを回避するため、結果を使わない場合は変数に束縛しない
        pipeline
            .compile_and_cache("simple".to_string(), graph1)
            .unwrap();
        pipeline
            .compile_and_cache("complex".to_string(), graph2)
            .unwrap();

        assert_eq!(pipeline.cache_size(), 2);
    }
}
