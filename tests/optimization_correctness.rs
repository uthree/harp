/// 最適化の正確性を検証する結合テスト
///
/// 複雑な計算グラフに対して最適化・コード生成が正しく動作することを確認します。
#[cfg(test)]
mod tests {
    use harp::backend::GenericPipeline;
    use harp::backend::opencl::{OpenCLCompiler, OpenCLRenderer};
    use harp::graph::{DType, Graph};

    /// テストケース構造体
    struct TestCase {
        name: &'static str,
        graph: Graph,
    }

    /// シンプルな加算テスト
    fn create_add_test() -> TestCase {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![4, 4]);
        let b = graph.input("b", DType::F32, vec![4, 4]);
        graph.output("result", a + b);

        TestCase { name: "add", graph }
    }

    /// 連鎖演算テスト
    fn create_chain_test() -> TestCase {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![8, 8]);
        let b = graph.input("b", DType::F32, vec![8, 8]);
        let c = graph.input("c", DType::F32, vec![8, 8]);
        graph.output("result", (a + b) * c);

        TestCase {
            name: "chain",
            graph,
        }
    }

    /// 複雑なグラフテスト
    fn create_complex_test() -> TestCase {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![6, 6]);
        let b = graph.input("b", DType::F32, vec![6, 6]);
        let c = graph.input("c", DType::F32, vec![6, 6]);
        let d = graph.input("d", DType::F32, vec![6, 6]);
        graph.output("result", ((a + b) * c) + d);

        TestCase {
            name: "complex",
            graph,
        }
    }

    /// Reduceテスト
    fn create_reduce_test() -> TestCase {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![4, 8]);
        let b = graph.input("b", DType::F32, vec![4, 8]);
        graph.output("result", (a + b).reduce_sum(0));

        TestCase {
            name: "reduce",
            graph,
        }
    }

    /// 行列積パターンテスト
    fn create_matmul_like_test() -> TestCase {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![8, 16]);
        let b = graph.input("b", DType::F32, vec![8, 16]);
        graph.output("result", (a * b).reduce_sum(1));

        TestCase {
            name: "matmul_like",
            graph,
        }
    }

    /// コード生成が正しく行われることを確認
    fn test_code_generation(test_case: TestCase) {
        let renderer = OpenCLRenderer::new();
        let compiler = OpenCLCompiler::new();
        let mut pipeline = GenericPipeline::new(renderer, compiler);

        let (program, _) = pipeline
            .optimize_graph_with_all_histories(test_case.graph)
            .expect(&format!("Failed to optimize graph: {}", test_case.name));

        let mut opencl_renderer = OpenCLRenderer::new();
        let code = opencl_renderer.render_program(&program);
        let code_str = code.to_string();

        // 生成されたコードが空でないことを確認
        assert!(
            !code_str.is_empty(),
            "Generated code should not be empty for test: {}",
            test_case.name
        );

        // カーネル関数が生成されていることを確認
        assert!(
            code_str.contains("__kernel") || code_str.contains("void"),
            "Code should contain kernel functions for test: {}",
            test_case.name
        );

        println!("✓ {} test passed", test_case.name);
    }

    #[test]
    fn test_add_code_generation() {
        test_code_generation(create_add_test());
    }

    #[test]
    fn test_chain_code_generation() {
        test_code_generation(create_chain_test());
    }

    #[test]
    fn test_complex_code_generation() {
        test_code_generation(create_complex_test());
    }

    #[test]
    fn test_reduce_code_generation() {
        test_code_generation(create_reduce_test());
    }

    #[test]
    fn test_matmul_like_code_generation() {
        test_code_generation(create_matmul_like_test());
    }
}
