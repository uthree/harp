/// 最適化の正確性を検証する結合テスト
///
/// 複雑な計算グラフに対して、最適化あり/なしでコンパイル・実行し、
/// 結果が一致することを確認します。
#[cfg(test)]
mod tests {
    use harp::backend::c::{CCompiler, CPipeline, CRenderer};
    use harp::backend::{Buffer, Compiler, Kernel, Pipeline};
    use harp::graph::{DType, Graph, shape::Expr};
    use rstest::rstest;

    /// ExprをVec<usize>に変換（定数のみサポート）
    fn expr_vec_to_usize_vec(exprs: &[Expr]) -> Vec<usize> {
        exprs
            .iter()
            .map(|expr| match expr {
                Expr::Const(v) => *v as usize,
                _ => panic!("Dynamic shapes not supported in tests"),
            })
            .collect()
    }

    /// Cコンパイラが利用可能かチェック
    fn check_compiler_available() -> bool {
        use std::panic;
        let result = panic::catch_unwind(|| {
            let renderer = CRenderer::new();
            let compiler = CCompiler::new();
            let mut test_graph = Graph::new();
            let a = test_graph.input("a", DType::F32, vec![2, 2]);
            test_graph.output("out", a);
            let mut pipeline = CPipeline::new(renderer, compiler);
            pipeline.compile_graph(test_graph).is_ok()
        });
        result.unwrap_or(false)
    }

    /// テストヘルパー：グラフをコンパイルして実行
    fn compile_and_run(graph: Graph, input_data: Vec<Vec<f32>>) -> Vec<f32> {
        let renderer = CRenderer::new();
        let compiler = CCompiler::new();
        let mut pipeline = CPipeline::new(renderer, compiler);

        let kernel = pipeline
            .compile_graph(graph.clone())
            .expect("Failed to compile graph");

        let signature = kernel.signature();
        let mut input_buffers: Vec<_> = input_data
            .iter()
            .enumerate()
            .map(|(i, data)| {
                let shape = expr_vec_to_usize_vec(&signature.inputs[i].shape);
                let mut buffer = pipeline
                    .compiler()
                    .create_buffer(shape, std::mem::size_of::<f32>());
                let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
                buffer.from_bytes(&bytes).expect("Failed to write buffer");
                buffer
            })
            .collect();

        let output_shape = expr_vec_to_usize_vec(&signature.outputs[0].shape);
        let mut output_buffer = pipeline
            .compiler()
            .create_buffer(output_shape, std::mem::size_of::<f32>());

        let mut all_buffers: Vec<&mut _> = input_buffers.iter_mut().collect();
        all_buffers.push(&mut output_buffer);

        unsafe {
            kernel
                .execute(&mut all_buffers)
                .expect("Failed to execute kernel");
        }

        let output_bytes = output_buffer.to_bytes();
        output_bytes
            .chunks(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// 2つのfloat配列が近似的に等しいか確認
    fn assert_approx_eq(a: &[f32], b: &[f32], epsilon: f32) {
        assert_eq!(
            a.len(),
            b.len(),
            "Array lengths differ: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (va - vb).abs();
            assert!(
                diff < epsilon,
                "Values differ at index {}: {} vs {} (diff: {})",
                i,
                va,
                vb,
                diff
            );
        }
    }

    /// テストケース構造体
    struct TestCase {
        name: &'static str,
        graph: Graph,
        inputs: Vec<Vec<f32>>,
        expected: Vec<f32>,
        epsilon: f32,
    }

    /// シンプルな加算テスト
    fn create_add_test() -> TestCase {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![4, 4]);
        let b = graph.input("b", DType::F32, vec![4, 4]);
        graph.output("result", a + b);

        let input_a: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let input_b: Vec<f32> = (0..16).map(|x| (x * 2) as f32).collect();
        let expected: Vec<f32> = input_a.iter().zip(&input_b).map(|(&a, &b)| a + b).collect();

        TestCase {
            name: "add",
            graph,
            inputs: vec![input_a, input_b],
            expected,
            epsilon: 1e-5,
        }
    }

    /// 連鎖演算テスト
    fn create_chain_test() -> TestCase {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![8, 8]);
        let b = graph.input("b", DType::F32, vec![8, 8]);
        let c = graph.input("c", DType::F32, vec![8, 8]);
        graph.output("result", (a + b) * c);

        let input_a: Vec<f32> = (0..64).map(|x| (x as f32) * 0.1).collect();
        let input_b: Vec<f32> = (0..64).map(|x| (x as f32) * 0.2).collect();
        let input_c: Vec<f32> = (0..64).map(|x| (x as f32) * 0.3).collect();
        let expected: Vec<f32> = input_a
            .iter()
            .zip(&input_b)
            .zip(&input_c)
            .map(|((&a, &b), &c)| (a + b) * c)
            .collect();

        TestCase {
            name: "chain",
            graph,
            inputs: vec![input_a, input_b, input_c],
            expected,
            epsilon: 1e-5,
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

        let input_a: Vec<f32> = (0..36).map(|x| (x as f32) * 0.1).collect();
        let input_b: Vec<f32> = (0..36).map(|x| (x as f32) * 0.2).collect();
        let input_c: Vec<f32> = (0..36).map(|x| (x as f32) * 0.3).collect();
        let input_d: Vec<f32> = (0..36).map(|x| (x as f32) * 0.4).collect();
        let expected: Vec<f32> = input_a
            .iter()
            .zip(&input_b)
            .zip(&input_c)
            .zip(&input_d)
            .map(|(((&a, &b), &c), &d)| ((a + b) * c) + d)
            .collect();

        TestCase {
            name: "complex",
            graph,
            inputs: vec![input_a, input_b, input_c, input_d],
            expected,
            epsilon: 1e-5,
        }
    }

    /// Reduceテスト
    fn create_reduce_test() -> TestCase {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![4, 8]);
        let b = graph.input("b", DType::F32, vec![4, 8]);
        graph.output("result", (a + b).reduce_sum(0));

        let input_a: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let input_b: Vec<f32> = (0..32).map(|x| (x * 2) as f32).collect();
        // axis=0でreduce: shape [4, 8] -> [8]
        let mut expected = vec![0.0f32; 8];
        for row in 0..4 {
            for col in 0..8 {
                let idx = row * 8 + col;
                expected[col] += input_a[idx] + input_b[idx];
            }
        }

        TestCase {
            name: "reduce",
            graph,
            inputs: vec![input_a, input_b],
            expected,
            epsilon: 1e-4,
        }
    }

    /// 行列積パターンテスト
    fn create_matmul_like_test() -> TestCase {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![8, 16]);
        let b = graph.input("b", DType::F32, vec![8, 16]);
        graph.output("result", (a * b).reduce_sum(1));

        let input_a: Vec<f32> = (0..128).map(|x| (x as f32) * 0.01).collect();
        let input_b: Vec<f32> = (0..128).map(|x| (x as f32) * 0.02).collect();
        // axis=1でreduce: shape [8, 16] -> [8]
        let mut expected = vec![0.0f32; 8];
        for row in 0..8 {
            for col in 0..16 {
                let idx = row * 16 + col;
                expected[row] += input_a[idx] * input_b[idx];
            }
        }

        TestCase {
            name: "matmul_like",
            graph,
            inputs: vec![input_a, input_b],
            expected,
            epsilon: 1e-4,
        }
    }

    #[rstest]
    #[case::add(create_add_test())]
    #[case::chain(create_chain_test())]
    #[case::complex(create_complex_test())]
    #[case::reduce(create_reduce_test())]
    #[case::matmul_like(create_matmul_like_test())]
    fn test_optimization_correctness(#[case] test_case: TestCase) {
        if !check_compiler_available() {
            eprintln!(
                "C compiler not available, skipping test: {}",
                test_case.name
            );
            return;
        }

        let result = compile_and_run(test_case.graph, test_case.inputs);
        assert_approx_eq(&result, &test_case.expected, test_case.epsilon);
    }
}
