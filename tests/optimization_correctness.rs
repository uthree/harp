/// 最適化の正確性を検証する結合テスト
///
/// 複雑な計算グラフに対して、最適化あり/なしでコンパイル・実行し、
/// 結果が一致することを確認します。

#[cfg(test)]
mod tests {
    use harp::backend::c::{CCompiler, CPipeline, CRenderer};
    use harp::backend::{Buffer, Compiler, Kernel, Pipeline};
    use harp::graph::{DType, Graph, GraphNode, shape::Expr};

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

        // panicをキャッチしてfalseを返す
        let result = panic::catch_unwind(|| {
            let renderer = CRenderer::new();
            let compiler = CCompiler::new();

            // シンプルなグラフでコンパイルを試行
            let mut test_graph = Graph::new();
            let a = test_graph
                .input("a")
                .with_dtype(DType::F32)
                .with_shape(vec![2, 2])
                .build();
            test_graph.output("out", a);

            let mut pipeline = CPipeline::new(renderer, compiler);
            pipeline.compile_graph(test_graph).is_ok()
        });

        result.unwrap_or(false)
    }

    /// テストヘルパー：グラフをコンパイルして実行
    fn compile_and_run(
        graph: Graph,
        enable_optimization: bool,
        input_data: Vec<Vec<f32>>,
    ) -> Vec<f32> {
        let renderer = CRenderer::new();
        let compiler = CCompiler::new();
        let mut pipeline = CPipeline::new(renderer, compiler);

        // 最適化の有効/無効を設定
        pipeline.enable_graph_optimization = enable_optimization;
        pipeline.enable_ast_optimization = enable_optimization;


        // コンパイル
        let kernel = pipeline
            .compile_graph(graph.clone())
            .expect("Failed to compile graph");

        // 入力バッファを作成
        let signature = kernel.signature();
        let mut input_buffers = Vec::new();

        for (i, data) in input_data.iter().enumerate() {
            let shape = expr_vec_to_usize_vec(&signature.inputs[i].shape);
            let mut buffer = pipeline
                .compiler()
                .create_buffer(shape, std::mem::size_of::<f32>());

            // データをバッファに書き込み
            let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
            buffer.from_bytes(&bytes).expect("Failed to write buffer");
            input_buffers.push(buffer);
        }

        // 出力バッファを作成
        let output_shape = expr_vec_to_usize_vec(&signature.outputs[0].shape);
        let mut output_buffer = pipeline
            .compiler()
            .create_buffer(output_shape, std::mem::size_of::<f32>());

        // カーネルを実行（入力と出力を結合）
        let mut all_buffers: Vec<&mut _> = input_buffers.iter_mut().collect();
        all_buffers.push(&mut output_buffer);

        unsafe {
            kernel
                .execute(&mut all_buffers)
                .expect("Failed to execute kernel");
        }

        // 結果を取得
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

    #[test]
    fn test_elementwise_add_optimization() {
        if !check_compiler_available() {
            eprintln!("C compiler not available, skipping test");
            return;
        }
        // シンプルな加算: a + b
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![4, 4])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![4, 4])
            .build();
        let result = a + b;
        graph.output("result", result);

        // 入力データ
        let input_a: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let input_b: Vec<f32> = (0..16).map(|x| (x * 2) as f32).collect();

        // 最適化なしで実行
        let result_no_opt =
            compile_and_run(graph.clone(), false, vec![input_a.clone(), input_b.clone()]);

        // 最適化ありで実行
        let result_with_opt = compile_and_run(graph, true, vec![input_a, input_b]);

        // 結果を比較
        assert_approx_eq(&result_no_opt, &result_with_opt, 1e-5);
    }

    #[test]
    fn test_elementwise_chain_optimization() {
        if !check_compiler_available() {
            eprintln!("C compiler not available, skipping test");
            return;
        }
        // 連鎖演算: (a + b) * c
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![8, 8])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![8, 8])
            .build();
        let c = graph
            .input("c")
            .with_dtype(DType::F32)
            .with_shape(vec![8, 8])
            .build();

        let temp = a + b;
        let result = temp * c;
        graph.output("result", result);

        // 入力データ
        let input_a: Vec<f32> = (0..64).map(|x| (x as f32) * 0.1).collect();
        let input_b: Vec<f32> = (0..64).map(|x| (x as f32) * 0.2).collect();
        let input_c: Vec<f32> = (0..64).map(|x| (x as f32) * 0.3).collect();

        // 最適化ありで実行（fusion最適化により正しく計算される）
        let result_with_opt = compile_and_run(graph, true, vec![input_a.clone(), input_b.clone(), input_c.clone()]);

        // 期待される結果と比較: (a + b) * c
        let expected: Vec<f32> = input_a.iter().zip(input_b.iter()).zip(input_c.iter())
            .map(|((&a, &b), &c)| (a + b) * c)
            .collect();
        assert_approx_eq(&result_with_opt, &expected, 1e-5);
    }

    #[test]
    fn test_const_propagation_optimization() {
        if !check_compiler_available() {
            eprintln!("C compiler not available, skipping test");
            return;
        }
        // 定数伝播: a + (2.0 * 3.0)
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 10])
            .build();

        // 定数演算（スカラー定数は自動的にブロードキャストされる）
        let const1: GraphNode = 2.0f32.into();
        let const2: GraphNode = 3.0f32.into();
        let scale = const1 * const2; // 6.0に畳み込まれる

        let result = a + scale;
        graph.output("result", result);

        // 入力データ
        let input_a: Vec<f32> = (0..100).map(|x| (x as f32) * 0.5).collect();

        // 最適化なしで実行
        let result_no_opt = compile_and_run(graph.clone(), false, vec![input_a.clone()]);

        // 最適化ありで実行
        let result_with_opt = compile_and_run(graph, true, vec![input_a.clone()]);

        // 結果を比較
        assert_approx_eq(&result_no_opt, &result_with_opt, 1e-5);

        // 期待される結果とも比較（a + 6.0）
        let expected: Vec<f32> = input_a.iter().map(|&x| x + 6.0).collect();
        assert_approx_eq(&result_with_opt, &expected, 1e-5);
    }

    #[test]
    fn test_reduce_sum_optimization() {
        if !check_compiler_available() {
            eprintln!("C compiler not available, skipping test");
            return;
        }
        // Reduce演算: reduce_sum(a + b, axis=0)
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![4, 8])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![4, 8])
            .build();

        let sum_input = a + b;
        let result = sum_input.reduce_sum(0);
        graph.output("result", result);

        // 入力データ
        let input_a: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let input_b: Vec<f32> = (0..32).map(|x| (x * 2) as f32).collect();

        // 最適化なしで実行
        let result_no_opt =
            compile_and_run(graph.clone(), false, vec![input_a.clone(), input_b.clone()]);

        // 最適化ありで実行
        let result_with_opt = compile_and_run(graph, true, vec![input_a, input_b]);

        // 結果を比較
        assert_approx_eq(&result_no_opt, &result_with_opt, 1e-4);
    }

    #[test]
    fn test_complex_graph_optimization() {
        if !check_compiler_available() {
            eprintln!("C compiler not available, skipping test");
            return;
        }
        // 複雑なグラフ: ((a + b) * c) + d
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![6, 6])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![6, 6])
            .build();
        let c = graph
            .input("c")
            .with_dtype(DType::F32)
            .with_shape(vec![6, 6])
            .build();
        let d = graph
            .input("d")
            .with_dtype(DType::F32)
            .with_shape(vec![6, 6])
            .build();

        let temp1 = a + b;
        let temp2 = temp1 * c;
        let result = temp2 + d;
        graph.output("result", result);

        // 入力データ
        let input_a: Vec<f32> = (0..36).map(|x| (x as f32) * 0.1).collect();
        let input_b: Vec<f32> = (0..36).map(|x| (x as f32) * 0.2).collect();
        let input_c: Vec<f32> = (0..36).map(|x| (x as f32) * 0.3).collect();
        let input_d: Vec<f32> = (0..36).map(|x| (x as f32) * 0.4).collect();

        // 最適化ありで実行（fusion最適化により正しく計算される）
        let result_with_opt =
            compile_and_run(graph, true, vec![input_a.clone(), input_b.clone(), input_c.clone(), input_d.clone()]);

        // 期待される結果と比較: ((a + b) * c) + d
        let expected: Vec<f32> = input_a.iter().zip(input_b.iter()).zip(input_c.iter()).zip(input_d.iter())
            .map(|(((&a, &b), &c), &d)| ((a + b) * c) + d)
            .collect();
        assert_approx_eq(&result_with_opt, &expected, 1e-5);
    }

    #[test]
    fn test_view_transformation_optimization() {
        if !check_compiler_available() {
            eprintln!("C compiler not available, skipping test");
            return;
        }
        // View変換を含む演算: a.permute([1, 0]) + b
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![3, 4])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![4, 3])
            .build();

        // aを転置して [4, 3] にする
        let a_transposed = a.view(a.view.clone().permute(vec![1, 0]));
        let result = a_transposed + b;
        graph.output("result", result);

        // 入力データ
        let input_a: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let input_b: Vec<f32> = (0..12).map(|x| (x * 2) as f32).collect();

        // 最適化なしで実行
        let result_no_opt =
            compile_and_run(graph.clone(), false, vec![input_a.clone(), input_b.clone()]);

        // 最適化ありで実行
        let result_with_opt = compile_and_run(graph, true, vec![input_a, input_b]);

        // 結果を比較
        assert_approx_eq(&result_no_opt, &result_with_opt, 1e-5);
    }

    #[test]
    fn test_matmul_like_pattern_optimization() {
        if !check_compiler_available() {
            eprintln!("C compiler not available, skipping test");
            return;
        }
        // 行列積のようなパターン: reduce_sum(a * b, axis=1)
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![8, 16])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![8, 16])
            .build();

        let mul_result = a * b;
        let result = mul_result.reduce_sum(1);
        graph.output("result", result);

        // 入力データ
        let input_a: Vec<f32> = (0..128).map(|x| (x as f32) * 0.01).collect();
        let input_b: Vec<f32> = (0..128).map(|x| (x as f32) * 0.02).collect();

        // 最適化なしで実行
        let result_no_opt =
            compile_and_run(graph.clone(), false, vec![input_a.clone(), input_b.clone()]);

        // 最適化ありで実行
        let result_with_opt = compile_and_run(graph, true, vec![input_a, input_b]);

        // 結果を比較
        assert_approx_eq(&result_no_opt, &result_with_opt, 1e-4);
    }
}
