//! 複数の定数配列を組み合わせた計算グラフのテスト
//!
//! lazy-arrayで発見されたバグ（3つ以上の定数配列を組み合わせると誤った結果）を
//! core側で再現・調査するためのテスト。

#[cfg(feature = "opencl")]
mod opencl_tests {
    use harp_core::backend::{Buffer, Compiler, Pipeline};
    use harp_core::graph::shape::Expr;
    use harp_core::graph::{DType, Graph, GraphNode};
    use std::collections::HashMap;

    use harp_backend_opencl::{OpenCLBuffer, OpenCLCompiler, OpenCLDevice, OpenCLRenderer};

    type TestPipeline = Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler>;

    fn create_pipeline() -> TestPipeline {
        let device = OpenCLDevice::new().expect("OpenCL device");
        let renderer = OpenCLRenderer::new();
        let compiler = OpenCLCompiler::new();
        Pipeline::new(renderer, compiler, device)
    }

    /// 定数スカラーをbroadcastして[4,4]配列にする（lazy-arrayと同じ方法）
    fn broadcast_constant(value: f32) -> GraphNode {
        // constant(value) でスカラーを作成
        let mut node = GraphNode::constant(value);

        // 2次元に unsqueeze: [] -> [1] -> [1, 1]
        for _ in 0..2 {
            let new_view = node.view.clone().unsqueeze(0);
            node = node.view(new_view);
        }

        // broadcast_to で [4, 4] に拡張
        let target_shape: Vec<Expr> = vec![Expr::from(4isize), Expr::from(4isize)];
        node = node.broadcast_to(target_shape);

        // contiguous でメモリレイアウトを実体化
        node.contiguous()
    }

    #[test]
    fn test_single_constant() {
        // 単一の定数: full(2.0) -> 全て2.0
        let mut pipeline = create_pipeline();
        let device = pipeline.device().clone();

        let node = broadcast_constant(2.0);

        let mut graph = Graph::new();
        graph.output("result", node);

        println!("=== Single constant test ===");
        println!("DSL:\n{}", harp_dsl::decompile(&graph));

        let compiled = pipeline.compile_program(graph).expect("compile");
        let mut output =
            OpenCLBuffer::allocate(&device, vec![4, 4], harp_core::ast::DType::F32).expect("alloc");

        let inputs: HashMap<String, &OpenCLBuffer> = HashMap::new();
        let mut outputs: HashMap<String, &mut OpenCLBuffer> = HashMap::new();
        outputs.insert("result".to_string(), &mut output);

        compiled
            .execute(&device, &inputs, &mut outputs)
            .expect("execute");

        let data: Vec<f32> = output.read_vec().expect("read");
        println!("Result: {:?}", &data[..4]);

        for (i, v) in data.iter().enumerate() {
            assert!((v - 2.0).abs() < 1e-5, "[{}] Expected 2.0, got {}", i, v);
        }
        println!("PASS: Single constant");
    }

    #[test]
    fn test_two_constants_add() {
        // 2つの定数: 2.0 + 3.0 = 5.0
        let mut pipeline = create_pipeline();
        let device = pipeline.device().clone();

        let a = broadcast_constant(2.0);
        let b = broadcast_constant(3.0);
        let result = a + b;

        let mut graph = Graph::new();
        graph.output("result", result);

        println!("=== Two constants add test ===");
        println!("DSL:\n{}", harp_dsl::decompile(&graph));

        let compiled = pipeline.compile_program(graph).expect("compile");
        let mut output =
            OpenCLBuffer::allocate(&device, vec![4, 4], harp_core::ast::DType::F32).expect("alloc");

        let inputs: HashMap<String, &OpenCLBuffer> = HashMap::new();
        let mut outputs: HashMap<String, &mut OpenCLBuffer> = HashMap::new();
        outputs.insert("result".to_string(), &mut output);

        compiled
            .execute(&device, &inputs, &mut outputs)
            .expect("execute");

        let data: Vec<f32> = output.read_vec().expect("read");
        println!("Result: {:?}", &data[..4]);

        for (i, v) in data.iter().enumerate() {
            assert!((v - 5.0).abs() < 1e-5, "[{}] Expected 5.0, got {}", i, v);
        }
        println!("PASS: Two constants add");
    }

    #[test]
    fn test_two_constants_mul() {
        // 2つの定数: 2.0 * 3.0 = 6.0
        let mut pipeline = create_pipeline();
        let device = pipeline.device().clone();

        let a = broadcast_constant(2.0);
        let b = broadcast_constant(3.0);
        let result = a * b;

        let mut graph = Graph::new();
        graph.output("result", result);

        println!("=== Two constants mul test ===");
        println!("DSL:\n{}", harp_dsl::decompile(&graph));

        let compiled = pipeline.compile_program(graph).expect("compile");
        let mut output =
            OpenCLBuffer::allocate(&device, vec![4, 4], harp_core::ast::DType::F32).expect("alloc");

        let inputs: HashMap<String, &OpenCLBuffer> = HashMap::new();
        let mut outputs: HashMap<String, &mut OpenCLBuffer> = HashMap::new();
        outputs.insert("result".to_string(), &mut output);

        compiled
            .execute(&device, &inputs, &mut outputs)
            .expect("execute");

        let data: Vec<f32> = output.read_vec().expect("read");
        println!("Result: {:?}", &data[..4]);

        for (i, v) in data.iter().enumerate() {
            assert!((v - 6.0).abs() < 1e-5, "[{}] Expected 6.0, got {}", i, v);
        }
        println!("PASS: Two constants mul");
    }

    #[test]
    fn test_three_constants_add_then_mul() {
        // 3つの定数: (2.0 + 3.0) * 1.0 = 5.0
        // これがlazy-arrayで8.0を返すバグの再現テスト
        let mut pipeline = create_pipeline();
        let device = pipeline.device().clone();

        let a = broadcast_constant(2.0);
        let b = broadcast_constant(3.0);
        let c = broadcast_constant(1.0);

        let ab = a + b;
        let result = ab * c;

        let mut graph = Graph::new();
        graph.output("result", result);

        println!("=== Three constants (a+b)*c test ===");
        println!("DSL:\n{}", harp_dsl::decompile(&graph));

        let compiled = pipeline.compile_program(graph).expect("compile");
        let mut output =
            OpenCLBuffer::allocate(&device, vec![4, 4], harp_core::ast::DType::F32).expect("alloc");

        let inputs: HashMap<String, &OpenCLBuffer> = HashMap::new();
        let mut outputs: HashMap<String, &mut OpenCLBuffer> = HashMap::new();
        outputs.insert("result".to_string(), &mut output);

        compiled
            .execute(&device, &inputs, &mut outputs)
            .expect("execute");

        let data: Vec<f32> = output.read_vec().expect("read");
        println!("Result: {:?}", &data[..4]);

        // 期待値: (2 + 3) * 1 = 5
        for (i, v) in data.iter().enumerate() {
            assert!((v - 5.0).abs() < 1e-5, "[{}] Expected 5.0, got {}", i, v);
        }
        println!("PASS: Three constants (a+b)*c");
    }

    #[test]
    fn test_three_constants_different_order() {
        // 3つの定数: 2.0 * (3.0 + 1.0) = 8.0
        let mut pipeline = create_pipeline();
        let device = pipeline.device().clone();

        let a = broadcast_constant(2.0);
        let b = broadcast_constant(3.0);
        let c = broadcast_constant(1.0);

        let bc = b + c;
        let result = a * bc;

        let mut graph = Graph::new();
        graph.output("result", result);

        println!("=== Three constants a*(b+c) test ===");
        println!("DSL:\n{}", harp_dsl::decompile(&graph));

        let compiled = pipeline.compile_program(graph).expect("compile");
        let mut output =
            OpenCLBuffer::allocate(&device, vec![4, 4], harp_core::ast::DType::F32).expect("alloc");

        let inputs: HashMap<String, &OpenCLBuffer> = HashMap::new();
        let mut outputs: HashMap<String, &mut OpenCLBuffer> = HashMap::new();
        outputs.insert("result".to_string(), &mut output);

        compiled
            .execute(&device, &inputs, &mut outputs)
            .expect("execute");

        let data: Vec<f32> = output.read_vec().expect("read");
        println!("Result: {:?}", &data[..4]);

        // 期待値: 2 * (3 + 1) = 8
        for (i, v) in data.iter().enumerate() {
            assert!((v - 8.0).abs() < 1e-5, "[{}] Expected 8.0, got {}", i, v);
        }
        println!("PASS: Three constants a*(b+c)");
    }

    #[test]
    fn test_four_constants() {
        // 4つの定数: (2.0 + 3.0) * (1.0 + 4.0) = 5 * 5 = 25
        let mut pipeline = create_pipeline();
        let device = pipeline.device().clone();

        let a = broadcast_constant(2.0);
        let b = broadcast_constant(3.0);
        let c = broadcast_constant(1.0);
        let d = broadcast_constant(4.0);

        let ab = a + b;
        let cd = c + d;
        let result = ab * cd;

        let mut graph = Graph::new();
        graph.output("result", result);

        println!("=== Four constants (a+b)*(c+d) test ===");
        println!("DSL:\n{}", harp_dsl::decompile(&graph));

        let compiled = pipeline.compile_program(graph).expect("compile");
        let mut output =
            OpenCLBuffer::allocate(&device, vec![4, 4], harp_core::ast::DType::F32).expect("alloc");

        let inputs: HashMap<String, &OpenCLBuffer> = HashMap::new();
        let mut outputs: HashMap<String, &mut OpenCLBuffer> = HashMap::new();
        outputs.insert("result".to_string(), &mut output);

        compiled
            .execute(&device, &inputs, &mut outputs)
            .expect("execute");

        let data: Vec<f32> = output.read_vec().expect("read");
        println!("Result: {:?}", &data[..4]);

        // 期待値: (2 + 3) * (1 + 4) = 25
        for (i, v) in data.iter().enumerate() {
            assert!((v - 25.0).abs() < 1e-5, "[{}] Expected 25.0, got {}", i, v);
        }
        println!("PASS: Four constants (a+b)*(c+d)");
    }

    /// 入力バッファを使った場合のテスト（定数ではなく外部入力）
    #[test]
    fn test_three_inputs_add_then_mul() {
        // 3つの入力: (input_a + input_b) * input_c
        let mut pipeline = create_pipeline();
        let device = pipeline.device().clone();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![4, 4]);
        let b = graph.input("b", DType::F32, vec![4, 4]);
        let c = graph.input("c", DType::F32, vec![4, 4]);

        let ab = a + b;
        let result = ab * c;
        graph.output("result", result);

        println!("=== Three inputs (a+b)*c test ===");
        println!("DSL:\n{}", harp_dsl::decompile(&graph));

        let compiled = pipeline.compile_program(graph).expect("compile");

        // 入力バッファを作成
        let buf_a = OpenCLBuffer::from_vec(
            &device,
            vec![4, 4],
            harp_core::ast::DType::F32,
            &[2.0f32; 16],
        )
        .expect("alloc a");
        let buf_b = OpenCLBuffer::from_vec(
            &device,
            vec![4, 4],
            harp_core::ast::DType::F32,
            &[3.0f32; 16],
        )
        .expect("alloc b");
        let buf_c = OpenCLBuffer::from_vec(
            &device,
            vec![4, 4],
            harp_core::ast::DType::F32,
            &[1.0f32; 16],
        )
        .expect("alloc c");
        let mut output = OpenCLBuffer::allocate(&device, vec![4, 4], harp_core::ast::DType::F32)
            .expect("alloc out");

        let mut inputs: HashMap<String, &OpenCLBuffer> = HashMap::new();
        inputs.insert("a".to_string(), &buf_a);
        inputs.insert("b".to_string(), &buf_b);
        inputs.insert("c".to_string(), &buf_c);

        let mut outputs: HashMap<String, &mut OpenCLBuffer> = HashMap::new();
        outputs.insert("result".to_string(), &mut output);

        compiled
            .execute(&device, &inputs, &mut outputs)
            .expect("execute");

        let data: Vec<f32> = output.read_vec().expect("read");
        println!("Result: {:?}", &data[..4]);

        // 期待値: (2 + 3) * 1 = 5
        for (i, v) in data.iter().enumerate() {
            assert!((v - 5.0).abs() < 1e-5, "[{}] Expected 5.0, got {}", i, v);
        }
        println!("PASS: Three inputs (a+b)*c");
    }
}
