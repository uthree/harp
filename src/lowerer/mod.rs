use crate::ast::{
    AccessRegion, AstNode, DType as AstDType, Function, FunctionKind, Literal, Mutability, Scope,
    VarDecl, VarKind, helper::*,
};
use crate::graph::{DType as GraphDType, Graph, GraphNode, ops::ElementwiseOp, ops::GraphOp};
use log::debug;
use std::collections::{HashMap, HashSet, VecDeque};

pub struct Lowerer {
    alu_counter: usize, // 一時変数のカウンター
}

/// トポロジカルソートの結果。各世代（Generation）は並列実行可能なノード群。
pub type TopologicalOrder = Vec<Vec<GraphNode>>;

/// GraphNodeから内部のポインタを取得するヘルパー関数
fn node_ptr(node: &GraphNode) -> *const () {
    node.as_ptr() as *const ()
}

impl Lowerer {
    pub fn new() -> Self {
        Self { alu_counter: 0 }
    }

    /// 新しい一時変数名を生成
    fn fresh_alu(&mut self) -> String {
        let name = format!("alu{}", self.alu_counter);
        self.alu_counter += 1;
        name
    }

    /// GraphNodeを一つのカーネル関数に変換（最も単純なケース）
    /// 前提：contiguous, 全軸Sequential, SIMD未使用
    pub fn lower_node_to_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
    ) -> Result<Function, String> {
        // 現時点では、Elementwise演算のみをサポート
        match &node.op {
            GraphOp::Elementwise(op) => self.lower_elementwise_kernel(node, node_id, op),
            _ => Err(format!("Unsupported operation: {:?}", node.op)),
        }
    }

    /// Elementwise演算をカーネル関数に変換
    fn lower_elementwise_kernel(
        &mut self,
        node: &GraphNode,
        _node_id: usize,
        op: &ElementwiseOp,
    ) -> Result<Function, String> {
        debug!("Lowering elementwise operation: {:?}", op);
        debug!("View: {:?}", node.view);
        debug!("Is contiguous: {}", node.view.is_contiguous());

        let shape = node.view.shape();
        let ndim = shape.len();

        // パラメータを生成: 入力バッファー、出力バッファー、shape変数
        let mut params = Vec::new();

        // 入力バッファー（srcノード）
        for (i, src) in node.src.iter().enumerate() {
            let dtype = self.graph_dtype_to_ast_ptr(&src.dtype)?;
            params.push(VarDecl {
                name: format!("input{}", i),
                dtype,
                mutability: Mutability::Immutable,
                region: AccessRegion::Shared,
                kind: VarKind::Normal,
            });
        }

        // 出力バッファー
        let output_dtype = self.graph_dtype_to_ast_ptr(&node.dtype)?;
        params.push(VarDecl {
            name: "output".to_string(),
            dtype: output_dtype,
            mutability: Mutability::Mutable,
            region: AccessRegion::Shared,
            kind: VarKind::Normal,
        });

        // Shape変数（各軸のサイズ）
        for i in 0..ndim {
            params.push(VarDecl {
                name: format!("shape{}", i),
                dtype: AstDType::Usize,
                mutability: Mutability::Immutable,
                region: AccessRegion::Shared,
                kind: VarKind::Normal,
            });
        }

        // ループ本体の生成
        let body_statements = self.generate_elementwise_loops(node, op, ndim)?;

        // カーネル関数を作成
        let function = Function::new(
            FunctionKind::Normal, // まずは通常の関数として（並列化は後で）
            params,
            AstDType::Tuple(vec![]), // unit型
            body_statements,
        )?;

        // 生成されたコードをログ出力
        debug!(
            "Generated function with {} parameters",
            function.params.len()
        );
        if log::log_enabled!(log::Level::Debug) {
            use crate::backend::metal::MetalRenderer;
            let mut renderer = MetalRenderer::new();
            let code = renderer.render_function("kernel_fn", &function);
            debug!("Generated code:\n{}", code);
        }

        Ok(function)
    }

    /// Elementwise演算のループを生成
    fn generate_elementwise_loops(
        &mut self,
        node: &GraphNode,
        op: &ElementwiseOp,
        ndim: usize,
    ) -> Result<Vec<AstNode>, String> {
        if ndim == 0 {
            // スカラー演算（ループなし）
            return self.generate_elementwise_body(node, op, &[]);
        }

        // ネストしたループを生成（外側から内側へ）
        let mut body_statements =
            self.generate_elementwise_body(node, op, &(0..ndim).collect::<Vec<_>>())?;

        // ループを逆順に作成（内側から外側へ）
        for axis in (0..ndim).rev() {
            let loop_var = format!("ridx{}", axis);
            let shape_var = var(format!("shape{}", axis));

            let loop_body = AstNode::Block {
                statements: body_statements,
                scope: Box::new(Scope::new()),
            };

            body_statements = vec![AstNode::Range {
                var: loop_var.clone(),
                start: Box::new(AstNode::Const(Literal::Usize(0))),
                step: Box::new(AstNode::Const(Literal::Usize(1))),
                stop: Box::new(shape_var),
                body: Box::new(loop_body),
            }];
        }

        Ok(body_statements)
    }

    /// Elementwise演算の本体を生成（ループ内部の処理）
    fn generate_elementwise_body(
        &mut self,
        node: &GraphNode,
        op: &ElementwiseOp,
        axes: &[usize],
    ) -> Result<Vec<AstNode>, String> {
        let mut statements = Vec::new();

        // 入力をロード（各入力のViewを考慮）
        let mut loaded_values = Vec::new();
        for (i, src) in node.src.iter().enumerate() {
            let alu_var = self.fresh_alu();
            let input_ptr = var(format!("input{}", i));

            // 各srcノードのViewからオフセットを計算
            let offset = self.compute_offset_from_view(src, axes);
            let load_node = load(input_ptr, offset);

            statements.push(assign(&alu_var, load_node));
            loaded_values.push(var(&alu_var));
        }

        // 演算を適用
        let result = self.apply_elementwise_op(op, &loaded_values)?;
        let result_var = self.fresh_alu();
        statements.push(assign(&result_var, result));

        // 結果をストア（出力のViewを考慮）
        let output_ptr = var("output");
        let output_offset = self.compute_offset_from_view(node, axes);
        statements.push(store(output_ptr, output_offset, var(&result_var)));

        Ok(statements)
    }

    /// Viewを考慮したオフセット計算
    fn compute_offset_from_view(&self, node: &GraphNode, axes: &[usize]) -> AstNode {
        use crate::graph::shape::View;

        if axes.is_empty() {
            // スカラーの場合
            match &node.view {
                View::Linear { offset, .. } => {
                    // Expr::intoでAstNodeに変換
                    offset.clone().into()
                }
            }
        } else {
            // テンソルの場合：offset + sum(ridx[i] * stride[i])
            match &node.view {
                View::Linear {
                    strides, offset, ..
                } => {
                    let mut result: AstNode = offset.clone().into();

                    for &axis in axes {
                        let ridx = var(format!("ridx{}", axis));
                        let stride: AstNode = strides[axis].clone().into();
                        result = result + ridx * stride;
                    }

                    result
                }
            }
        }
    }

    /// Elementwise演算をASTノードに変換
    fn apply_elementwise_op(
        &self,
        op: &ElementwiseOp,
        operands: &[AstNode],
    ) -> Result<AstNode, String> {
        match op {
            ElementwiseOp::Add => {
                if operands.len() != 2 {
                    return Err("Add requires 2 operands".to_string());
                }
                Ok(operands[0].clone() + operands[1].clone())
            }
            ElementwiseOp::Mul => {
                if operands.len() != 2 {
                    return Err("Mul requires 2 operands".to_string());
                }
                Ok(operands[0].clone() * operands[1].clone())
            }
            ElementwiseOp::Neg => {
                if operands.len() != 1 {
                    return Err("Neg requires 1 operand".to_string());
                }
                // -x = -1 * x
                Ok(AstNode::Const(Literal::F32(-1.0)) * operands[0].clone())
            }
            ElementwiseOp::Max => {
                if operands.len() != 2 {
                    return Err("Max requires 2 operands".to_string());
                }
                Ok(max(operands[0].clone(), operands[1].clone()))
            }
            ElementwiseOp::Rem => {
                if operands.len() != 2 {
                    return Err("Rem requires 2 operands".to_string());
                }
                Ok(operands[0].clone() % operands[1].clone())
            }
            ElementwiseOp::Idiv => {
                if operands.len() != 2 {
                    return Err("Idiv requires 2 operands".to_string());
                }
                Ok(idiv(operands[0].clone(), operands[1].clone()))
            }
            ElementwiseOp::Recip => {
                if operands.len() != 1 {
                    return Err("Recip requires 1 operand".to_string());
                }
                Ok(recip(operands[0].clone()))
            }
        }
    }

    /// GraphのDTypeをASTのPtr<DType>に変換
    fn graph_dtype_to_ast_ptr(&self, dtype: &GraphDType) -> Result<AstDType, String> {
        let element_dtype = match dtype {
            GraphDType::F32 => AstDType::F32,
            GraphDType::Unknown => return Err("Cannot convert Unknown dtype".to_string()),
        };
        Ok(AstDType::Ptr(Box::new(element_dtype)))
    }

    // === トポロジカルソート関連 ===

    /// Kahnのアルゴリズムを使用してグラフをトポロジカルソートし、世代別にグループ化する。
    /// 各世代のノードは同時に計算可能。
    pub fn topological_sort(graph: &Graph) -> TopologicalOrder {
        // 1. すべてのノードを収集（出力ノードから再帰的に辿る）
        let all_nodes = Self::collect_all_nodes(graph);

        // 2. 各ノードの入次数を計算（何個のノードから参照されているか）
        let mut in_degree: HashMap<*const (), usize> = HashMap::new();
        for node in &all_nodes {
            let ptr = node_ptr(node);
            in_degree.entry(ptr).or_insert(0);

            // このノードが参照する各srcノードの入次数を増やす
            for src in &node.src {
                let src_ptr = node_ptr(src);
                *in_degree.entry(src_ptr).or_insert(0) += 1;
            }
        }

        // 3. Kahnのアルゴリズムで世代別にグループ化
        let mut result: TopologicalOrder = Vec::new();
        let mut queue: VecDeque<GraphNode> = VecDeque::new();

        // 入次数が0のノード（誰からも参照されていない=出力ノード）をキューに追加
        for node in &all_nodes {
            let ptr = node_ptr(node);
            if in_degree[&ptr] == 0 {
                queue.push_back(node.clone());
            }
        }

        // 世代ごとに処理
        while !queue.is_empty() {
            let generation_size = queue.len();
            let mut current_generation = Vec::new();

            // 現在の世代のノードをすべて処理
            for _ in 0..generation_size {
                let node = queue.pop_front().unwrap();
                current_generation.push(node.clone());

                // このノードが参照するsrcノードの入次数を減らす
                for src in &node.src {
                    let src_ptr = node_ptr(src);
                    let degree = in_degree.get_mut(&src_ptr).unwrap();
                    *degree -= 1;

                    // 入次数が0になったらキューに追加
                    if *degree == 0 {
                        queue.push_back(src.clone());
                    }
                }
            }

            result.push(current_generation);
        }

        result
    }

    /// グラフの出力ノードから再帰的にすべてのノードを収集する
    fn collect_all_nodes(graph: &Graph) -> Vec<GraphNode> {
        let mut visited: HashSet<*const ()> = HashSet::new();
        let mut nodes: Vec<GraphNode> = Vec::new();

        for (_, output_node) in graph.outputs() {
            Self::collect_nodes_recursive(output_node, &mut visited, &mut nodes);
        }

        nodes
    }

    /// 再帰的にノードを収集する（深さ優先探索）
    fn collect_nodes_recursive(
        node: &GraphNode,
        visited: &mut HashSet<*const ()>,
        nodes: &mut Vec<GraphNode>,
    ) {
        let ptr = node_ptr(node);

        if visited.contains(&ptr) {
            return;
        }

        visited.insert(ptr);

        // 先にsrcノードを訪問（依存関係の順序）
        for src in &node.src {
            Self::collect_nodes_recursive(src, visited, nodes);
        }

        nodes.push(node.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DType as GraphDType;

    #[test]
    fn test_lower_simple_add() {
        let _ = env_logger::builder().is_test(true).try_init();

        // a + b のグラフをカーネルに変換
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![10])
            .build();
        let result = a + b;

        // カーネル関数を生成
        let mut lowerer = Lowerer::new();
        let function = lowerer.lower_node_to_kernel(&result, 0);

        assert!(function.is_ok());
        let function = function.unwrap();

        // パラメータをチェック: input0, input1, output, shape0
        assert_eq!(function.params.len(), 4);
        assert_eq!(function.params[0].name, "input0");
        assert_eq!(function.params[1].name, "input1");
        assert_eq!(function.params[2].name, "output");
        assert_eq!(function.params[3].name, "shape0");

        // 返り値の型はunit型
        assert_eq!(function.return_type, AstDType::Tuple(vec![]));

        // 生成されたコードを表示（テスト実行時に確認用）
        use crate::backend::metal::MetalRenderer;
        let mut renderer = MetalRenderer::new();
        let code = renderer.render_function("test_add_kernel", &function);
        eprintln!(
            "\n=== Generated Code for test_lower_simple_add ===\n{}\n",
            code
        );
    }

    #[test]
    fn test_lower_simple_mul() {
        // a * b のグラフをカーネルに変換
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![20])
            .build();
        let b = graph
            .input("b")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![20])
            .build();
        let result = a * b;

        let mut lowerer = Lowerer::new();
        let function = lowerer.lower_node_to_kernel(&result, 0);

        assert!(function.is_ok());
        let function = function.unwrap();

        // パラメータをチェック
        assert_eq!(function.params.len(), 4);
        assert_eq!(function.params[0].name, "input0");
        assert_eq!(function.params[1].name, "input1");
        assert_eq!(function.params[2].name, "output");
    }

    #[test]
    fn test_lower_neg() {
        // -a のグラフをカーネルに変換
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![10])
            .build();
        let result = -a;

        let mut lowerer = Lowerer::new();
        let function = lowerer.lower_node_to_kernel(&result, 0);

        assert!(function.is_ok());
        let function = function.unwrap();

        // パラメータをチェック: input0, output, shape0
        assert_eq!(function.params.len(), 3);
        assert_eq!(function.params[0].name, "input0");
        assert_eq!(function.params[1].name, "output");
        assert_eq!(function.params[2].name, "shape0");
    }

    #[test]
    fn test_lower_with_permute() {
        use crate::graph::ops::GraphOp;

        // 転置されたテンソルの加算
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![3, 4])
            .build();
        let _b = graph
            .input("b")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![3, 4])
            .build();

        // aを転置: (3, 4) -> (4, 3)
        let a_transposed = GraphNode::new(
            GraphDType::F32,
            GraphOp::View(a.view.clone().permute(vec![1, 0])),
            vec![a.clone()],
            a.view.clone().permute(vec![1, 0]),
        );

        // 転置されたaと同じshapeのbの加算は失敗するはず
        // （ここでは単純に転置されたViewの動作をテスト）
        let mut lowerer = Lowerer::new();

        // 転置されたテンソルのloweringをテスト
        let function = lowerer.lower_node_to_kernel(&a_transposed, 0);

        // Viewノードは直接lowering対象ではないのでエラーになる
        assert!(function.is_err());
    }

    #[test]
    fn test_lower_with_flipped_view() {
        let _ = env_logger::builder().is_test(true).try_init();

        // flipされたテンソルの否定演算
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![10])
            .build();

        // aをflip
        let flipped_view = a.view.clone().flip(0);

        // flipされたaの否定演算
        // Viewの変更を直接Elementwise演算のsrcに含める
        let a_flipped = GraphNode::new(GraphDType::F32, a.op.clone(), a.src.clone(), flipped_view);

        let result = -a_flipped;

        let mut lowerer = Lowerer::new();
        let function = lowerer.lower_node_to_kernel(&result, 0);

        // View変換が実装されたので成功するはず
        assert!(function.is_ok());
        let function = function.unwrap();

        // パラメータをチェック
        assert_eq!(function.params.len(), 3);

        // 生成されたコードを表示（テスト実行時に確認用）
        use crate::backend::metal::MetalRenderer;
        let mut renderer = MetalRenderer::new();
        let code = renderer.render_function("test_flip_kernel", &function);
        eprintln!(
            "\n=== Generated Code for test_lower_with_flipped_view ===\n{}\n",
            code
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_end_to_end_execution() {
        let _ = env_logger::builder().is_test(true).try_init();

        // 手動でMetalカーネルを作成（lowererの出力を参考に）
        // 後でlowererと統合する予定
        let source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void test_add(
    device const float* input0 [[buffer(0)]],
    device const float* input1 [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    output[tid] = input0[tid] + input1[tid];
}
"#;

        eprintln!("\n=== Metal Kernel ===\n{}\n", source);

        // Metal compilerで実行
        use crate::backend::metal::{MetalCode, MetalCompiler};
        use crate::backend::Compiler;
        if let Some(mut compiler) = MetalCompiler::with_default_device() {
            let code = MetalCode::new(source.to_string());
            let mut kernel = compiler.compile(&code);

            // バッファを作成
            let mut input0_buffer = compiler.create_buffer(vec![10], 4);
            let mut input1_buffer = compiler.create_buffer(vec![10], 4);
            let output_buffer = compiler.create_buffer(vec![10], 4);

            // 入力データを設定
            let input0_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
            let input1_data: Vec<f32> = (0..10).map(|i| (i * 2) as f32).collect();

            input0_buffer.write_data(&input0_data);
            input1_buffer.write_data(&input1_data);

            // グリッドサイズを設定
            kernel.set_grid_size(10, 1, 1);

            // カーネルを実行
            kernel
                .dispatch(&[&input0_buffer, &input1_buffer, &output_buffer])
                .unwrap();

            // 結果を読み出し
            let mut output_data = vec![0.0f32; 10];
            output_buffer.read_data(&mut output_data);

            // 確認
            let expected: Vec<f32> = input0_data
                .iter()
                .zip(input1_data.iter())
                .map(|(&x, &y)| x + y)
                .collect();

            eprintln!("Input 0: {:?}", input0_data);
            eprintln!("Input 1: {:?}", input1_data);
            eprintln!("Output:  {:?}", output_data);
            eprintln!("Expected: {:?}", expected);

            assert_eq!(output_data, expected);
            eprintln!("\n✅ End-to-end execution successful!\n");
        } else {
            eprintln!("⚠️ Metal not available, skipping test");
        }
    }

    #[test]
    fn test_topological_sort_simple() {
        // a + b のグラフ
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![10])
            .build();
        let result = a + b;
        graph.output("result", result);

        let order = Lowerer::topological_sort(&graph);

        // 2世代に分かれる：
        // Generation 0: result (+)
        // Generation 1: a, b (並列実行可能な入力ノード)
        assert_eq!(order.len(), 2);
        assert_eq!(order[0].len(), 1); // result
        assert_eq!(order[1].len(), 2); // a, b
    }

    #[test]
    fn test_topological_sort_complex() {
        // (a + b) * (c + d) のグラフ
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![10])
            .build();
        let c = graph
            .input("c")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![10])
            .build();
        let d = graph
            .input("d")
            .with_dtype(GraphDType::F32)
            .with_shape(vec![10])
            .build();

        let sum1 = a + b;
        let sum2 = c + d;
        let result = sum1 * sum2;
        graph.output("result", result);

        let order = Lowerer::topological_sort(&graph);

        // 世代構造を確認
        // Generation 0: result (*)
        // Generation 1: sum1 (+), sum2 (+) - 並列実行可能
        // Generation 2: a, b, c, d - 並列実行可能（入力ノード）
        assert_eq!(order.len(), 3);
        assert_eq!(order[0].len(), 1); // result
        assert_eq!(order[1].len(), 2); // sum1, sum2
        assert_eq!(order[2].len(), 4); // a, b, c, d
    }
}
