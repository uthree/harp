use super::Lowerer;
use crate::ast::helper::function;
use crate::ast::{AstNode, ConstLiteral, DType, Scope, VariableDecl};
use crate::graph::{Graph, GraphNode};
use crate::lowerer::reduce::ReduceLowerer;
use crate::lowerer::utils::LowererUtils;

impl Lowerer {
    pub(super) fn create_kernel_function(&mut self, graph: &Graph) -> AstNode {
        // 0.5. graphのinputsの順序通りに入力ノードをマッピング
        for weak_input in graph.inputs.iter() {
            if let Some(_input_rc) = weak_input.upgrade() {
                // GraphNodeを作成するには、トポロジカルソートで得られたノードと照合する必要がある
                // しかし、ここではまだトポロジカルソートしていないので、一旦保留
            }
        }

        // 1. トポロジカルソート（世代別）
        let generations = self.topological_sort_by_generation(graph);

        // 1.5. 入力ノードと出力ノードに対して変数名を事前マッピング
        // graphのinputsと照合して正しい順序を維持
        for (i, weak_input) in graph.inputs.iter().enumerate() {
            if let Some(input_rc) = weak_input.upgrade() {
                // generationsから同じノードを探す
                // GraphNodeはRc<GraphNodeData>をラップしており、Eq/Hashが実装されている
                let input_node = GraphNode::from_rc(input_rc);
                let var_name = format!("input_{}", i);
                self.node_to_var.insert(input_node, var_name);
            }
        }

        // 出力ノードもマッピング
        for (i, output_node) in graph.outputs.iter().enumerate() {
            let var_name = format!("output_{}", i);
            self.node_to_var.insert(output_node.clone(), var_name);
        }

        // 2. 各世代のノードを処理してAST文を生成
        // 世代間にBarrierを挿入
        let mut statements = Vec::new();
        let mut declarations = Vec::new();

        for (gen_idx, generation) in generations.iter().enumerate() {
            // 世代内の各ノードを処理
            for node in generation {
                let ast_stmt = self.lower_node(node, &mut declarations);
                if let Some(stmt) = ast_stmt {
                    statements.push(stmt);
                }
            }

            // 最後の世代でなければ、Barrierを挿入
            if gen_idx < generations.len() - 1 {
                statements.push(AstNode::Barrier);
            }
        }

        // 3. 出力ノードに対して、必要に応じてコピーコードを生成
        for (output_idx, output_node) in graph.outputs.iter().enumerate() {
            let output_var = format!("output_{}", output_idx);
            let source_var = self.get_or_create_var_name(output_node);

            eprintln!(
                "Output {}: output_var={}, source_var={}",
                output_idx, output_var, source_var
            );

            // 出力変数とソース変数が異なる場合、コピーが必要
            if output_var != source_var {
                // メモリコピーを生成
                let output_view = &output_node.view;
                // シンプルなコピーループを生成
                let copy_stmt =
                    ReduceLowerer::create_copy_loop(output_view, &source_var, &output_var, 0);
                statements.push(copy_stmt);
            }
        }

        // 3. 入力パラメータを作成（入力+出力）
        let arguments = self.create_kernel_arguments(graph);

        // 4. カーネル関数を構築
        function(
            "kernel_impl".to_string(),
            arguments,
            DType::Void,
            Scope { declarations },
            statements,
        )
    }

    pub(super) fn create_entry_function(
        &self,
        graph: &Graph,
        _kernel_function: &AstNode,
    ) -> AstNode {
        let mut statements = Vec::new();
        let mut local_vars = Vec::new();

        // バッファポインタから具体的な型付きポインタを取得
        let mut arg_index = 0;

        // 入力バッファの型キャスト
        for (i, weak_ref) in graph.inputs.iter().enumerate() {
            if let Some(node_data) = weak_ref.upgrade() {
                let var_name = format!("input_{}", i);
                let _cast_type = LowererUtils::get_c_type(&node_data.dtype);

                local_vars.push(VariableDecl {
                    name: var_name.clone(),
                    dtype: DType::Ptr(Box::new(node_data.dtype.clone())),
                    constant: false,
                    size_expr: None,
                });

                // cast: float* input_0 = (float*)bufs[0];
                statements.push(AstNode::Assign(
                    var_name,
                    Box::new(AstNode::Cast {
                        dtype: DType::Ptr(Box::new(node_data.dtype.clone())),
                        expr: Box::new(AstNode::Load {
                            target: Box::new(AstNode::Var("bufs".to_string())),
                            index: Box::new(AstNode::Const(ConstLiteral::Usize(arg_index))),
                            vector_width: 1,
                        }),
                    }),
                ));
                arg_index += 1;
            }
        }

        // 出力バッファの型キャスト
        for (i, output_node) in graph.outputs.iter().enumerate() {
            let var_name = format!("output_{}", i);
            let _cast_type = LowererUtils::get_c_type(&output_node.dtype);

            local_vars.push(VariableDecl {
                name: var_name.clone(),
                dtype: DType::Ptr(Box::new(output_node.dtype.clone())),
                constant: false,
                size_expr: None,
            });

            statements.push(AstNode::Assign(
                var_name,
                Box::new(AstNode::Cast {
                    dtype: DType::Ptr(Box::new(output_node.dtype.clone())),
                    expr: Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var("bufs".to_string())),
                        index: Box::new(AstNode::Const(ConstLiteral::Usize(arg_index))),
                        vector_width: 1,
                    }),
                }),
            ));
            arg_index += 1;
        }

        // カーネル関数呼び出し
        let mut call_args = Vec::new();

        // 入力引数
        for (i, _) in graph.inputs.iter().enumerate() {
            call_args.push(AstNode::Var(format!("input_{}", i)));
        }

        // 出力引数
        for (i, _) in graph.outputs.iter().enumerate() {
            call_args.push(AstNode::Var(format!("output_{}", i)));
        }

        statements.push(AstNode::CallFunction {
            name: "kernel_impl".to_string(),
            args: call_args,
        });

        function(
            "kernel_main".to_string(),
            vec![
                (
                    "bufs".to_string(),
                    DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))),
                ),
                ("shape_vars".to_string(), DType::Ptr(Box::new(DType::Usize))),
            ],
            DType::Void,
            Scope {
                declarations: local_vars,
            },
            statements,
        )
    }

    fn create_kernel_arguments(&self, graph: &Graph) -> Vec<(String, DType)> {
        let mut arguments = Vec::new();

        // 入力引数
        for (i, weak_ref) in graph.inputs.iter().enumerate() {
            if let Some(node_data) = weak_ref.upgrade() {
                arguments.push((
                    format!("input_{}", i),
                    DType::Ptr(Box::new(node_data.dtype.clone())),
                ));
            }
        }

        // 出力引数
        for (i, output_node) in graph.outputs.iter().enumerate() {
            arguments.push((
                format!("output_{}", i),
                DType::Ptr(Box::new(output_node.dtype.clone())),
            ));
        }

        arguments
    }
}
