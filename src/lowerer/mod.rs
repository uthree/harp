mod copy_loops;
mod core;
mod cumulative;
mod elementwise;
mod fold_loops;
mod fused_elementwise;
mod fused_elementwise_cumulative;
mod fused_elementwise_reduce;
mod fused_reduce;
mod kernel;
mod lower_node;
mod reduce;
mod topological;
mod utils;

pub use core::{lower, Lowerer};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AstNode, DType};
    use crate::graph::{Graph, GraphNode};

    #[test]
    fn test_simple_constant() {
        let mut graph = Graph::new();
        let mut lowerer = Lowerer::new();

        // 単純な定数のみのグラフ
        let constant_node = GraphNode::f32(1.0);
        graph.output(constant_node);

        // lower処理
        let program = lowerer.lower(&graph);

        // 基本的なチェック
        if let AstNode::Program {
            entry_point,
            functions,
        } = &program
        {
            assert_eq!(entry_point, "kernel_main");
            assert_eq!(functions.len(), 2); // kernel_impl + kernel_main

            // エントリーポイント関数のチェック
            if let AstNode::Function {
                name,
                return_type,
                arguments,
                ..
            } = &functions[1]
            {
                assert_eq!(name, "kernel_main");
                assert_eq!(return_type, &DType::Void);
                assert_eq!(arguments.len(), 2); // bufs, shape_vars
            } else {
                panic!("Expected Function node");
            }
        } else {
            panic!("Expected Program node");
        }
    }

    #[test]
    fn test_input_only() {
        let mut graph = Graph::new();
        let mut lowerer = Lowerer::new();

        // 入力のみのグラフ
        let input_node = graph.input(DType::F32, vec![2.into(), 3.into()]);
        graph.output(input_node);

        // lower処理
        let program = lowerer.lower(&graph);

        // 基本的なチェック
        if let AstNode::Program {
            entry_point,
            functions,
        } = &program
        {
            assert_eq!(entry_point, "kernel_main");
            assert_eq!(functions.len(), 2);

            // カーネル実装関数のチェック
            if let AstNode::Function {
                name, arguments, ..
            } = &functions[0]
            {
                assert_eq!(name, "kernel_impl");
                assert_eq!(arguments.len(), 2); // input_0 + output_0
            } else {
                panic!("Expected Function node");
            }
        } else {
            panic!("Expected Program node");
        }
    }

    #[test]
    fn test_elementwise_negation() {
        let mut graph = Graph::new();
        let mut lowerer = Lowerer::new();

        // 単項演算: -constant
        let constant_node = GraphNode::f32(1.0);
        let negated = -constant_node;
        graph.output(negated);

        // lower処理
        let program = lowerer.lower(&graph);

        // 基本的なチェック
        if let AstNode::Program {
            entry_point,
            functions,
        } = &program
        {
            assert_eq!(entry_point, "kernel_main");
            assert_eq!(functions.len(), 2);

            // カーネル実装関数のチェック
            if let AstNode::Function { statements, .. } = &functions[0] {
                // const assignment + barrier + neg loop
                assert_eq!(statements.len(), 3);
                // 2番目のステートメントがBarrierであることを確認
                assert!(matches!(statements[1], AstNode::Barrier));
            } else {
                panic!("Expected Function node");
            }
        } else {
            panic!("Expected Program node");
        }
    }

    #[test]
    fn test_entry_point_structure() {
        let mut graph = Graph::new();
        let mut lowerer = Lowerer::new();

        // 入力と出力があるグラフ
        let input_node = graph.input(DType::F32, vec![4.into()]);
        let _constant = GraphNode::f32(2.0);
        let result = -input_node; // 単項演算
        graph.output(result);

        let program = lowerer.lower(&graph);

        // エントリーポイント関数の詳細チェック
        if let AstNode::Program { functions, .. } = &program {
            if let AstNode::Function {
                name,
                arguments,
                scope,
                statements,
                ..
            } = &functions[1]
            {
                assert_eq!(name, "kernel_main");

                // 引数チェック: (void** bufs, size_t* shape_vars)
                assert_eq!(arguments.len(), 2);
                assert_eq!(arguments[0].0, "bufs");
                assert_eq!(arguments[1].0, "shape_vars");

                // エントリー関数の本体をチェック
                // 入力と出力バッファの型キャストがある
                assert!(statements.len() >= 3); // 最低でも input cast + output cast + kernel call

                // 変数宣言をチェック
                assert!(scope.declarations.len() >= 2); // input_0, output_0

                // 最後の文はkernel_impl呼び出し
                if let AstNode::CallFunction { name, args } = statements.last().unwrap() {
                    assert_eq!(name, "kernel_impl");
                    assert_eq!(args.len(), 2); // input_0, output_0
                } else {
                    panic!("Expected kernel call as last statement");
                }
            } else {
                panic!("Expected Function node");
            }
        } else {
            panic!("Expected Program node");
        }
    }
}
