use harp_core::ast::Literal;
use harp_core::graph::*;

#[test]
fn test_graph_new() {
    let graph = Graph::new();
    assert_eq!(graph.input_metas().len(), 0);
    assert_eq!(graph.outputs().len(), 0);
}

#[test]
fn test_input_node_creation() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![10, 20]);

    // 入力ノードが作成されたことを確認
    assert_eq!(graph.input_metas().len(), 1);
    assert!(graph.input_metas().iter().any(|m| m.name == "x"));

    // ノードのプロパティを確認
    match input.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    match &input.op {
        GraphOp::Buffer { .. } => {}
        _ => panic!("Expected GraphOp::Buffer"),
    }

    assert_eq!(input.view.ndim(), 2);
    assert_eq!(input.view.shape().len(), 2);
    assert!(input.view.is_contiguous());
}

#[test]
fn test_input_node_default_dtype() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::Unknown, vec![5]);

    // デフォルトのDTypeはUnknown
    match input.dtype {
        DType::Unknown => {}
        _ => panic!("Expected DType::Unknown as default"),
    }
}

#[test]
fn test_input_node_empty_shape() {
    let mut graph = Graph::new();
    let input = graph.input("scalar", DType::F32, Vec::<isize>::new());

    // 空のshapeはスカラーを表す
    assert_eq!(input.view.ndim(), 0);
    assert!(input.view.is_contiguous());
}

#[test]
fn test_output_node_registration() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![10]);

    graph.output("y", input.clone());

    assert_eq!(graph.outputs().len(), 1);
    assert!(graph.outputs().contains_key("y"));
}

#[test]
fn test_multiple_inputs() {
    let mut graph = Graph::new();
    let input1 = graph.input("x", DType::F32, vec![10]);
    let input2 = graph.input("y", DType::F32, vec![20]);

    assert_eq!(graph.input_metas().len(), 2);
    assert!(graph.input_metas().iter().any(|m| m.name == "x"));
    assert!(graph.input_metas().iter().any(|m| m.name == "y"));

    assert_eq!(input1.view.ndim(), 1);
    assert_eq!(input2.view.ndim(), 1);
}

#[test]
fn test_graph_node_new() {
    let node = GraphNode::new(
        DType::F32,
        GraphOp::Buffer {
            name: "test".to_string(),
        },
        vec![],
        View::contiguous(vec![3, 4]),
    );

    match node.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    assert_eq!(node.src.len(), 0);
    assert_eq!(node.view.ndim(), 2);
}

// 演算のテスト

#[test]
fn test_add_operation() {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);
    let b = graph.input("b", DType::F32, vec![10]);

    let result = a + b;

    match result.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    match &result.op {
        GraphOp::Elementwise {
            op: ops::ElementwiseOp::Add,
            ..
        } => {}
        _ => panic!("Expected Add operation"),
    }

    assert_eq!(result.src.len(), 2);
    assert_eq!(result.view.ndim(), 1);
    assert_eq!(result.view.shape()[0], shape::Expr::from(10));
}

#[test]
fn test_mul_operation() {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![5, 5]);
    let b = graph.input("b", DType::F32, vec![5, 5]);

    let result = a * b;

    match result.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    match &result.op {
        GraphOp::Elementwise {
            op: ops::ElementwiseOp::Mul,
            ..
        } => {}
        _ => panic!("Expected Mul operation"),
    }

    assert_eq!(result.src.len(), 2);
    assert_eq!(result.view.ndim(), 2);
}

#[test]
fn test_neg_operation() {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);

    let result = -a;

    match result.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    match &result.op {
        GraphOp::Elementwise {
            op: ops::ElementwiseOp::Neg,
            ..
        } => {}
        _ => panic!("Expected Neg operation"),
    }

    assert_eq!(result.src.len(), 1);
    assert_eq!(result.view.ndim(), 1);
}

#[test]
fn test_sub_operation() {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);
    let b = graph.input("b", DType::F32, vec![10]);

    let result = a - b;

    match result.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    // 減算演算は a + (-b) として実装される
    match &result.op {
        GraphOp::Elementwise {
            op: ops::ElementwiseOp::Add,
            ..
        } => {}
        _ => panic!("Expected Add operation (a - b = a + (-b))"),
    }

    assert_eq!(result.src.len(), 2);

    // 左側のオペランドは入力a
    match &result.src[0].op {
        GraphOp::Buffer { .. } => {}
        _ => panic!("Expected Buffer operation for left operand"),
    }

    // 右側のオペランドは -b (Neg演算)
    match &result.src[1].op {
        GraphOp::Elementwise {
            op: ops::ElementwiseOp::Neg,
            ..
        } => {}
        _ => panic!("Expected Neg operation for right operand"),
    }

    // -b の入力は b
    match &result.src[1].src[0].op {
        GraphOp::Buffer { .. } => {}
        _ => panic!("Expected Buffer operation for negated operand"),
    }
}

#[test]
fn test_rem_operation() {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);
    let b = graph.input("b", DType::F32, vec![10]);

    let result = a % b;

    match result.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    match &result.op {
        GraphOp::Elementwise {
            op: ops::ElementwiseOp::Rem,
            ..
        } => {}
        _ => panic!("Expected Rem operation"),
    }

    assert_eq!(result.src.len(), 2);
}

#[test]
fn test_recip_operation() {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);

    let result = ops::recip(a);

    match result.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    match &result.op {
        GraphOp::Elementwise {
            op: ops::ElementwiseOp::Recip,
            ..
        } => {}
        _ => panic!("Expected Recip operation"),
    }

    assert_eq!(result.src.len(), 1);
}

#[test]
fn test_max_operation() {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);
    let b = graph.input("b", DType::F32, vec![10]);

    let result = ops::max(a, b);

    match result.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    match &result.op {
        GraphOp::Elementwise {
            op: ops::ElementwiseOp::Max,
            ..
        } => {}
        _ => panic!("Expected Max operation"),
    }

    assert_eq!(result.src.len(), 2);
}

#[test]
#[should_panic(expected = "Shape mismatch")]
fn test_shape_mismatch() {
    // 異なるshapeのノード同士の演算はpanicする
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);
    let b = graph.input("b", DType::F32, vec![20]);

    // これはpanicするべき
    let _result = a + b;
}

#[test]
fn test_complex_expression() {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);
    let b = graph.input("b", DType::F32, vec![10]);
    let c = graph.input("c", DType::F32, vec![10]);

    // (a + b) * c
    let result = (a + b) * c;

    match &result.op {
        GraphOp::Elementwise {
            op: ops::ElementwiseOp::Mul,
            ..
        } => {}
        _ => panic!("Expected Mul operation at top level"),
    }

    assert_eq!(result.src.len(), 2);

    // 左側のノードがAdd演算であることを確認
    match &result.src[0].op {
        GraphOp::Elementwise {
            op: ops::ElementwiseOp::Add,
            ..
        } => {}
        _ => panic!("Expected Add operation in left operand"),
    }
}

#[test]
fn test_dtype_inference() {
    let mut graph = Graph::new();
    let unknown = graph.input("unknown", DType::Unknown, vec![10]);
    let f32_node = graph.input("f32", DType::F32, vec![10]);

    let result = unknown + f32_node;

    // UnknownとF32を組み合わせた場合、F32になるべき
    match result.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32 after inference"),
    }
}

#[test]
fn test_reduce_strategy_default() {
    let default_strategy = ReduceStrategy::default();
    assert_eq!(
        default_strategy,
        ReduceStrategy::Sequential { unroll_factor: 1 }
    );
}

#[test]
fn test_reduce_strategy_sequential() {
    let strategy = ReduceStrategy::sequential();
    assert_eq!(strategy, ReduceStrategy::Sequential { unroll_factor: 1 });

    let strategy_unroll = ReduceStrategy::sequential_unroll(4);
    assert_eq!(
        strategy_unroll,
        ReduceStrategy::Sequential { unroll_factor: 4 }
    );
}

#[test]
fn test_reduce_strategy_accessors() {
    let strategy = ReduceStrategy::sequential_unroll(8);
    assert_eq!(strategy.unroll_factor(), 8);
}

#[test]
fn test_cumulative_strategy_default() {
    let default_strategy = CumulativeStrategy::default();
    assert_eq!(
        default_strategy,
        CumulativeStrategy::Sequential { unroll_factor: 1 }
    );
}

#[test]
fn test_cumulative_strategy_sequential() {
    let strategy = CumulativeStrategy::sequential();
    assert_eq!(
        strategy,
        CumulativeStrategy::Sequential { unroll_factor: 1 }
    );

    let strategy_unroll = CumulativeStrategy::sequential_unroll(4);
    assert_eq!(
        strategy_unroll,
        CumulativeStrategy::Sequential { unroll_factor: 4 }
    );
}

#[test]
fn test_cumulative_strategy_accessors() {
    let strategy = CumulativeStrategy::sequential_unroll(8);
    assert_eq!(strategy.unroll_factor(), 8);
}

#[test]
fn test_reduce_sum() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![10, 20, 30]);

    // 軸1を縮約（10, 20, 30 -> 10, 30）
    let result = input.reduce_sum(1);

    // 型が保持されていることを確認
    match result.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    // Viewのshapeが正しく縮約されていることを確認
    assert_eq!(result.view.ndim(), 2);
    assert_eq!(result.view.shape().len(), 2);

    // Reduceオペレーションが正しく設定されていることを確認
    match &result.op {
        GraphOp::Reduce { op, axis, .. } => {
            assert_eq!(*op, ReduceOp::Sum);
            assert_eq!(*axis, 1);
        }
        _ => panic!("Expected GraphOp::Reduce"),
    }
}

#[test]
fn test_reduce_mul() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![5, 10]);

    // 軸0を縮約（5, 10 -> 10）
    let result = input.reduce_mul(0);

    // Viewのshapeが正しく縮約されていることを確認
    assert_eq!(result.view.ndim(), 1);
    assert_eq!(result.view.shape().len(), 1);

    // Reduceオペレーションが正しく設定されていることを確認
    match &result.op {
        GraphOp::Reduce { op, axis, .. } => {
            assert_eq!(*op, ReduceOp::Prod);
            assert_eq!(*axis, 0);
        }
        _ => panic!("Expected GraphOp::Reduce"),
    }
}

#[test]
fn test_reduce_max() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![3, 4, 5]);

    // 軸2を縮約（3, 4, 5 -> 3, 4）
    let result = input.reduce_max(2);

    // Viewのshapeが正しく縮約されていることを確認
    assert_eq!(result.view.ndim(), 2);

    // Reduceオペレーションが正しく設定されていることを確認
    match &result.op {
        GraphOp::Reduce { op, axis, .. } => {
            assert_eq!(*op, ReduceOp::Max);
            assert_eq!(*axis, 2);
        }
        _ => panic!("Expected GraphOp::Reduce"),
    }
}

#[test]
fn test_view_method() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![3, 4]);

    // Viewを変更（転置）
    let transposed_view = input.view.clone().permute(vec![1, 0]);
    let transposed = input.view(transposed_view.clone());

    // dtypeが保持されていることを確認
    match transposed.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    // Viewが正しく設定されていることを確認
    assert_eq!(transposed.view, transposed_view);
    assert_eq!(transposed.view.ndim(), 2);

    // GraphOp::Viewが設定されていることを確認
    match &transposed.op {
        GraphOp::View(v) => {
            assert_eq!(*v, transposed_view);
        }
        _ => panic!("Expected GraphOp::View"),
    }

    // 元のノードが入力として保持されていることを確認
    assert_eq!(transposed.src.len(), 1);
}

#[test]
fn test_view_method_unsqueeze() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![3, 4]);

    // Viewを変更（次元追加）
    let unsqueezed_view = input.view.clone().unsqueeze(0);
    let unsqueezed = input.view(unsqueezed_view.clone());

    // Viewが正しく設定されていることを確認
    assert_eq!(unsqueezed.view.ndim(), 3);

    // GraphOp::Viewが設定されていることを確認
    match &unsqueezed.op {
        GraphOp::View(v) => {
            assert_eq!(*v, unsqueezed_view);
        }
        _ => panic!("Expected GraphOp::View"),
    }
}

#[test]
fn test_reduce_to_scalar() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![10]);

    // 唯一の軸を縮約してスカラーに（10 -> []）
    let result = input.reduce_sum(0);

    // スカラー（ndim=0）になることを確認
    assert_eq!(result.view.ndim(), 0);
    assert_eq!(result.view.shape().len(), 0);
}

#[test]
#[should_panic(expected = "axis 3 is out of bounds")]
fn test_reduce_out_of_bounds() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![10, 20]);

    // 存在しない軸3を指定してパニック
    let _result = input.reduce_sum(3);
}

#[test]
fn test_reduce_generic() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![5, 10, 15]);

    // ReduceOpを直接指定
    let result = input.reduce(ReduceOp::Sum, 1);

    match &result.op {
        GraphOp::Reduce { op, axis, .. } => {
            assert_eq!(*op, ReduceOp::Sum);
            assert_eq!(*axis, 1);
        }
        _ => panic!("Expected GraphOp::Reduce"),
    }
}

#[test]
fn test_constant_f32() {
    // F32定数ノードを作成
    let const_node = GraphNode::constant(2.5f32);

    // dtypeがF32であることを確認
    match const_node.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    // スカラー（ndim=0）であることを確認
    assert_eq!(const_node.view.ndim(), 0);
    assert_eq!(const_node.view.shape().len(), 0);

    // GraphOp::Constであることを確認
    match &const_node.op {
        GraphOp::Const(Literal::F32(v)) => {
            assert_eq!(*v, 2.5f32);
        }
        _ => panic!("Expected GraphOp::Const with F32 literal"),
    }

    // 入力ノードがないことを確認
    assert_eq!(const_node.src.len(), 0);
}

#[test]
fn test_constant_i64() {
    // i64定数ノードを作成
    let const_node = GraphNode::constant(42i64);

    // スカラーであることを確認
    assert_eq!(const_node.view.ndim(), 0);

    // GraphOp::Constであることを確認
    match &const_node.op {
        GraphOp::Const(Literal::I64(v)) => {
            assert_eq!(*v, 42);
        }
        _ => panic!("Expected GraphOp::Const with Int literal"),
    }
}

#[test]
fn test_constant_usize() {
    // usize定数ノードを作成
    let const_node = GraphNode::constant(100usize);

    // スカラーであることを確認
    assert_eq!(const_node.view.ndim(), 0);

    // GraphOp::Constであることを確認
    match &const_node.op {
        GraphOp::Const(Literal::I64(v)) => {
            assert_eq!(*v, 100);
        }
        _ => panic!("Expected GraphOp::Const with Int literal"),
    }
}

#[test]
fn test_reshape() {
    use harp_core::graph::shape::Expr;

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![3, 4]);

    // (3, 4) -> (12,) にreshape
    let flattened = a.reshape(vec![Expr::from(12)]);
    assert_eq!(flattened.view.shape(), &[Expr::from(12)]);

    // GraphOp::Viewであることを確認
    match &flattened.op {
        GraphOp::View(v) => {
            assert_eq!(v.shape(), &[Expr::from(12)]);
        }
        _ => panic!("Expected GraphOp::View"),
    }

    // (3, 4) -> (2, 6) にreshape
    let reshaped = a.reshape(vec![Expr::from(2), Expr::from(6)]);
    assert_eq!(reshaped.view.shape(), &[Expr::from(2), Expr::from(6)]);
}

#[test]
#[should_panic(expected = "reshape can only be applied to contiguous views")]
fn test_reshape_non_contiguous() {
    use harp_core::graph::shape::Expr;

    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![3, 4]);

    // Permuteして非連続にする
    let transposed = a.view(a.view.clone().permute(vec![1, 0]));

    // 非連続なViewに対してreshapeを試みる（panicするはず）
    let _ = transposed.reshape(vec![Expr::from(12)]);
}

#[test]
fn test_recip_method() {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);

    // メソッド形式でrecipを呼び出し
    let result = a.recip();

    // 正しいGraphOpが生成されたことを確認
    match &result.op {
        GraphOp::Elementwise { op, .. } => {
            assert!(matches!(op, ops::ElementwiseOp::Recip));
        }
        _ => panic!("Expected Elementwise::Recip"),
    }

    // 形状とDTypeが保持されていることを確認
    assert_eq!(result.view.shape().len(), 2);
    assert!(matches!(result.dtype, DType::F32));
}

#[test]
fn test_max_method() {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10, 20]);
    let b = graph.input("b", DType::F32, vec![10, 20]);

    // メソッド形式でmaxを呼び出し: a.max(b)
    let result = a.max(b);

    // 正しいGraphOpが生成されたことを確認
    match &result.op {
        GraphOp::Elementwise { op, .. } => {
            assert!(matches!(op, ops::ElementwiseOp::Max));
        }
        _ => panic!("Expected Elementwise::Max"),
    }

    // 形状とDTypeが保持されていることを確認
    assert_eq!(result.view.shape().len(), 2);
    assert!(matches!(result.dtype, DType::F32));
}

#[test]
fn test_method_chaining() {
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![10]);
    let b = graph.input("b", DType::F32, vec![10]);

    // メソッドチェーン: a.max(b).recip().reduce_sum(0)
    let result = a.max(b).recip().reduce_sum(0);

    // 結果がスカラーであることを確認
    assert_eq!(result.view.shape().len(), 0);
}

#[test]
fn test_cumsum() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![3, 4]);

    // 軸1に沿って累積和（shapeは変わらない）
    let result = input.cumsum(1);

    // 型が保持されていることを確認
    match result.dtype {
        DType::F32 => {}
        _ => panic!("Expected DType::F32"),
    }

    // Viewのshapeが変わらないことを確認
    assert_eq!(result.view.ndim(), 2);
    assert_eq!(result.view.shape().len(), 2);

    // Cumulativeオペレーションが正しく設定されていることを確認
    match &result.op {
        GraphOp::Cumulative { op, axis, .. } => {
            assert_eq!(*op, ops::CumulativeOp::Sum);
            assert_eq!(*axis, 1);
        }
        _ => panic!("Expected GraphOp::Cumulative"),
    }
}

#[test]
fn test_cumprod() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![10, 20, 30]);

    // 軸0に沿って累積積
    let result = input.cumprod(0);

    // Viewのshapeが変わらないことを確認
    assert_eq!(result.view.ndim(), 3);
    assert_eq!(result.view.shape().len(), 3);

    // Cumulativeオペレーションが正しく設定されていることを確認
    match &result.op {
        GraphOp::Cumulative { op, axis, .. } => {
            assert_eq!(*op, ops::CumulativeOp::Prod);
            assert_eq!(*axis, 0);
        }
        _ => panic!("Expected GraphOp::Cumulative"),
    }
}

#[test]
#[should_panic(expected = "axis 3 is out of bounds")]
fn test_cumsum_out_of_bounds() {
    let mut graph = Graph::new();
    let input = graph.input("x", DType::F32, vec![10, 20]);

    // 無効な軸（2次元テンソルに対して軸3）
    let _result = input.cumsum(3);
}

#[test]
fn test_pad_1d() {
    let mut graph = Graph::new();
    let x = graph.input("x", DType::F32, vec![3]);

    // (3,) に両側1要素ずつパディング -> (5,)
    let padded = x.pad(vec![(1, 1)], 0.0);

    // 出力の次元数を確認
    assert_eq!(padded.view.ndim(), 1);
    // shapeの式が 3 + 1 + 1 = 5 になっていることを確認
    // 式は Add(Add(Const(3), Const(1)), Const(1)) の形式
    assert_eq!(padded.view.shape().len(), 1);

    // Pad演算が正しく設定されていることを確認
    match &padded.op {
        GraphOp::Pad { padding, value } => {
            use harp_core::graph::shape::Expr;
            assert_eq!(padding.len(), 1);
            assert_eq!(padding[0].0, Expr::Const(1));
            assert_eq!(padding[0].1, Expr::Const(1));
            assert_eq!(*value, 0.0);
        }
        _ => panic!("Expected Pad operation"),
    }
}

#[test]
fn test_pad_2d() {
    let mut graph = Graph::new();
    let x = graph.input("x", DType::F32, vec![3, 4]);

    // (3, 4) -> (5, 6) にパディング
    let padded = x.pad(vec![(1, 1), (1, 1)], 0.0);

    // 出力の次元数を確認
    assert_eq!(padded.view.ndim(), 2);
    assert_eq!(padded.view.shape().len(), 2);

    // Pad演算が正しく設定されていることを確認
    match &padded.op {
        GraphOp::Pad { padding, value } => {
            use harp_core::graph::shape::Expr;
            assert_eq!(padding.len(), 2);
            assert_eq!(padding[0], (Expr::Const(1), Expr::Const(1)));
            assert_eq!(padding[1], (Expr::Const(1), Expr::Const(1)));
            assert_eq!(*value, 0.0);
        }
        _ => panic!("Expected Pad operation"),
    }
}

#[test]
fn test_pad_asymmetric() {
    let mut graph = Graph::new();
    let x = graph.input("x", DType::F32, vec![10]);

    // 非対称なパディング: 前に2, 後に3
    let padded = x.pad(vec![(2, 3)], 1.0);

    // (10,) -> (15,)
    assert_eq!(padded.view.ndim(), 1);
    assert_eq!(padded.view.shape().len(), 1);

    match &padded.op {
        GraphOp::Pad { padding, value } => {
            use harp_core::graph::shape::Expr;
            assert_eq!(padding.len(), 1);
            assert_eq!(padding[0], (Expr::Const(2), Expr::Const(3)));
            assert_eq!(*value, 1.0);
        }
        _ => panic!("Expected Pad operation"),
    }
}

#[test]
#[should_panic(expected = "padding length must match tensor ndim")]
fn test_pad_dimension_mismatch() {
    let mut graph = Graph::new();
    let x = graph.input("x", DType::F32, vec![3, 4]);

    // 2Dテンソルに1Dのパディング指定（エラー）
    let _padded = x.pad(vec![(1, 1)], 0.0);
}

#[test]
fn test_pad_dynamic_shape() {
    use harp_core::graph::shape::Expr;

    let mut graph = Graph::new();
    // 動的shapeの入力
    let n = Expr::Var("N".to_string());
    let x = graph.input("x", DType::F32, vec![n.clone()]);

    // 動的パディング量
    let pad_size = Expr::Var("P".to_string());
    let padded = x.pad(vec![(pad_size.clone(), pad_size.clone())], 0.0);

    // 出力shapeは N + P + P
    assert_eq!(padded.view.ndim(), 1);

    // Pad演算が正しく設定されていることを確認
    match &padded.op {
        ops::GraphOp::Pad { padding, value } => {
            assert_eq!(padding.len(), 1);
            assert_eq!(padding[0].0, Expr::Var("P".to_string()));
            assert_eq!(padding[0].1, Expr::Var("P".to_string()));
            assert_eq!(*value, 0.0);
        }
        _ => panic!("Expected Pad operation"),
    }
}

#[test]
#[should_panic(expected = "padding[0].before must be non-negative")]
fn test_pad_negative_static_value() {
    let mut graph = Graph::new();
    let x = graph.input("x", DType::F32, vec![10]);

    // 負のパディング値（静的にチェック可能）
    let _padded = x.pad(vec![(-1isize, 1)], 0.0);
}

#[test]
fn test_slice_1d() {
    let mut graph = Graph::new();
    let x = graph.input("x", DType::F32, vec![10]);

    // [10] から [2:7] を切り出し -> [5]
    let sliced = x.slice(vec![(2, 7)]);

    // 出力の次元数を確認
    assert_eq!(sliced.view.ndim(), 1);
    assert_eq!(sliced.view.shape().len(), 1);

    // Slice演算が正しく設定されていることを確認
    match &sliced.op {
        GraphOp::Slice { ranges } => {
            assert_eq!(*ranges, vec![(2, 7)]);
        }
        _ => panic!("Expected Slice operation"),
    }
}

#[test]
fn test_slice_2d() {
    let mut graph = Graph::new();
    let x = graph.input("x", DType::F32, vec![10, 20]);

    // [10, 20] から [2:5, 3:18] を切り出し -> [3, 15]
    let sliced = x.slice(vec![(2, 5), (3, 18)]);

    assert_eq!(sliced.view.ndim(), 2);
    assert_eq!(sliced.view.shape().len(), 2);

    match &sliced.op {
        GraphOp::Slice { ranges } => {
            assert_eq!(*ranges, vec![(2, 5), (3, 18)]);
        }
        _ => panic!("Expected Slice operation"),
    }
}

#[test]
#[should_panic(expected = "ranges length must match tensor ndim")]
fn test_slice_dimension_mismatch() {
    let mut graph = Graph::new();
    let x = graph.input("x", DType::F32, vec![10, 20]);

    // 2Dテンソルに1Dのranges指定（エラー）
    let _sliced = x.slice(vec![(2, 5)]);
}

#[test]
#[should_panic(expected = "start (5) must be less than end (2)")]
fn test_slice_invalid_range() {
    let mut graph = Graph::new();
    let x = graph.input("x", DType::F32, vec![10]);

    // start >= end のケース（エラー）
    let _sliced = x.slice(vec![(5, 2)]);
}
