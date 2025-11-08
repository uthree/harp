use super::super::recursive::*;
use crate::ast::DType;
use crate::graph::ops::cumulative::CumulativeOps;
use crate::graph::ops::reduce::ReduceOps;
use crate::graph::{Graph, GraphNode};

#[test]
fn test_new_lowerer() {
    let lowerer = RecursiveLowerer::new();
    // 公開フィールドが空であることを確認
    assert!(lowerer.declarations.is_empty());
    assert!(lowerer.statements.is_empty());
}

#[test]
fn test_set_var_name() {
    let mut lowerer = RecursiveLowerer::new();
    let mut graph = Graph::new();
    let input = graph.input(DType::F32, vec![10.into()]);

    lowerer.set_var_name(&input, "input_0".to_string());
    assert_eq!(lowerer.get_var_name(&input), Some("input_0".to_string()));
}

#[test]
fn test_var_name_mapping() {
    let mut lowerer = RecursiveLowerer::new();
    let mut graph = Graph::new();
    let input = graph.input(DType::F32, vec![10.into()]);

    // loweringを実行すると変数名がマッピングされる
    lowerer.lower_node(&input);

    // 同じノードは同じ変数名を持つ
    let var1 = lowerer.get_var_name(&input);
    assert!(var1.is_some());

    // 2回目のloweringでも同じ変数名
    lowerer.lower_node(&input);
    let var2 = lowerer.get_var_name(&input);
    assert_eq!(var1, var2);

    // 別のノードは別の変数名を持つ
    let input2 = graph.input(DType::F32, vec![10.into()]);
    lowerer.lower_node(&input2);
    let var3 = lowerer.get_var_name(&input2);
    assert!(var3.is_some());
    assert_ne!(var1, var3);
}

#[test]
fn test_lower_const_node() {
    let mut lowerer = RecursiveLowerer::new();
    let const_node = GraphNode::f32(42.0);

    let ast = lowerer.lower_node(&const_node);

    // ASTが生成されたことを確認（空のBlockではない）
    assert!(matches!(ast, AstNode::Assign(_, _)));

    // 変数宣言が追加されたことを確認
    assert_eq!(lowerer.declarations.len(), 1);
    assert_eq!(lowerer.declarations[0].name, "temp0");
    assert_eq!(lowerer.declarations[0].dtype, DType::F32);

    // ステートメントが追加されたことを確認
    assert_eq!(lowerer.statements.len(), 1);
}

#[test]
fn test_lower_input_node() {
    let mut lowerer = RecursiveLowerer::new();
    let mut graph = Graph::new();
    let input = graph.input(DType::F32, vec![10.into()]);

    let ast = lowerer.lower_node(&input);

    // 入力ノードはステートメントを生成しない（空のBlock）
    assert!(matches!(
        ast,
        AstNode::Block {
            statements,
            ..
        } if statements.is_empty()
    ));

    // しかし変数名はマッピングされる
    assert!(lowerer.get_var_name(&input).is_some());
}

#[test]
fn test_memoization() {
    let mut lowerer = RecursiveLowerer::new();
    let const_node = GraphNode::f32(42.0);

    // 1回目のlower
    let ast1 = lowerer.lower_node(&const_node);

    // キャッシュに保存されたことを確認
    assert!(lowerer.cache.contains_key(&const_node));

    // 2回目のlower（キャッシュから取得）
    let ast2 = lowerer.lower_node(&const_node);

    // 同じASTが返されることを確認
    assert_eq!(format!("{:?}", ast1), format!("{:?}", ast2));

    // ステートメントは1回だけ追加される
    assert_eq!(lowerer.statements.len(), 1);
}

#[test]
fn test_view_node_shares_var() {
    let mut lowerer = RecursiveLowerer::new();
    let mut graph = Graph::new();
    let input = graph.input(DType::F32, vec![10.into(), 20.into()]);

    // Viewノードを作成（permuteで軸を入れ替え）
    let view_node = input.clone().permute(vec![1, 0]);

    // lowering
    lowerer.lower_node(&view_node);

    // Viewノードとソースノードが同じ変数名を持つことを確認
    let input_var = lowerer.get_var_name(&input).unwrap();
    let view_var = lowerer.get_var_name(&view_node).unwrap();
    assert_eq!(input_var, view_var);
}

// 統合テスト：様々なGraphOpタイプのloweringをテスト

#[test]
fn test_elementwise_binary_op() {
    let mut lowerer = RecursiveLowerer::new();
    let mut graph = Graph::new();

    // 二項演算: input1 + input2
    let input1 = graph.input(DType::F32, vec![10.into()]);
    let input2 = graph.input(DType::F32, vec![10.into()]);
    let result = input1 + input2;

    // lowering実行
    lowerer.lower_node(&result);

    // ステートメントが生成されたことを確認（加算ループ）
    assert!(!lowerer.statements.is_empty());

    // 結果ノードに変数名がマッピングされていることを確認
    assert!(lowerer.get_var_name(&result).is_some());

    // 変数宣言が追加されていることを確認（result変数）
    assert!(!lowerer.declarations.is_empty());
}

#[test]
fn test_reduce_op() {
    let mut lowerer = RecursiveLowerer::new();
    let mut graph = Graph::new();

    // Reduce操作: sum along axis 0
    let input = graph.input(DType::F32, vec![10.into(), 20.into()]);
    let result = input.sum(0);

    // lowering実行
    lowerer.lower_node(&result);

    // ステートメントが生成されたことを確認（reduceループ）
    assert!(!lowerer.statements.is_empty());

    // 結果ノードに変数名がマッピングされていることを確認
    assert!(lowerer.get_var_name(&result).is_some());
}

#[test]
fn test_cast_op() {
    let mut lowerer = RecursiveLowerer::new();
    let mut graph = Graph::new();

    // Cast操作: F32 -> Isize
    let input = graph.input(DType::F32, vec![10.into()]);
    let result = input.cast(DType::Isize);

    // lowering実行
    lowerer.lower_node(&result);

    // ステートメントが生成されたことを確認（castループ）
    assert!(!lowerer.statements.is_empty());

    // 結果ノードに変数名がマッピングされていることを確認
    assert!(lowerer.get_var_name(&result).is_some());

    // 型がIsizeであることを確認
    assert_eq!(result.dtype, DType::Isize);
}

#[test]
fn test_complex_graph() {
    let mut lowerer = RecursiveLowerer::new();
    let mut graph = Graph::new();

    // 複雑なグラフ: (input1 + input2) * constant
    let input1 = graph.input(DType::F32, vec![5.into(), 10.into()]);
    let input2 = graph.input(DType::F32, vec![5.into(), 10.into()]);
    let constant = GraphNode::f32(2.0);

    let sum = input1 + input2;
    let result = sum.clone() * constant.clone();

    // lowering実行
    lowerer.lower_node(&result);

    // 複数のステートメントが生成されたことを確認
    // (constant代入、sum計算、mul計算)
    assert!(lowerer.statements.len() >= 2);

    // すべての中間ノードに変数名がマッピングされていることを確認
    assert!(lowerer.get_var_name(&constant).is_some());
    assert!(lowerer.get_var_name(&sum).is_some());
    assert!(lowerer.get_var_name(&result).is_some());
}

#[test]
fn test_cumulative_op() {
    let mut lowerer = RecursiveLowerer::new();
    let mut graph = Graph::new();

    // Cumulative操作: cumsum along axis 0
    let input = graph.input(DType::F32, vec![10.into(), 20.into()]);
    let result = input.cumsum(0);

    // lowering実行
    lowerer.lower_node(&result);

    // ステートメントが生成されたことを確認（cumulativeループ）
    assert!(!lowerer.statements.is_empty());

    // 結果ノードに変数名がマッピングされていることを確認
    assert!(lowerer.get_var_name(&result).is_some());
}

#[test]
fn test_contiguous_op() {
    let mut lowerer = RecursiveLowerer::new();
    let mut graph = Graph::new();

    // Contiguous操作: permuteした後にcontiguousにする
    let input = graph.input(DType::F32, vec![10.into(), 20.into()]);
    let permuted = input.permute(vec![1, 0]);
    let result = permuted.contiguous();

    // lowering実行
    lowerer.lower_node(&result);

    // ステートメントが生成されたことを確認（copyループ）
    assert!(!lowerer.statements.is_empty());

    // 結果ノードに変数名がマッピングされていることを確認
    assert!(lowerer.get_var_name(&result).is_some());
}
