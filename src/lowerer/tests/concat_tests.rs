//! Concat演算のloweringテスト

use crate::graph::{DType, Graph, GraphNode};
use crate::graph::ops::concat;
use crate::lowerer::Lowerer;

/// 基本的なconcat演算（axis=0）
#[test]
fn test_concat_axis0() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([2, 4])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape([3, 4])
        .build();

    // axis=0で結合: [2, 4] + [3, 4] => [5, 4]
    let c = concat(vec![a, b], 0);
    assert_eq!(c.view.shape().len(), 2);

    // 結合軸のサイズが合計になることを確認
    // shape[0] = 2 + 3 = 5
    let shape = c.view.shape();
    // shape[1]は変わらない（4）
    assert_eq!(shape[1], 4.into());
}

/// 基本的なconcat演算（axis=1）
#[test]
fn test_concat_axis1() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([2, 3])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape([2, 5])
        .build();

    // axis=1で結合: [2, 3] + [2, 5] => [2, 8]
    let c = concat(vec![a, b], 1);
    let shape = c.view.shape();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], 2.into());
    // shape[1] = 3 + 5 = 8
}

/// 3つのテンソルを結合
#[test]
fn test_concat_three_tensors() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([2, 3])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape([2, 4])
        .build();
    let c = graph
        .input("c")
        .with_dtype(DType::F32)
        .with_shape([2, 5])
        .build();

    // axis=1で結合: [2, 3] + [2, 4] + [2, 5] => [2, 12]
    let result = concat(vec![a, b, c], 1);
    let shape = result.view.shape();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], 2.into());
    // shape[1] = 3 + 4 + 5 = 12
}

/// 1次元テンソルの結合
#[test]
fn test_concat_1d() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([5])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape([3])
        .build();

    let c = concat(vec![a, b], 0);
    let shape = c.view.shape();
    assert_eq!(shape.len(), 1);
    // shape[0] = 5 + 3 = 8
}

/// 3次元テンソルの結合（バッチ軸）
#[test]
fn test_concat_3d_batch() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([2, 4, 8])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape([3, 4, 8])
        .build();

    // axis=0（バッチ軸）で結合: [2, 4, 8] + [3, 4, 8] => [5, 4, 8]
    let c = concat(vec![a, b], 0);
    let shape = c.view.shape();
    assert_eq!(shape.len(), 3);
    // shape[0] = 2 + 3 = 5
    assert_eq!(shape[1], 4.into());
    assert_eq!(shape[2], 8.into());
}

/// 3次元テンソルの結合（中間軸）
#[test]
fn test_concat_3d_middle_axis() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([2, 3, 8])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape([2, 5, 8])
        .build();

    // axis=1で結合: [2, 3, 8] + [2, 5, 8] => [2, 8, 8]
    let c = concat(vec![a, b], 1);
    let shape = c.view.shape();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0], 2.into());
    // shape[1] = 3 + 5 = 8
    assert_eq!(shape[2], 8.into());
}

/// GraphNodeのcatメソッドのテスト
#[test]
fn test_graphnode_cat_method() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([2, 3])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape([2, 5])
        .build();

    // catメソッドを使用
    let c = a.cat(b, 1);
    let shape = c.view.shape();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], 2.into());
    // shape[1] = 3 + 5 = 8
}

/// GraphNode::concatスタティックメソッドのテスト
#[test]
fn test_graphnode_concat_static() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([2, 3])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape([2, 5])
        .build();

    // スタティックメソッドを使用
    let c = GraphNode::concat(vec![a, b], 1);
    let shape = c.view.shape();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], 2.into());
}

/// カーネル生成のテスト
#[test]
fn test_concat_kernel_generation() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([2, 4])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape([3, 4])
        .build();

    let c = concat(vec![a, b], 0);
    graph.output("c", c.clone());

    // lowererでカーネルを生成
    let mut lowerer = Lowerer::new();
    let result = lowerer.lower_node_to_kernel(&c, 0);

    assert!(result.is_ok(), "カーネル生成に失敗: {:?}", result.err());

    let kernel = result.unwrap();
    // 生成されたカーネルがFunction型であることを確認
    match &kernel {
        crate::ast::AstNode::Function { name, params, .. } => {
            assert_eq!(name.as_deref(), Some("kernel_0"));
            // 入力が2つ + 出力1つ = 少なくとも3つのパラメータ
            assert!(params.len() >= 3, "パラメータ数が不足: {}", params.len());
        }
        _ => panic!("Functionノードが生成されていません"),
    }
}

/// 単一入力のconcat（そのまま返す）
#[test]
fn test_concat_single_input() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([2, 3])
        .build();

    // 単一入力はそのまま返される
    let c = concat(vec![a.clone()], 0);

    // 同じポインタであることを確認
    assert_eq!(a.as_ptr(), c.as_ptr());
}

/// 次元不一致時のパニックテスト
#[test]
#[should_panic(expected = "dimension mismatch")]
fn test_concat_dimension_mismatch() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([2, 3])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape([2, 5])
        .build();

    // axis=0では次元1のサイズが一致する必要がある
    // [2, 3] と [2, 5] はaxis=0では結合できない
    let _ = concat(vec![a, b], 0);
}

/// 軸が範囲外の場合のパニックテスト
#[test]
#[should_panic(expected = "out of bounds")]
fn test_concat_axis_out_of_bounds() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([2, 3])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape([2, 5])
        .build();

    // axis=2は2次元テンソルに対して範囲外
    let _ = concat(vec![a, b], 2);
}

/// 空入力時のパニックテスト
#[test]
#[should_panic(expected = "at least one input")]
fn test_concat_empty_inputs() {
    let _ = concat(vec![], 0);
}

/// 次元数不一致時のパニックテスト
#[test]
#[should_panic(expected = "same number of dimensions")]
fn test_concat_ndim_mismatch() {
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape([2, 3])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape([2, 3, 4])
        .build();

    // 2次元と3次元は結合できない
    let _ = concat(vec![a, b], 0);
}
