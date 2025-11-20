//! 自動微分のテスト

use super::Tensor;
use crate::graph::{DType, Graph};

#[test]
fn test_tensor_creation() {
    let mut graph = Graph::new();
    let node = graph
        .input("x")
        .with_dtype(DType::F32)
        .with_shape([2, 3])
        .build();

    let tensor = Tensor::from_graph_node(node, true);
    assert!(tensor.requires_grad());
    assert!(tensor.grad().is_none());
}

#[test]
fn test_simple_add() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let c = &a + &b;
    assert!(c.requires_grad());
}

#[test]
fn test_simple_mul() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let c = &a * &b;
    assert!(c.requires_grad());
}

#[test]
fn test_backward_add() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    // c = a + b
    let c = &a + &b;

    // loss = sum(c)
    let loss = c.sum(0);

    // backward
    loss.backward();

    // 勾配が計算されているか確認
    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
}

#[test]
fn test_backward_mul() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    // c = a * b
    let c = &a * &b;

    // loss = sum(c)
    let loss = c.sum(0);

    // backward
    loss.backward();

    // 勾配が計算されているか確認
    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
}

#[test]
fn test_backward_complex() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([4])
            .build(),
        true,
    );

    // y = 2 * x + 1
    let y = 2.0 * &x + 1.0;

    // loss = sum(y)
    let loss = y.sum(0);

    // backward
    loss.backward();

    // xの勾配が計算されているか確認
    assert!(x.grad().is_some());
}

#[test]
fn test_zero_grad() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = &x * 2.0;
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());

    x.zero_grad();
    assert!(x.grad().is_none());
}

#[test]
fn test_no_grad() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        false, // requires_grad=false
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let c = &a + &b;
    assert!(c.requires_grad()); // bがrequires_grad=trueなので

    let loss = c.sum(0);
    loss.backward();

    // aは勾配計算されない
    assert!(a.grad().is_none());
    // bは勾配計算される
    assert!(b.grad().is_some());
}

#[test]
fn test_recip() {
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(
        graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    let y = x.recip();
    let loss = y.sum(0);
    loss.backward();

    assert!(x.grad().is_some());
}

#[test]
fn test_div() {
    let mut graph = Graph::new();
    let a = Tensor::from_graph_node(
        graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );
    let b = Tensor::from_graph_node(
        graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape([3])
            .build(),
        true,
    );

    // c = a / b
    let c = &a / &b;
    let loss = c.sum(0);
    loss.backward();

    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
}
