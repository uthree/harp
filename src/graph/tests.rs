//! Integration tests for the graph module

use super::*;
use crate::ast::DType;

#[test]
fn test_simple_computation_graph() {
    // Create input tensors
    let a = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("a");
    let b = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("b");

    // Build computation graph
    let c = (&a + &b).with_name("c");
    let d = c.sum(1).with_name("d");

    // Verify structure
    assert_eq!(d.ndim(), 2);
    assert_eq!(d.sources().len(), 1);

    // Verify inputs
    let inputs = collect_inputs(&[d.clone()]);
    assert_eq!(inputs.len(), 2);
}

#[test]
fn test_complex_graph() {
    let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
    let y = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);

    // Complex expression: (x + y) * x - y.sqrt()
    let z = (&(&x + &y) * &x) - &y.sqrt();

    // Count nodes
    let count = count_nodes(&[z.clone()]);
    assert!(count >= 4); // At least x, y, and intermediate nodes
}

#[test]
fn test_view_transformations() {
    let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);

    // Test permute
    let t = x.permute(&[1, 0]);
    assert_eq!(t.shape()[0], Expr::Const(64));
    assert_eq!(t.shape()[1], Expr::Const(32));

    // Test unsqueeze
    let u = x.unsqueeze(0);
    assert_eq!(u.ndim(), 3);

    // Test reshape
    let r = x.reshape(vec![Expr::Const(2048)]);
    assert_eq!(r.ndim(), 1);
}

#[test]
fn test_reductions() {
    let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);

    // Sum along axis 1
    let s = x.sum(1);
    assert_eq!(s.shape()[1], Expr::Const(1));

    // Max along axis 0
    let m = x.max(0);
    assert_eq!(m.shape()[0], Expr::Const(1));
}

#[test]
fn test_topological_order() {
    let a = input(vec![Expr::Const(32)], DType::F32).with_name("a");
    let b = input(vec![Expr::Const(32)], DType::F32).with_name("b");
    let c = (&a + &b).with_name("c");
    let d = (&c * &a).with_name("d");

    let sorted = topological_sort(&[d.clone()]);

    // Find positions
    let get_pos = |name: &str| sorted.iter().position(|n| n.name() == Some(name)).unwrap();

    // Verify order: a, b before c; c, a before d
    assert!(get_pos("a") < get_pos("c"));
    assert!(get_pos("b") < get_pos("c"));
    assert!(get_pos("c") < get_pos("d"));
}

#[test]
fn test_graph_builder() {
    let x = GraphNodeBuilder::new()
        .shape_const(&[16, 32, 64])
        .dtype(DType::F64)
        .name("my_tensor")
        .build_input();

    assert_eq!(x.ndim(), 3);
    assert_eq!(x.dtype(), &DType::F64);
    assert_eq!(x.name(), Some("my_tensor"));
}

#[test]
fn test_math_functions() {
    let x = input(vec![Expr::Const(32)], DType::F32);

    // Test various math functions
    let _ = x.sqrt();
    let _ = x.recip();
    let _ = x.log2();
    let _ = x.exp2();
    let _ = x.sin();
    let _ = x.floor();
    let _ = x.ln();
    let _ = x.exp();
    let _ = x.cos();
    let _ = x.abs();
}

#[test]
fn test_comparison_ops() {
    let a = input(vec![Expr::Const(32)], DType::F32);
    let b = input(vec![Expr::Const(32)], DType::F32);

    let lt = a.lt(&b);
    assert_eq!(lt.dtype(), &DType::Bool);

    let gt = a.gt(&b);
    assert_eq!(gt.dtype(), &DType::Bool);

    let eq = a.eq_node(&b);
    assert_eq!(eq.dtype(), &DType::Bool);
}

#[test]
fn test_where_cond() {
    let cond = input(vec![Expr::Const(32)], DType::Bool);
    let a = input(vec![Expr::Const(32)], DType::F32);
    let b = input(vec![Expr::Const(32)], DType::F32);

    let result = a.where_cond(&cond, &b);
    assert_eq!(result.sources().len(), 3);
    assert_eq!(result.dtype(), &DType::F32);
}

#[test]
fn test_clamp() {
    let x = input(vec![Expr::Const(32)], DType::F32);
    let min_val = input(vec![Expr::Const(32)], DType::F32);
    let max_val = input(vec![Expr::Const(32)], DType::F32);

    let clamped = x.clamp(&min_val, &max_val);
    assert_eq!(clamped.dtype(), &DType::F32);
}

#[test]
fn test_cast() {
    let x = input(vec![Expr::Const(32)], DType::F32);
    let y = x.cast(DType::F64);

    assert_eq!(y.dtype(), &DType::F64);
}
