use harp::backend::c::{CBackend, CBuffer};
use harp::backend::{Backend, Buffer, Kernel};
use harp::graph::shape::Expr as ShapeExpr;
use harp::graph::{Graph, ShapeVariableSignature};
use harp::ast::DType;
use ndarray::{Array, Array2, ArrayD};

fn main() {
    let _ = env_logger::builder().is_test(true).try_init();
    // 1. 計算グラフを定義する
    let mut graph = Graph::new();
    graph
        .signature
        .shape_variables
        .push(ShapeVariableSignature {
            name: "M".to_string(),
            condition: ShapeExpr::gt(ShapeExpr::var("M"), ShapeExpr::from(0)),
            default: 64,
        });
    graph
        .signature
        .shape_variables
        .push(ShapeVariableSignature {
            name: "K".to_string(),
            condition: ShapeExpr::gt(ShapeExpr::var("K"), ShapeExpr::from(0)),
            default: 128,
        });
    graph
        .signature
        .shape_variables
        .push(ShapeVariableSignature {
            name: "N".to_string(),
            condition: ShapeExpr::gt(ShapeExpr::var("N"), ShapeExpr::from(0)),
            default: 256,
        });

    // 行列のシェイプを定義する
    let m = ShapeExpr::var("M");
    let k = ShapeExpr::var("K");
    let n = ShapeExpr::var("N");

    // 入力テンソルを定義する
    let a = graph.add_input(vec![m.clone(), k.clone()], &DType::F32);
    let b = graph.add_input(vec![k.clone(), n.clone()], &DType::F32);

    // 行列乗算の実装
    let expanded_shape = vec![m.clone(), k.clone(), n.clone()];
    // A(M, K) -> A(M, K, 1) -> A(M, K, N)
    let a_expanded = a.unsqueeze(2).expand(expanded_shape.clone());
    // B(K, N) -> B(1, K, N) -> B(M, K, N)
    let b_expanded = b.unsqueeze(0).expand(expanded_shape);
    // C(M, N) = sum(A(M, K, N) * B(M, K, N), axis=1)
    let c = (a_expanded * b_expanded).sum(1);

    graph.outputs.push(c);

    // 2. バックエンドを選択して計算グラフをコンパイルする
    let mut backend = CBackend::new();
    let mut kernel = backend.compile(&graph);

    // 3. 実際のデータを準備する
    let m_val = 64;
    let k_val = 128;
    let n_val = 256;

    let a_data: Array2<f32> = Array::from_shape_fn((m_val, k_val), |(i, j)| (i * k_val + j) as f32);
    let b_data: Array2<f32> = Array::from_shape_fn((k_val, n_val), |(i, j)| (i * n_val + j) as f32);

    let a_buffer = CBuffer::from_slice(
        a_data.as_slice().unwrap(),
        &[m_val, k_val],
        DType::F32,
    );
    let b_buffer = CBuffer::from_slice(
        b_data.as_slice().unwrap(),
        &[k_val, n_val],
        DType::F32,
    );
    let c_buffer = CBuffer::allocate(DType::F32, vec![m_val, n_val]);

    let mut shape_vars_values = vec![];
    for var in &kernel.details().shape_variables {
        match var.name.as_str() {
            "M" => shape_vars_values.push(m_val),
            "K" => shape_vars_values.push(k_val),
            "N" => shape_vars_values.push(n_val),
            _ => panic!("Unknown shape variable"),
        }
    }

    // 4. カーネルを実行する
        let result_buffers = kernel.call(vec![c_buffer, a_buffer, b_buffer], &shape_vars_values);
    let c_vec = result_buffers[0].to_vec::<f32>();
    let c_data = ArrayD::from_shape_vec(vec![m_val, n_val], c_vec).unwrap();

    // 5. 結果を検証する
    let c_ndarray = a_data.dot(&b_data);
    let difference = (&c_data.into_dimensionality::<ndarray::Ix2>().unwrap() - &c_ndarray)
        .mapv(|x| x.abs())
        .sum();

    println!("Harp matmul result shape: {:?}", [m_val, n_val]);
    println!("ndarray matmul result shape: {:?}", c_ndarray.shape());
    println!("Total difference: {}", difference);

    assert!(difference < 1e-3);
    println!("Matrix multiplication test passed!");
}
