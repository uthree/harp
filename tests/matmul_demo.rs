/// 行列積のデモ: View操作、Elementwise演算、Reduce演算を組み合わせて実装
///
/// この統合テストは、Harpの公開APIのみを使用して、行列積を実装する方法を示します。
///
/// 行列積 C = A @ B を以下のように計算：
/// 1. A[m,n] → A[m,n,1] (unsqueeze)
/// 2. B[n,p] → B[n,p,1] (unsqueeze) → B[1,n,p] (permute)
/// 3. A, Bを[m,n,p]に拡張 (expand)
/// 4. 要素ごとに乗算 (elementwise mul)
/// 5. 軸1(n軸)で縮約 (reduce sum) → C[m,p]
use harp::prelude::*;

#[test]
fn test_matmul_demo() {
    let mut graph = Graph::new();

    // A: [3, 4] 行列
    let a = graph.input("A", DType::F32, vec![3, 4]);

    // B: [4, 5] 行列
    let b = graph.input("B", DType::F32, vec![4, 5]);

    // 1. A[3,4] → A[3,4,1] (unsqueezeで次元追加)
    let a_view_expanded = a.view.clone().unsqueeze(2); // [3, 4, 1]
    let a_unsqueezed = a.view(a_view_expanded.clone());

    // 2. B[4,5] → B[4,5,1] (unsqueeze) → B[1,4,5] (permute)
    let b_view_unsqueezed = b.view.clone().unsqueeze(2); // [4, 5, 1]
    let b_unsqueezed = b.view(b_view_unsqueezed.clone());

    let b_view_permuted = b_view_unsqueezed.permute(vec![2, 0, 1]); // [1, 4, 5]
    let b_permuted = b_unsqueezed.view(b_view_permuted.clone());

    // 3. A[3,4,1] → A[3,4,5] (expand)
    let a_view_broadcast =
        a_view_expanded.expand(vec![Expr::from(3), Expr::from(4), Expr::from(5)]); // [3, 4, 5]
    let a_broadcast = a_unsqueezed.view(a_view_broadcast.clone());

    // 4. B[1,4,5] → B[3,4,5] (expand)
    let b_view_broadcast =
        b_view_permuted.expand(vec![Expr::from(3), Expr::from(4), Expr::from(5)]); // [3, 4, 5]
    let b_broadcast = b_permuted.view(b_view_broadcast.clone());

    // 5. 要素ごとの乗算: [3, 4, 5]
    let elementwise_product = a_broadcast * b_broadcast;

    // 6. 軸1(k軸)で縮約: [3, 4, 5] → [3, 5]
    let matmul_result = elementwise_product.reduce_sum(1);

    // 結果をグラフに登録
    graph.output("C", matmul_result.clone());

    // 検証: 出力の形状が [3, 5] であることを確認
    assert_eq!(matmul_result.view.ndim(), 2);
    let result_shape = matmul_result.view.shape();
    assert_eq!(result_shape[0], Expr::from(3));
    assert_eq!(result_shape[1], Expr::from(5));

    // 検証: 最後の演算がReduceであることを確認
    match &matmul_result.op {
        GraphOp::Reduce { op, axis, .. } => {
            assert_eq!(*op, ReduceOp::Sum);
            assert_eq!(*axis, 1); // k軸で縮約
        }
        _ => panic!("Expected GraphOp::Reduce"),
    }

    // グラフの出力が正しく登録されていることを確認
    assert_eq!(graph.outputs().len(), 1);
    assert!(graph.outputs().contains_key("C"));

    println!("✓ 行列積デモ成功: A[3,4] @ B[4,5] = C[3,5]");
}
