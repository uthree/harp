//! Lowererユーティリティ
//!
//! グラフからシグネチャを生成するためのユーティリティ関数を提供します。

use crate::backend::KernelSignature;
use crate::graph::Graph;
use std::collections::HashSet;

/// GraphからKernelSignatureを生成
pub fn create_signature(graph: &Graph) -> KernelSignature {
    use crate::backend::BufferSignature;

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut shape_vars = HashSet::new();

    // 入力バッファのシグネチャを生成（メタデータから）
    // ソートして決定論的な順序にする
    let mut sorted_inputs: Vec<_> = graph.input_metas().to_vec();
    sorted_inputs.sort_by(|a, b| a.name.cmp(&b.name));

    for meta in sorted_inputs {
        let shape = meta.shape.clone();

        // shape内の変数名を収集
        for expr in &shape {
            collect_shape_vars(expr, &mut shape_vars);
        }

        inputs.push(BufferSignature::new(meta.name.clone(), shape));
    }

    // 出力バッファのシグネチャを生成
    // HashMapの順序は不安定なので、名前でソートして決定論的な順序にする
    let outputs_map = graph.outputs();
    let mut sorted_outputs: Vec<_> = outputs_map.iter().collect();
    sorted_outputs.sort_by(|a, b| a.0.cmp(b.0));

    for (name, node) in sorted_outputs {
        let shape: Vec<_> = node.view.shape().to_vec();

        // shape内の変数名を収集
        for expr in &shape {
            collect_shape_vars(expr, &mut shape_vars);
        }

        outputs.push(BufferSignature::new(name.clone(), shape));
    }

    // shape_varsのHashMapを作成（名前とデフォルト値）
    let shape_var_defaults = graph.shape_var_defaults();
    let mut shape_vars_map = std::collections::HashMap::new();

    for var_name in shape_vars {
        // デフォルト値が設定されているか確認
        if let Some(&default_value) = shape_var_defaults.get(&var_name) {
            shape_vars_map.insert(var_name, default_value);
        } else {
            panic!(
                "Shape variable '{}' is used but no default value is set. \
                Use graph.set_shape_var_default(\"{}\", value) to set a default value.",
                var_name, var_name
            );
        }
    }

    KernelSignature::new(inputs, outputs, shape_vars_map)
}

/// Exprから変数名を再帰的に収集
fn collect_shape_vars(expr: &crate::graph::shape::Expr, vars: &mut HashSet<String>) {
    use crate::graph::shape::Expr;

    match expr {
        Expr::Var(name) => {
            vars.insert(name.clone());
        }
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Rem(a, b) => {
            collect_shape_vars(a, vars);
            collect_shape_vars(b, vars);
        }
        Expr::Const(_) => {}
    }
}
