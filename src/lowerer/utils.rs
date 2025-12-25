//! Lowererユーティリティ
//!
//! グラフからシグネチャを生成するためのユーティリティ関数を提供します。

use crate::backend::KernelSignature;
use crate::graph::Graph;

/// GraphからKernelSignatureを生成
pub fn create_signature(graph: &Graph) -> KernelSignature {
    use crate::backend::BufferSignature;

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    // 入力バッファのシグネチャを生成（メタデータから）
    // ソートして決定論的な順序にする
    let mut sorted_inputs: Vec<_> = graph.input_metas().to_vec();
    sorted_inputs.sort_by(|a, b| a.name.cmp(&b.name));

    for meta in sorted_inputs {
        inputs.push(BufferSignature::new(meta.name.clone(), meta.shape.clone()));
    }

    // 出力バッファのシグネチャを生成
    // HashMapの順序は不安定なので、名前でソートして決定論的な順序にする
    let outputs_map = graph.outputs();
    let mut sorted_outputs: Vec<_> = outputs_map.iter().collect();
    sorted_outputs.sort_by(|a, b| a.0.cmp(b.0));

    for (name, node) in sorted_outputs {
        let shape: Vec<_> = node.view.shape().to_vec();
        outputs.push(BufferSignature::new(name.clone(), shape));
    }

    KernelSignature::new(inputs, outputs)
}
