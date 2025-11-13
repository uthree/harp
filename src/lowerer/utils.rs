use crate::ast::{AstNode, DType as AstDType, helper::*};
use crate::backend::KernelSignature;
use crate::graph::{DType as GraphDType, Graph, GraphNode};
use std::collections::HashSet;

use super::Lowerer;

impl Lowerer {
    /// GraphからKernelSignatureを生成
    pub fn create_signature(graph: &Graph) -> KernelSignature {
        use crate::backend::{BufferSignature, KernelSignature};
        use std::collections::HashSet;

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut shape_vars = HashSet::new();

        // 入力バッファのシグネチャを生成
        for (name, weak_node) in graph.inputs() {
            if let Some(node_rc) = weak_node.upgrade() {
                let shape: Vec<_> = node_rc.view.shape().to_vec();

                // shape内の変数名を収集
                for expr in &shape {
                    Self::collect_shape_vars(expr, &mut shape_vars);
                }

                inputs.push(BufferSignature::new(name.clone(), shape));
            }
        }

        // 出力バッファのシグネチャを生成
        for (name, node) in graph.outputs() {
            let shape: Vec<_> = node.view.shape().to_vec();

            // shape内の変数名を収集
            for expr in &shape {
                Self::collect_shape_vars(expr, &mut shape_vars);
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
    pub(super) fn collect_shape_vars(expr: &crate::graph::shape::Expr, vars: &mut HashSet<String>) {
        use crate::graph::shape::Expr;

        match expr {
            Expr::Var(name) => {
                vars.insert(name.clone());
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Rem(a, b) => {
                Self::collect_shape_vars(a, vars);
                Self::collect_shape_vars(b, vars);
            }
            Expr::Const(_) => {}
        }
    }

    /// GraphのDTypeをASTのPtr<DType>に変換
    pub(super) fn graph_dtype_to_ast_ptr(&self, dtype: &GraphDType) -> Result<AstDType, String> {
        let element_dtype = match dtype {
            GraphDType::F32 => AstDType::F32,
            GraphDType::Unknown => return Err("Cannot convert Unknown dtype".to_string()),
        };
        Ok(AstDType::Ptr(Box::new(element_dtype)))
    }

    /// Viewを考慮したオフセット計算
    pub(super) fn compute_offset_from_view(&self, node: &GraphNode, axes: &[usize]) -> AstNode {
        use crate::graph::shape::View;

        if axes.is_empty() {
            // スカラーの場合
            match &node.view {
                View::Linear { offset, .. } => {
                    // Expr::intoでAstNodeに変換
                    offset.clone().into()
                }
            }
        } else {
            // テンソルの場合：offset + sum(ridx[i] * stride[i])
            match &node.view {
                View::Linear {
                    strides, offset, ..
                } => {
                    let mut result: AstNode = offset.clone().into();

                    for &axis in axes {
                        let ridx = var(format!("ridx{}", axis));
                        let stride: AstNode = strides[axis].clone().into();
                        result = result + ridx * stride;
                    }

                    result
                }
            }
        }
    }

    /// 入力のオフセット計算（ridx変数を使用）
    pub(super) fn compute_offset_for_input(&self, axes: &[usize], input: &GraphNode) -> AstNode {
        use crate::graph::shape::View;

        match &input.view {
            View::Linear {
                strides, offset, ..
            } => {
                let mut result: AstNode = offset.clone().into();

                for &axis in axes {
                    let ridx = var(format!("ridx{}", axis));
                    let stride: AstNode = strides[axis].clone().into();
                    result = result + ridx * stride;
                }

                result
            }
        }
    }

    /// 出力のオフセット計算（oidx変数を使用）
    pub(super) fn compute_offset_for_output(&self, axes: &[usize], output: &GraphNode) -> AstNode {
        use crate::graph::shape::View;

        match &output.view {
            View::Linear {
                strides, offset, ..
            } => {
                let mut result: AstNode = offset.clone().into();

                for &axis in axes {
                    let oidx = var(format!("oidx{}", axis));
                    let stride: AstNode = strides[axis].clone().into();
                    result = result + oidx * stride;
                }

                result
            }
        }
    }

    /// 入力のオフセット計算（縮約軸を含む、oidxとridxを組み合わせ）
    pub(super) fn compute_offset_for_input_with_reduce_axis(
        &self,
        output_axes: &[usize],
        reduce_axis: usize,
        input: &GraphNode,
    ) -> AstNode {
        use crate::graph::shape::View;

        match &input.view {
            View::Linear {
                strides, offset, ..
            } => {
                let mut result: AstNode = offset.clone().into();

                // 出力軸に対応する入力軸
                for (out_idx, &in_axis) in output_axes.iter().enumerate() {
                    let oidx = var(format!("oidx{}", out_idx));
                    let stride: AstNode = strides[in_axis].clone().into();
                    result = result + oidx * stride;
                }

                // 縮約軸
                let ridx = var(format!("ridx{}", reduce_axis));
                let stride: AstNode = strides[reduce_axis].clone().into();
                result = result + ridx * stride;

                result
            }
        }
    }
}
