//! Custom関数のLowering
//!
//! GraphOp::Customを処理し、プレースホルダー変数を実際のパラメータに置換します。

use crate::ast::{AstNode, DType as AstDType, FunctionKind, VarDecl};
use crate::graph::GraphNode;
use crate::graph::ops::custom_placeholders as ph;
use log::debug;
use std::collections::{HashMap, HashSet};

use super::Lowerer;

impl Lowerer {
    /// Custom関数をカーネル関数に変換
    ///
    /// Custom関数は既にAstNode::Functionの形式で構築されています。
    /// ここでは以下を行います：
    /// 1. 関数名を設定（kernel_N）
    /// 2. パラメータリストを構築（input0, input1, ..., output, shape0, shape1, ...）
    /// 3. プレースホルダー変数を実際のパラメータに置換
    pub(super) fn lower_custom_function(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        custom_fn: &AstNode,
    ) -> Result<AstNode, String> {
        debug!("Lowering custom function");
        debug!("View: {:?}", node.view);

        // Custom関数が構築された時の次元数を決定
        // Reduce/Cumulative操作では入力と出力の次元数が異なるため、
        // 入力ノードの次元数を使用する
        let input_shape = if !node.src.is_empty() {
            // 最初の入力ノードのshapeを使用
            node.src[0].view.shape().to_vec()
        } else {
            node.view.shape().to_vec()
        };
        let ndim = input_shape.len();

        // パラメータを生成
        let mut params = Vec::new();

        // 入力バッファー（srcノード）
        for (i, src) in node.src.iter().enumerate() {
            params.push(self.create_input_param(i, &src.dtype)?);
        }

        // 出力バッファー
        params.push(self.create_output_param(&node.dtype)?);

        // Shape変数（shape式が変数を含む場合のみパラメータとして追加）
        // また、プレースホルダーshape0, shape1などを実際の値に置換するマッピングを作成
        let mut shape_substitutions: HashMap<String, AstNode> = HashMap::new();
        let mut shape_var_names = HashSet::new();

        for (axis, expr) in input_shape.iter().enumerate() {
            let placeholder_name = ph::shape(axis);
            let ast_expr: AstNode = expr.clone().into();

            // 式が変数を含むかチェック
            Self::collect_shape_vars(expr, &mut shape_var_names);

            // プレースホルダーを実際の式に置換するマッピングを作成
            shape_substitutions.insert(placeholder_name, ast_expr);
        }

        // shape式に含まれる変数をパラメータとして追加
        let mut sorted_vars: Vec<_> = shape_var_names.into_iter().collect();
        sorted_vars.sort();
        for var_name in sorted_vars {
            params.push(VarDecl {
                name: var_name,
                dtype: AstDType::Int,
                mutability: crate::ast::Mutability::Immutable,
                kind: crate::ast::VarKind::Normal,
            });
        }

        // カスタム関数からボディを取得し、shape変数を置換
        let body = self.extract_and_substitute_body(node, custom_fn, ndim, &shape_substitutions)?;

        // カーネル関数を作成
        let function_name = format!("kernel_{}", node_id);

        Ok(AstNode::Function {
            name: Some(function_name),
            kind: FunctionKind::Normal,
            params,
            return_type: AstDType::Tuple(vec![]),
            body: Box::new(body),
        })
    }

    /// Custom関数のボディを抽出し、プレースホルダー変数を置換
    fn extract_and_substitute_body(
        &self,
        _node: &GraphNode,
        custom_fn: &AstNode,
        _ndim: usize,
        shape_substitutions: &HashMap<String, AstNode>,
    ) -> Result<AstNode, String> {
        // 関数からボディを取得
        let body = match custom_fn {
            AstNode::Function { body, .. } => body.as_ref().clone(),
            _ => return Err("Custom function must be AstNode::Function".to_string()),
        };

        // プレースホルダー変数（shape0, shape1, ...）を実際の値に置換
        // substitute_vars を使用して Var ノードを置換
        let substituted_body = body.substitute_vars(shape_substitutions);

        Ok(substituted_body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Graph};

    #[test]
    fn test_lower_custom_elementwise() {
        use crate::ast::helper::wildcard;

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);

        // (a + b) のカスタム関数を作成
        let expr = wildcard("0") + wildcard("1");

        // custom_elementwise_binaryを使用してCustomノードを作成
        let result = a.custom_elementwise_binary(b, expr);
        graph.output("result", result);

        // lowering
        let program = crate::lowerer::lower(graph);

        // Programが正常に生成されることを確認
        assert!(matches!(program, AstNode::Program { .. }));
    }
}
