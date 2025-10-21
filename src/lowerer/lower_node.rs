use super::core::LowerContext;
use crate::ast::{AstNode, VariableDecl};
use crate::graph::{GraphNode, GraphOp};
use crate::lowerer::{
    cumulative::CumulativeLowerer, elementwise::ElementwiseLowerer,
    fused_elementwise::FusedElementwiseLowerer,
    fused_elementwise_cumulative::FusedElementwiseCumulativeLowerer,
    fused_elementwise_reduce::FusedElementwiseReduceLowerer, fused_reduce::FusedReduceLowerer,
    reduce::ReduceLowerer, utils::LowererUtils,
};

impl LowerContext {
    pub(super) fn lower_node(
        &mut self,
        node: &GraphNode,
        declarations: &mut Vec<VariableDecl>,
    ) -> Option<AstNode> {
        match &node.op {
            GraphOp::Input(_) => {
                // 入力ノードは引数として処理される
                // get_or_create_var_nameで適切な名前が生成されるようにする
                self.get_or_create_var_name(node);
                None
            }
            GraphOp::Const(lit) => {
                let var_name = self.get_or_create_var_name(node);
                declarations.push(VariableDecl {
                    name: var_name.clone(),
                    dtype: node.dtype.clone(),
                    constant: false, // 現在は初期化と代入を分けているため、constにできない
                    size_expr: None,
                });
                Some(AstNode::Assign(
                    var_name,
                    Box::new(AstNode::Const(lit.clone())),
                ))
            }
            GraphOp::Elementwise(op) => ElementwiseLowerer::lower(
                node,
                op,
                |n| self.get_or_create_var_name(n),
                declarations,
            ),
            GraphOp::Reduce(op, axis, input) => ReduceLowerer::lower(
                node,
                op,
                *axis,
                input,
                |n| self.get_or_create_var_name(n),
                declarations,
            ),
            GraphOp::Cumulative(op, axis, input) => CumulativeLowerer::lower(
                node,
                op,
                *axis,
                input,
                |n| self.get_or_create_var_name(n),
                declarations,
            ),
            GraphOp::View(source_node) => {
                // Viewノードは単にview情報を変更するだけで、メモリコピーは不要
                // 変数名はsourceと同じものを使い、view情報（stride/offset）だけが変わる
                let source_var = self.get_or_create_var_name(source_node);

                // Viewノードの変数名をsourceと同じにする（コピー不要）
                self.node_to_var.insert(node.clone(), source_var);

                // コピーループは生成しない
                None
            }
            GraphOp::Contiguous(input) => {
                // Contiguous操作: 非連続なメモリレイアウトを連続に変換
                let result_var = self.get_or_create_var_name(node);
                let input_var = self.get_or_create_var_name(input);

                // 出力ノードの場合は配列を宣言しない
                LowererUtils::declare_result_variable(
                    &result_var,
                    &node.view,
                    &node.dtype,
                    declarations,
                );

                // 入力のview（非連続の可能性あり）と出力のview（連続）を取得
                let input_view = &input.view;
                let result_view = &node.view;

                // 入力から連続な出力へコピーするループを生成
                Some(Self::create_contiguous_copy_loop(
                    input_view,
                    result_view,
                    &input_var,
                    &result_var,
                    0,
                ))
            }
            GraphOp::Cast(input, target_dtype) => {
                // Cast操作: 型変換
                // 注意: 同じ型へのキャストでも、出力ノードへのコピーが必要な場合があるため、
                // Viewのように変数を共有するのではなく、常にコピーループを生成する

                let result_var = self.get_or_create_var_name(node);
                let input_var = self.get_or_create_var_name(input);

                // 出力ノードの場合は配列を宣言しない
                LowererUtils::declare_result_variable(
                    &result_var,
                    &node.view,
                    target_dtype,
                    declarations,
                );

                // キャストループを生成
                let input_view = &input.view;
                let result_view = &node.view;

                Some(Self::create_cast_loop(
                    input_view,
                    result_view,
                    &input_var,
                    &result_var,
                    target_dtype,
                    0,
                ))
            }
            GraphOp::FusedElementwise(ast, inputs) => {
                FusedElementwiseLowerer::lower(node, ast, inputs, declarations, |n| {
                    self.get_or_create_var_name(n)
                })
            }
            GraphOp::FusedReduce(op, axes, input) => {
                FusedReduceLowerer::lower(node, op, axes, input, declarations, |n| {
                    self.get_or_create_var_name(n)
                })
            }
            GraphOp::FusedElementwiseReduce(ast, inputs, op, axes) => {
                FusedElementwiseReduceLowerer::lower(
                    node,
                    ast,
                    inputs,
                    op,
                    axes,
                    declarations,
                    |n| self.get_or_create_var_name(n),
                )
            }
            GraphOp::FusedElementwiseCumulative(ast, inputs, op, axis) => {
                FusedElementwiseCumulativeLowerer::lower(
                    node,
                    ast,
                    inputs,
                    op,
                    *axis,
                    declarations,
                    |n| self.get_or_create_var_name(n),
                )
            }
            GraphOp::Fold(dim, _window_size, stride, dilation, input) => {
                // Fold operation (col2im): combines overlapping windows
                // Input:  [..., L', K] where last dim is window dimension
                // Output: [..., L] where L = (L'-1)*stride + (K-1)*dilation + 1
                let result_var = self.get_or_create_var_name(node);
                let input_var = self.get_or_create_var_name(input);

                // Declare output array if needed
                LowererUtils::declare_result_variable(
                    &result_var,
                    &node.view,
                    &node.dtype,
                    declarations,
                );

                let input_view = &input.view;
                let result_view = &node.view;

                // Generate fold loops: initialize to zero, then accumulate
                Some(Self::create_fold_loops(
                    input_view,
                    result_view,
                    *dim,
                    *stride,
                    *dilation,
                    &input_var,
                    &result_var,
                ))
            }
            GraphOp::Pad(_input, _axis, _amount) => {
                // TODO: Implement padding operation lowering
                // This will be implemented in M2 when graph-level optimizations are added
                todo!("Pad operation lowering not yet implemented")
            }
        }
    }
}
