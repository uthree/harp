//! 再帰的なLowering処理
//!
//! GraphノードをASTに変換する再帰的なLowererの実装。
//! メモ化により、同じノードは一度だけ変換される。

use crate::ast::{AstNode, Scope, VariableDecl};
use crate::graph::{GraphNode, GraphOp};
use crate::lowerer::{
    cumulative::CumulativeLowerer, elementwise::ElementwiseLowerer,
    fused_elementwise::FusedElementwiseLowerer,
    fused_elementwise_cumulative::FusedElementwiseCumulativeLowerer,
    fused_elementwise_reduce::FusedElementwiseReduceLowerer, fused_reduce::FusedReduceLowerer,
    pad::PadLowerer, reduce::ReduceLowerer, utils::LowererUtils,
};
use std::collections::HashMap;

/// 変数名のマッピングを管理
///
/// GraphNodeに対応する変数名を生成・管理する。
/// RecursiveLowererから分離することで、借用の問題を解決。
pub struct VarMapper {
    /// ノード → 変数名 のマッピング
    node_to_var: HashMap<GraphNode, String>,

    /// 次の一時変数ID
    next_temp_id: usize,
}

impl VarMapper {
    pub fn new() -> Self {
        Self {
            node_to_var: HashMap::new(),
            next_temp_id: 0,
        }
    }

    /// 既知のノードに変数名を設定
    pub fn set_var_name(&mut self, node: &GraphNode, var_name: String) {
        self.node_to_var.insert(node.clone(), var_name);
    }

    /// ノードに対応する変数名を取得または作成
    pub fn get_or_create_var_name(&mut self, node: &GraphNode) -> String {
        if let Some(name) = self.node_to_var.get(node) {
            name.clone()
        } else {
            let name = format!("temp{}", self.next_temp_id);
            self.next_temp_id += 1;
            self.node_to_var.insert(node.clone(), name.clone());
            name
        }
    }

    /// 変数名のマッピングを取得
    pub fn get_var_name(&self, node: &GraphNode) -> Option<String> {
        self.node_to_var.get(node).cloned()
    }
}

impl Default for VarMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// 再帰的にGraphノードをASTに変換するLowerer
///
/// 出力ノードから開始し、依存する入力ノードを再帰的に処理する。
/// メモ化により、同じノードは一度だけ変換される。
pub struct RecursiveLowerer {
    /// ノード → AST のキャッシュ
    /// 同じノードを複数回lowerしないようにメモ化
    cache: HashMap<GraphNode, AstNode>,

    /// 変数名マッパー
    var_mapper: VarMapper,

    /// 変数宣言の収集
    /// lowering中に生成された変数宣言を保持
    pub declarations: Vec<VariableDecl>,

    /// 生成されたASTステートメント
    /// lowering中に生成されたステートメントを保持
    pub statements: Vec<AstNode>,
}

impl RecursiveLowerer {
    /// 新しいRecursiveLowererを作成
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            var_mapper: VarMapper::new(),
            declarations: Vec::new(),
            statements: Vec::new(),
        }
    }

    /// 既知のノードに変数名を設定
    ///
    /// 入力ノードや出力ノードなど、事前に変数名が決まっている場合に使用
    pub fn set_var_name(&mut self, node: &GraphNode, var_name: String) {
        self.var_mapper.set_var_name(node, var_name);
    }

    /// 変数名のマッピングを取得
    pub fn get_var_name(&self, node: &GraphNode) -> Option<String> {
        self.var_mapper.get_var_name(node)
    }

    /// ノードを再帰的にlower
    ///
    /// 入力ノードが未処理なら再帰的に処理してから、このノードをlowerする。
    /// メモ化により、同じノードは一度だけ処理される。
    ///
    /// # Arguments
    ///
    /// * `node` - 変換するGraphNode
    ///
    /// # Returns
    ///
    /// 変換されたASTノード（ステートメントがない場合はNop）
    pub fn lower_node(&mut self, node: &GraphNode) -> AstNode {
        // キャッシュチェック
        if let Some(cached_ast) = self.cache.get(node) {
            return cached_ast.clone();
        }

        // 入力ノードを先に処理（再帰）
        for input_node in node.input_nodes() {
            self.lower_node(&input_node);
        }

        // このノード自体をlower
        let ast = self.lower_node_impl(node);

        // キャッシュに保存
        if let Some(stmt) = &ast {
            self.cache.insert(node.clone(), stmt.clone());
            // ステートメントリストに追加
            self.statements.push(stmt.clone());
        }

        ast.unwrap_or_else(|| AstNode::Block {
            scope: Scope {
                declarations: Vec::new(),
            },
            statements: Vec::new(),
        })
    }

    /// ノードの実際のlowering処理
    ///
    /// GraphOpの種類に応じて適切なASTを生成する。
    fn lower_node_impl(&mut self, node: &GraphNode) -> Option<AstNode> {
        match &node.op {
            GraphOp::Input(_) => {
                // 入力ノードは変数名をマッピングするだけ
                // ASTステートメントは生成しない（引数として渡される）
                self.var_mapper.get_or_create_var_name(node);
                None
            }
            GraphOp::Const(lit) => {
                // 定数ノードは変数宣言と代入文を生成
                let var_name = self.var_mapper.get_or_create_var_name(node);
                self.declarations.push(VariableDecl {
                    name: var_name.clone(),
                    dtype: node.dtype.clone(),
                    constant: false,
                    size_expr: None,
                });
                Some(AstNode::Assign(
                    var_name,
                    Box::new(AstNode::Const(lit.clone())),
                ))
            }
            GraphOp::Elementwise(op) => {
                // Elementwise演算のlowering
                ElementwiseLowerer::lower(
                    node,
                    op,
                    |n| self.var_mapper.get_or_create_var_name(n),
                    &mut self.declarations,
                    &node.strategy,
                )
            }
            GraphOp::Reduce(op, axis, input) => {
                // Reduce演算のlowering
                ReduceLowerer::lower(
                    node,
                    op,
                    *axis,
                    input,
                    |n| self.var_mapper.get_or_create_var_name(n),
                    &mut self.declarations,
                )
            }
            GraphOp::Cumulative(op, axis, input) => {
                // Cumulative演算のlowering
                CumulativeLowerer::lower(
                    node,
                    op,
                    *axis,
                    input,
                    |n| self.var_mapper.get_or_create_var_name(n),
                    &mut self.declarations,
                )
            }
            GraphOp::View(source_node) => {
                // Viewノードは単にview情報を変更するだけで、メモリコピーは不要
                // 変数名はsourceと同じものを使い、view情報（stride/offset）だけが変わる
                let source_var = self.var_mapper.get_or_create_var_name(source_node);

                // Viewノードの変数名をsourceと同じにする（コピー不要）
                self.var_mapper.set_var_name(node, source_var);

                // コピーループは生成しない
                None
            }
            GraphOp::Contiguous(input) => {
                // Contiguous操作: 非連続なメモリレイアウトを連続に変換
                let result_var = self.var_mapper.get_or_create_var_name(node);
                let input_var = self.var_mapper.get_or_create_var_name(input);

                // 出力ノードの場合は配列を宣言しない
                LowererUtils::declare_result_variable(
                    &result_var,
                    &node.view,
                    &node.dtype,
                    &mut self.declarations,
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
                let result_var = self.var_mapper.get_or_create_var_name(node);
                let input_var = self.var_mapper.get_or_create_var_name(input);

                // 出力ノードの場合は配列を宣言しない
                LowererUtils::declare_result_variable(
                    &result_var,
                    &node.view,
                    target_dtype,
                    &mut self.declarations,
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
                // 融合Elementwise演算のlowering
                FusedElementwiseLowerer::lower(node, ast, inputs, &mut self.declarations, |n| {
                    self.var_mapper.get_or_create_var_name(n)
                })
            }
            GraphOp::FusedReduce(op, axes, input) => {
                // 融合Reduce演算のlowering
                FusedReduceLowerer::lower(node, op, axes, input, &mut self.declarations, |n| {
                    self.var_mapper.get_or_create_var_name(n)
                })
            }
            GraphOp::FusedElementwiseReduce(ast, inputs, op, axes) => {
                // 融合ElementwiseReduce演算のlowering
                FusedElementwiseReduceLowerer::lower(
                    node,
                    ast,
                    inputs,
                    op,
                    axes,
                    &mut self.declarations,
                    |n| self.var_mapper.get_or_create_var_name(n),
                )
            }
            GraphOp::FusedElementwiseCumulative(ast, inputs, op, axis) => {
                // 融合ElementwiseCumulative演算のlowering
                FusedElementwiseCumulativeLowerer::lower(
                    node,
                    ast,
                    inputs,
                    op,
                    *axis,
                    &mut self.declarations,
                    |n| self.var_mapper.get_or_create_var_name(n),
                )
            }
            GraphOp::Fold(dim, _window_size, stride, dilation, input) => {
                // Fold operation (col2im): 重なったウィンドウを統合
                let result_var = self.var_mapper.get_or_create_var_name(node);
                let input_var = self.var_mapper.get_or_create_var_name(input);

                // 出力配列を宣言
                LowererUtils::declare_result_variable(
                    &result_var,
                    &node.view,
                    &node.dtype,
                    &mut self.declarations,
                );

                let input_view = &input.view;
                let result_view = &node.view;

                // Foldループを生成: 0で初期化してから累積
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
            GraphOp::Pad(input, axis, amount) => {
                // パディング処理のlowering
                let result_var = self.var_mapper.get_or_create_var_name(node);
                let input_var = self.var_mapper.get_or_create_var_name(input);

                // 出力配列を宣言
                LowererUtils::declare_result_variable(
                    &result_var,
                    &node.view,
                    &node.dtype,
                    &mut self.declarations,
                );

                // view情報を取得
                let input_view = &input.view;
                let output_view = &node.view;

                // パディング処理のコード生成
                Some(PadLowerer::lower(
                    input_view,
                    output_view,
                    *axis,
                    amount,
                    &input_var,
                    &result_var,
                    &node.dtype,
                ))
            }
        }
    }

    /// Contiguous変換のためのコピーループを作成
    fn create_contiguous_copy_loop(
        input_view: &crate::graph::shape::view::View,
        result_view: &crate::graph::shape::view::View,
        input_var: &str,
        result_var: &str,
        dim: usize,
    ) -> AstNode {
        use crate::graph::shape::view::View;

        let (
            View::Linear {
                shape,
                strides: input_strides,
                offset: input_offset,
                ..
            },
            View::Linear {
                strides: result_strides,
                offset: result_offset,
                ..
            },
        ) = (input_view, result_view);

        if dim >= shape.len() {
            // 最内レベル: コピーを実行
            let input_index = LowererUtils::compute_memory_index(input_strides, input_offset, dim);
            let result_index =
                LowererUtils::compute_memory_index(result_strides, result_offset, dim);

            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(AstNode::Load {
                    target: Box::new(AstNode::Var(input_var.to_string())),
                    index: Box::new(input_index),
                    vector_width: 1,
                }),
                vector_width: 1,
            }
        } else {
            // ループを生成
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_contiguous_copy_loop(
                input_view,
                result_view,
                input_var,
                result_var,
                dim + 1,
            );

            LowererUtils::create_dimension_loop(loop_var, &shape[dim], inner_body, None)
        }
    }

    /// Castのためのループを作成
    fn create_cast_loop(
        input_view: &crate::graph::shape::view::View,
        result_view: &crate::graph::shape::view::View,
        input_var: &str,
        result_var: &str,
        target_dtype: &crate::ast::DType,
        dim: usize,
    ) -> AstNode {
        use crate::graph::shape::view::View;

        let (
            View::Linear {
                shape,
                strides: input_strides,
                offset: input_offset,
                ..
            },
            View::Linear {
                strides: result_strides,
                offset: result_offset,
                ..
            },
        ) = (input_view, result_view);

        if dim >= shape.len() {
            // 最内レベル: キャストを実行
            let input_index = LowererUtils::compute_memory_index(input_strides, input_offset, dim);
            let result_index =
                LowererUtils::compute_memory_index(result_strides, result_offset, dim);

            // Cast AstNodeを使用して型変換
            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(AstNode::Cast {
                    dtype: target_dtype.clone(),
                    expr: Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var(input_var.to_string())),
                        index: Box::new(input_index),
                        vector_width: 1,
                    }),
                }),
                vector_width: 1,
            }
        } else {
            // ループを生成
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_cast_loop(
                input_view,
                result_view,
                input_var,
                result_var,
                target_dtype,
                dim + 1,
            );

            LowererUtils::create_dimension_loop(loop_var, &shape[dim], inner_body, None)
        }
    }

    /// Fold operation (col2im)のためのループを作成
    fn create_fold_loops(
        input_view: &crate::graph::shape::view::View,
        result_view: &crate::graph::shape::view::View,
        dim: usize,
        stride: usize,
        dilation: usize,
        input_var: &str,
        result_var: &str,
    ) -> AstNode {
        // まず出力を0で初期化
        let init_loop = Self::create_zero_init_loop(result_view, result_var, 0);

        // 次に、入力からデータを累積
        let fold_loop = Self::create_fold_accumulate_loop(
            input_view,
            result_view,
            dim,
            stride,
            dilation,
            input_var,
            result_var,
            0,
        );

        // 初期化と累積を順に実行
        AstNode::Block {
            scope: Scope {
                declarations: Vec::new(),
            },
            statements: vec![init_loop, fold_loop],
        }
    }

    /// 配列を0で初期化するループを作成
    fn create_zero_init_loop(
        view: &crate::graph::shape::view::View,
        var: &str,
        dim: usize,
    ) -> AstNode {
        use crate::ast::ConstLiteral;
        use crate::graph::shape::view::View;

        let View::Linear {
            shape,
            strides,
            offset,
            ..
        } = view;

        if dim >= shape.len() {
            // 最内レベル: 0を代入
            let index = LowererUtils::compute_memory_index(strides, offset, dim);

            AstNode::Store {
                target: Box::new(AstNode::Var(var.to_string())),
                index: Box::new(index),
                value: Box::new(AstNode::Const(ConstLiteral::F32(0.0))),
                vector_width: 1,
            }
        } else {
            // ループを生成
            let loop_var = format!("init{}", dim);
            let inner_body = Self::create_zero_init_loop(view, var, dim + 1);

            LowererUtils::create_dimension_loop(loop_var, &shape[dim], inner_body, None)
        }
    }

    /// Fold累積ループを作成
    #[allow(clippy::too_many_arguments)]
    fn create_fold_accumulate_loop(
        input_view: &crate::graph::shape::view::View,
        result_view: &crate::graph::shape::view::View,
        fold_dim: usize,
        stride: usize,
        dilation: usize,
        input_var: &str,
        result_var: &str,
        current_dim: usize,
    ) -> AstNode {
        use crate::graph::shape::view::View;

        let View::Linear {
            shape: input_shape,
            strides: input_strides,
            offset: input_offset,
            ..
        } = input_view;

        let window_dim = input_shape.len() - 1; // 最後の次元がウィンドウ次元

        if current_dim >= input_shape.len() {
            // 最内レベル: 累積
            // input[..., i_fold_dim, i_window_dim] を
            // output[..., i_fold_dim * stride + i_window_dim * dilation] に累積
            let input_index =
                LowererUtils::compute_memory_index(input_strides, input_offset, input_shape.len());

            // stride と dilation を考慮した result のインデックスを計算
            let result_index = Self::compute_fold_result_index(
                result_view,
                fold_dim,
                window_dim,
                stride,
                dilation,
                input_shape.len(),
            );

            // result[idx] += input[idx]
            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index.clone()),
                value: Box::new(AstNode::Add(
                    Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var(result_var.to_string())),
                        index: Box::new(result_index),
                        vector_width: 1,
                    }),
                    Box::new(AstNode::Load {
                        target: Box::new(AstNode::Var(input_var.to_string())),
                        index: Box::new(input_index),
                        vector_width: 1,
                    }),
                )),
                vector_width: 1,
            }
        } else {
            // ループを生成
            let loop_var = format!("ridx{}", current_dim);
            let inner_body = Self::create_fold_accumulate_loop(
                input_view,
                result_view,
                fold_dim,
                stride,
                dilation,
                input_var,
                result_var,
                current_dim + 1,
            );

            LowererUtils::create_dimension_loop(
                loop_var,
                &input_shape[current_dim],
                inner_body,
                None,
            )
        }
    }

    /// Fold操作のためのresultインデックスを計算
    /// input[..., i_fold_dim, ..., i_window_dim] を output[..., i_fold_dim * stride + i_window_dim * dilation, ...] にマップ
    fn compute_fold_result_index(
        result_view: &crate::graph::shape::view::View,
        fold_dim: usize,
        window_dim: usize,
        stride: usize,
        dilation: usize,
        num_input_dims: usize,
    ) -> AstNode {
        use crate::ast::ConstLiteral;
        use crate::graph::shape::view::View;

        let View::Linear {
            strides: result_strides,
            offset: result_offset,
            ..
        } = result_view;
        let mut index = LowererUtils::shape_expr_to_ast_node(result_offset);

        for dim in 0..num_input_dims {
            if dim == window_dim {
                // ウィンドウ次元はスキップ（fold_dimに折り畳まれている）
                continue;
            }

            let loop_var = format!("ridx{}", dim);
            let result_dim = if dim > fold_dim { dim - 1 } else { dim };

            if dim == fold_dim {
                // fold_dim の場合: result_index = i_fold_dim * stride + i_window_dim * dilation
                let fold_index = AstNode::Add(
                    Box::new(AstNode::Mul(
                        Box::new(AstNode::Var(loop_var)),
                        Box::new(AstNode::Const(ConstLiteral::Isize(stride as isize))),
                    )),
                    Box::new(AstNode::Mul(
                        Box::new(AstNode::Var(format!("ridx{}", window_dim))),
                        Box::new(AstNode::Const(ConstLiteral::Isize(dilation as isize))),
                    )),
                );
                index += LowererUtils::shape_expr_to_ast_node(&result_strides[result_dim].clone())
                    * fold_index;
            } else {
                index += LowererUtils::shape_expr_to_ast_node(&result_strides[result_dim].clone())
                    * AstNode::Var(loop_var);
            }
        }

        index
    }
}

impl Default for RecursiveLowerer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::ops::cumulative::CumulativeOps;
    use crate::graph::ops::reduce::ReduceOps;
    use crate::graph::{Graph, GraphNode};

    #[test]
    fn test_new_lowerer() {
        let lowerer = RecursiveLowerer::new();
        // 公開フィールドが空であることを確認
        assert!(lowerer.declarations.is_empty());
        assert!(lowerer.statements.is_empty());
    }

    #[test]
    fn test_set_var_name() {
        let mut lowerer = RecursiveLowerer::new();
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![10.into()]);

        lowerer.set_var_name(&input, "input_0".to_string());
        assert_eq!(lowerer.get_var_name(&input), Some("input_0".to_string()));
    }

    #[test]
    fn test_var_name_mapping() {
        let mut lowerer = RecursiveLowerer::new();
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![10.into()]);

        // loweringを実行すると変数名がマッピングされる
        lowerer.lower_node(&input);

        // 同じノードは同じ変数名を持つ
        let var1 = lowerer.get_var_name(&input);
        assert!(var1.is_some());

        // 2回目のloweringでも同じ変数名
        lowerer.lower_node(&input);
        let var2 = lowerer.get_var_name(&input);
        assert_eq!(var1, var2);

        // 別のノードは別の変数名を持つ
        let input2 = graph.input(DType::F32, vec![10.into()]);
        lowerer.lower_node(&input2);
        let var3 = lowerer.get_var_name(&input2);
        assert!(var3.is_some());
        assert_ne!(var1, var3);
    }

    #[test]
    fn test_lower_const_node() {
        let mut lowerer = RecursiveLowerer::new();
        let const_node = GraphNode::f32(42.0);

        let ast = lowerer.lower_node(&const_node);

        // ASTが生成されたことを確認（空のBlockではない）
        assert!(matches!(ast, AstNode::Assign(_, _)));

        // 変数宣言が追加されたことを確認
        assert_eq!(lowerer.declarations.len(), 1);
        assert_eq!(lowerer.declarations[0].name, "temp0");
        assert_eq!(lowerer.declarations[0].dtype, DType::F32);

        // ステートメントが追加されたことを確認
        assert_eq!(lowerer.statements.len(), 1);
    }

    #[test]
    fn test_lower_input_node() {
        let mut lowerer = RecursiveLowerer::new();
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![10.into()]);

        let ast = lowerer.lower_node(&input);

        // 入力ノードはステートメントを生成しない（空のBlock）
        assert!(matches!(
            ast,
            AstNode::Block {
                statements,
                ..
            } if statements.is_empty()
        ));

        // しかし変数名はマッピングされる
        assert!(lowerer.get_var_name(&input).is_some());
    }

    #[test]
    fn test_memoization() {
        let mut lowerer = RecursiveLowerer::new();
        let const_node = GraphNode::f32(42.0);

        // 1回目のlower
        let ast1 = lowerer.lower_node(&const_node);

        // キャッシュに保存されたことを確認
        assert!(lowerer.cache.contains_key(&const_node));

        // 2回目のlower（キャッシュから取得）
        let ast2 = lowerer.lower_node(&const_node);

        // 同じASTが返されることを確認
        assert_eq!(format!("{:?}", ast1), format!("{:?}", ast2));

        // ステートメントは1回だけ追加される
        assert_eq!(lowerer.statements.len(), 1);
    }

    #[test]
    fn test_view_node_shares_var() {
        let mut lowerer = RecursiveLowerer::new();
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![10.into(), 20.into()]);

        // Viewノードを作成（permuteで軸を入れ替え）
        let view_node = input.clone().permute(vec![1, 0]);

        // lowering
        lowerer.lower_node(&view_node);

        // Viewノードとソースノードが同じ変数名を持つことを確認
        let input_var = lowerer.get_var_name(&input).unwrap();
        let view_var = lowerer.get_var_name(&view_node).unwrap();
        assert_eq!(input_var, view_var);
    }

    // 統合テスト：様々なGraphOpタイプのloweringをテスト

    #[test]
    fn test_elementwise_binary_op() {
        let mut lowerer = RecursiveLowerer::new();
        let mut graph = Graph::new();

        // 二項演算: input1 + input2
        let input1 = graph.input(DType::F32, vec![10.into()]);
        let input2 = graph.input(DType::F32, vec![10.into()]);
        let result = input1 + input2;

        // lowering実行
        lowerer.lower_node(&result);

        // ステートメントが生成されたことを確認（加算ループ）
        assert!(!lowerer.statements.is_empty());

        // 結果ノードに変数名がマッピングされていることを確認
        assert!(lowerer.get_var_name(&result).is_some());

        // 変数宣言が追加されていることを確認（result変数）
        assert!(!lowerer.declarations.is_empty());
    }

    #[test]
    fn test_reduce_op() {
        let mut lowerer = RecursiveLowerer::new();
        let mut graph = Graph::new();

        // Reduce操作: sum along axis 0
        let input = graph.input(DType::F32, vec![10.into(), 20.into()]);
        let result = input.sum(0);

        // lowering実行
        lowerer.lower_node(&result);

        // ステートメントが生成されたことを確認（reduceループ）
        assert!(!lowerer.statements.is_empty());

        // 結果ノードに変数名がマッピングされていることを確認
        assert!(lowerer.get_var_name(&result).is_some());
    }

    #[test]
    fn test_cast_op() {
        let mut lowerer = RecursiveLowerer::new();
        let mut graph = Graph::new();

        // Cast操作: F32 -> Isize
        let input = graph.input(DType::F32, vec![10.into()]);
        let result = input.cast(DType::Isize);

        // lowering実行
        lowerer.lower_node(&result);

        // ステートメントが生成されたことを確認（castループ）
        assert!(!lowerer.statements.is_empty());

        // 結果ノードに変数名がマッピングされていることを確認
        assert!(lowerer.get_var_name(&result).is_some());

        // 型がIsizeであることを確認
        assert_eq!(result.dtype, DType::Isize);
    }

    #[test]
    fn test_complex_graph() {
        let mut lowerer = RecursiveLowerer::new();
        let mut graph = Graph::new();

        // 複雑なグラフ: (input1 + input2) * constant
        let input1 = graph.input(DType::F32, vec![5.into(), 10.into()]);
        let input2 = graph.input(DType::F32, vec![5.into(), 10.into()]);
        let constant = GraphNode::f32(2.0);

        let sum = input1 + input2;
        let result = sum.clone() * constant.clone();

        // lowering実行
        lowerer.lower_node(&result);

        // 複数のステートメントが生成されたことを確認
        // (constant代入、sum計算、mul計算)
        assert!(lowerer.statements.len() >= 2);

        // すべての中間ノードに変数名がマッピングされていることを確認
        assert!(lowerer.get_var_name(&constant).is_some());
        assert!(lowerer.get_var_name(&sum).is_some());
        assert!(lowerer.get_var_name(&result).is_some());
    }

    #[test]
    fn test_cumulative_op() {
        let mut lowerer = RecursiveLowerer::new();
        let mut graph = Graph::new();

        // Cumulative操作: cumsum along axis 0
        let input = graph.input(DType::F32, vec![10.into(), 20.into()]);
        let result = input.cumsum(0);

        // lowering実行
        lowerer.lower_node(&result);

        // ステートメントが生成されたことを確認（cumulativeループ）
        assert!(!lowerer.statements.is_empty());

        // 結果ノードに変数名がマッピングされていることを確認
        assert!(lowerer.get_var_name(&result).is_some());
    }

    #[test]
    fn test_contiguous_op() {
        let mut lowerer = RecursiveLowerer::new();
        let mut graph = Graph::new();

        // Contiguous操作: permuteした後にcontiguousにする
        let input = graph.input(DType::F32, vec![10.into(), 20.into()]);
        let permuted = input.permute(vec![1, 0]);
        let result = permuted.contiguous();

        // lowering実行
        lowerer.lower_node(&result);

        // ステートメントが生成されたことを確認（copyループ）
        assert!(!lowerer.statements.is_empty());

        // 結果ノードに変数名がマッピングされていることを確認
        assert!(lowerer.get_var_name(&result).is_some());
    }
}
