//! 再帰的なLowering処理
//!
//! GraphノードをASTに変換する再帰的なLowererの実装。
//! メモ化により、同じノードは一度だけ変換される。

use crate::ast::{AstNode, Scope, VariableDecl};
use crate::graph::{GraphNode, GraphOp};
use std::collections::HashMap;

/// 再帰的にGraphノードをASTに変換するLowerer
///
/// 出力ノードから開始し、依存する入力ノードを再帰的に処理する。
/// メモ化により、同じノードは一度だけ変換される。
pub struct RecursiveLowerer {
    /// ノード → AST のキャッシュ
    /// 同じノードを複数回lowerしないようにメモ化
    cache: HashMap<GraphNode, AstNode>,

    /// ノード → 変数名 のマッピング
    /// 各GraphNodeに対応するAST変数名を管理
    node_to_var: HashMap<GraphNode, String>,

    /// 次の一時変数ID
    /// 新しい変数名を生成する際に使用
    next_temp_id: usize,

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
            node_to_var: HashMap::new(),
            next_temp_id: 0,
            declarations: Vec::new(),
            statements: Vec::new(),
        }
    }

    /// 既知のノードに変数名を設定
    ///
    /// 入力ノードや出力ノードなど、事前に変数名が決まっている場合に使用
    pub fn set_var_name(&mut self, node: &GraphNode, var_name: String) {
        self.node_to_var.insert(node.clone(), var_name);
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
                self.get_or_create_var_name(node);
                None
            }
            GraphOp::Const(lit) => {
                // 定数ノードは変数宣言と代入文を生成
                let var_name = self.get_or_create_var_name(node);
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
            GraphOp::View(source_node) => {
                // Viewノードは単にview情報を変更するだけで、メモリコピーは不要
                // 変数名はsourceと同じものを使い、view情報（stride/offset）だけが変わる
                let source_var = self.get_or_create_var_name(source_node);

                // Viewノードの変数名をsourceと同じにする（コピー不要）
                self.node_to_var.insert(node.clone(), source_var);

                // コピーループは生成しない
                None
            }
            _ => {
                // その他の演算は、既存のlower_node実装を使用
                // TODO: Phase 2で実装
                // 現在は仮の実装として、変数名だけ作成
                let _var_name = self.get_or_create_var_name(node);
                None
            }
        }
    }

    /// ノードに対応する変数名を取得または作成
    ///
    /// 既にマッピングされている場合はその名前を返し、
    /// そうでない場合は新しい変数名を生成してマッピングする。
    pub(crate) fn get_or_create_var_name(&mut self, node: &GraphNode) -> String {
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
    ///
    /// プログラム構築時に使用
    pub fn get_var_name(&self, node: &GraphNode) -> Option<String> {
        self.node_to_var.get(node).cloned()
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
    use crate::graph::{Graph, GraphNode};

    #[test]
    fn test_new_lowerer() {
        let lowerer = RecursiveLowerer::new();
        assert_eq!(lowerer.next_temp_id, 0);
        assert!(lowerer.cache.is_empty());
        assert!(lowerer.node_to_var.is_empty());
        assert!(lowerer.declarations.is_empty());
        assert!(lowerer.statements.is_empty());
    }

    #[test]
    fn test_set_var_name() {
        let mut lowerer = RecursiveLowerer::new();
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![10.into()]);

        lowerer.set_var_name(&input, "input_0".to_string());
        assert_eq!(
            lowerer.get_var_name(&input),
            Some("input_0".to_string())
        );
    }

    #[test]
    fn test_get_or_create_var_name() {
        let mut lowerer = RecursiveLowerer::new();
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![10.into()]);

        // 最初の呼び出しで新しい変数名を作成
        let var1 = lowerer.get_or_create_var_name(&input);
        assert_eq!(var1, "temp0");

        // 2回目の呼び出しで同じ変数名を返す
        let var2 = lowerer.get_or_create_var_name(&input);
        assert_eq!(var2, "temp0");

        // 別のノードで新しい変数名を作成
        let input2 = graph.input(DType::F32, vec![10.into()]);
        let var3 = lowerer.get_or_create_var_name(&input2);
        assert_eq!(var3, "temp1");
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
}
