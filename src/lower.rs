//! `UOp`グラフを`UOp`ツリー（AST）に変換するLowering処理を実装するモジュール。

use crate::uop::{Op, UOp};
use std::collections::HashMap;
use std::rc::Rc;

/// Lowering処理の状態を保持する構造体。
struct Lowerer {
    // グラフの元ノードのポインタから、AST内での表現（変数など）へのマッピング
    node_map: HashMap<*const crate::uop::UOp_, UOp>,
    // ASTのループ本体を構成するステートメントのリスト
    kernel_body: Vec<UOp>,
    // 新しい変数を生成するためのカウンター
    var_counter: usize,
}

impl Lowerer {
    fn new() -> Self {
        Self {
            node_map: HashMap::new(),
            kernel_body: Vec::new(),
            var_counter: 0,
        }
    }

    /// 新しい一意な変数名を生成する。
    fn new_var(&mut self, prefix: &str) -> String {
        let name = format!("{}{}", prefix, self.var_counter);
        self.var_counter += 1;
        name
    }

    /// UOpグラフのノードを再帰的に処理し、ASTを構築する。
    fn process_node(&mut self, node: &UOp) -> UOp {
        // 既に処理済みのノードは、マッピングされた変数を返す
        let node_ptr = Rc::as_ptr(&node.0);
        if let Some(mapped) = self.node_map.get(&node_ptr) {
            return mapped.clone();
        }

        // 子ノードを先に処理する（後順走査）
        let new_srcs: Vec<UOp> = node.0.src.iter().map(|src| self.process_node(src)).collect();

        // 現在のノードを処理する
        let result_var = match node.0.op {
            // ConstとVarノードはそのまま伝播させる
            Op::Const(_) | Op::Var(_) => {
                return UOp::new(node.0.op.clone(), node.0.dtype.clone(), new_srcs);
            }
            // Loadノードは、結果を新しい変数に格納する
            Op::Load => {
                let var_name = self.new_var("tmp");
                let var_dtype = node.0.dtype.clone();
                let var_uop = UOp::var(&var_name, var_dtype.clone());

                // `tmp0 = buf0[i];` のようなStore文を生成
                let load_expr = UOp::new(Op::Load, var_dtype, new_srcs);
                let store_stmt = UOp::new(
                    Op::Store,
                    crate::dtype::DType::Unit,
                    vec![var_uop.clone(), load_expr],
                );
                self.kernel_body.push(store_stmt);
                var_uop
            }
            // その他の演算ノードも、結果を新しい変数に格納する
            _ => {
                let var_name = self.new_var("res");
                let var_dtype = node.0.dtype.clone();
                let var_uop = UOp::var(&var_name, var_dtype.clone());

                // `res0 = tmp0 + tmp1;` のようなStore文を生成
                let expr = UOp::new(node.0.op.clone(), var_dtype, new_srcs);
                let store_stmt = UOp::new(
                    Op::Store,
                    crate::dtype::DType::Unit,
                    vec![var_uop.clone(), expr],
                );
                self.kernel_body.push(store_stmt);
                var_uop
            }
        };

        // 元のノードを、新しく割り当てた変数にマッピングする
        self.node_map.insert(node_ptr, result_var.clone());
        result_var
    }
}

/// UOpグラフ（の終端ノード）を、単一のループを持つASTに変換する。
pub fn lower(root: &UOp) -> UOp {
    let mut lowerer = Lowerer::new();

    // グラフの終端から遡ってASTを構築する
    let final_result_var = lowerer.process_node(root);

    // 最終結果を出力バッファにストアする命令を追加する
    // TODO: 出力バッファのインデックスを正しく設定する
    let output_buf = UOp::var("out0", crate::dtype::DType::Unit); // DTypeは仮
    let loop_idx = UOp::var("i", crate::dtype::DType::U64);
    let final_store = UOp::new(
        Op::Store,
        crate::dtype::DType::Unit,
        vec![output_buf, loop_idx, final_result_var],
    );
    lowerer.kernel_body.push(final_store);

    // ループでkernel_bodyをラップする
    let body_block = UOp::new(
        Op::Block,
        crate::dtype::DType::Unit,
        lowerer.kernel_body,
    );

    // TODO: ループの上限を正しく設定する
    let loop_limit = UOp::var("N", crate::dtype::DType::U64);
    let loop_uop = UOp::new(
        Op::Loop,
        crate::dtype::DType::Unit,
        vec![loop_limit, body_block],
    );

    loop_uop
}
