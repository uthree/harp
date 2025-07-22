//! `UOp`グラフを`UOp`ツリー（AST）に変換するLowering処理を実装するモジュール。

use crate::uop::{UOp, Op};
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
    fn new_var(&mut self) -> String {
        let name = format!("v{}", self.var_counter);
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
            // Constノードはそのまま返す（変数に格納する必要はない）
            Op::Const(_) => {
                let new_node = UOp::new(node.0.op.clone(), node.0.dtype.clone(), new_srcs);
                self.node_map.insert(node_ptr, new_node.clone());
                return new_node;
            }
            // Varノードもそのまま返す
            Op::Var(_) => {
                let new_node = UOp::new(node.0.op.clone(), node.0.dtype.clone(), new_srcs);
                self.node_map.insert(node_ptr, new_node.clone());
                return new_node;
            }
            // その他のノードは、計算を行い、結果を新しい変数に格納する
            _ => {
                let var_name = self.new_var();
                let var_dtype = node.0.dtype.clone();
                let var_uop = UOp::var(&var_name, var_dtype.clone());

                // `v1 = a + b;` のようなステートメントを生成
                let statement = UOp::new(node.0.op.clone(), var_dtype, new_srcs);
                
                // TODO: 本来はStore命令を使うべきだが、簡単のため、
                // Renderer側で「v1 = ...」のように解釈する前提で進める。
                // ここでは、生成した計算ノードをそのままbodyに追加する。
                self.kernel_body.push(statement);

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

    // TODO: 最終結果をStoreする命令を追加する
    // let final_store = UOp::new(Op::Store, DType::Unit, vec![...]);
    // lowerer.kernel_body.push(final_store);

    // ループでkernel_bodyをラップする
    let body_block = UOp::new(Op::Block, crate::dtype::DType::Unit, lowerer.kernel_body);
    
    // TODO: ループの上限を正しく設定する
    let loop_limit = UOp::var("N", crate::dtype::DType::U64);
    let loop_uop = UOp::new(Op::Loop, crate::dtype::DType::Unit, vec![loop_limit, body_block]);

    loop_uop
}
