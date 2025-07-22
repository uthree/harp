//! `UOp`グラフを`UOp`ツリー（AST）に変換するLowering処理を実装するモジュール。

use crate::uop::{Op, UOp};
use std::collections::HashMap;
use std::rc::Rc;

/// Lowering処理の状態を保持する構造体。
struct Lowerer {
    node_map: HashMap<*const crate::uop::UOp_, UOp>,
    kernel_body: Vec<UOp>,
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

    fn new_var(&mut self, prefix: &str) -> String {
        let name = format!("{}{}", prefix, self.var_counter);
        self.var_counter += 1;
        name
    }

    fn process_node(&mut self, node: &UOp) -> UOp {
        let node_ptr = Rc::as_ptr(&node.0);
        if let Some(mapped) = self.node_map.get(&node_ptr) {
            return mapped.clone();
        }

        let new_srcs: Vec<UOp> = node.0.src.iter().map(|src| self.process_node(src)).collect();

        let result_var = match node.0.op {
            Op::Const(_) | Op::Var(_) => {
                return UOp::new(node.0.op.clone(), node.0.dtype.clone(), new_srcs);
            }
            _ => {
                let var_name = self.new_var("v");
                let var_dtype = node.0.dtype.clone();
                let var_uop = UOp::var(&var_name, var_dtype.clone());
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

        self.node_map.insert(node_ptr, result_var.clone());
        result_var
    }
}

/// UOpグラフ（の終端ノード）を、単一のループを持つASTに変換する。
pub fn lower(root: &UOp) -> UOp {
    let mut lowerer = Lowerer::new();
    let final_result_var = lowerer.process_node(root);

    // 最終結果を出力バッファにストアする
    let output_buf = UOp::var("out0", root.0.dtype.clone());
    let loop_idx = UOp::var("i", crate::dtype::DType::U64);
    let final_store = UOp::new(
        Op::Store,
        crate::dtype::DType::Unit,
        vec![output_buf, loop_idx, final_result_var],
    );
    lowerer.kernel_body.push(final_store);

    let body_block = UOp::new(
        Op::Block,
        crate::dtype::DType::Unit,
        lowerer.kernel_body,
    );

    let loop_limit = UOp::var("N", crate::dtype::DType::U64);
    UOp::new(
        Op::Loop,
        crate::dtype::DType::Unit,
        vec![loop_limit, body_block],
    )
}
