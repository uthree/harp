use crate::helper;
use crate::uop::{ElementwiseOp, ReduceOp, UOp};
use harp::DType;
use std::rc::Rc;

/// 高レベル演算を低レベル演算にlowerする
pub struct Lowerer {
    #[allow(dead_code)]
    thread_count: usize,
}

impl Lowerer {
    pub fn new(thread_count: usize) -> Self {
        Self { thread_count }
    }

    /// UOpグラフをlower（高レベル→低レベル変換）
    pub fn lower(&self, uop: &Rc<UOp>) -> Rc<UOp> {
        self.lower_impl(uop, 0)
    }

    fn lower_impl(&self, uop: &Rc<UOp>, depth: usize) -> Rc<UOp> {
        // 最大深度チェック（無限ループ防止）
        if depth > 100 {
            return uop.clone();
        }

        match &**uop {
            // ========== 高レベル演算のlowering ==========
            UOp::Elementwise { op, inputs, dtype } => {
                self.lower_elementwise(op, inputs, dtype.clone(), depth)
            }

            UOp::Reduce {
                op,
                input,
                axis,
                input_shape,
                dtype,
            } => self.lower_reduce(*op, input, *axis, input_shape, dtype.clone(), depth),

            // ========== その他の演算は子ノードのみlower ==========
            _ => self.lower_children(uop, depth),
        }
    }

    /// 子ノードを再帰的にlowerする
    fn lower_children(&self, uop: &Rc<UOp>, depth: usize) -> Rc<UOp> {
        match &**uop {
            UOp::Input { .. }
            | UOp::Const { .. }
            | UOp::ThreadIdx { .. }
            | UOp::GroupIdx { .. }
            | UOp::Var { .. }
            | UOp::Barrier { .. }
            | UOp::Wildcard { .. } => uop.clone(),

            UOp::Loop {
                dtype,
                var,
                start,
                end,
                parallel,
                body,
            } => {
                let new_body = self.lower_impl(body, depth + 1);
                if Rc::ptr_eq(&new_body, body) {
                    uop.clone()
                } else {
                    Rc::new(UOp::Loop {
                        dtype: dtype.clone(),
                        var: var.clone(),
                        start: *start,
                        end: *end,
                        parallel: *parallel,
                        body: new_body,
                    })
                }
            }

            UOp::Load {
                dtype,
                buffer,
                index,
            } => {
                let new_index = index.as_ref().map(|i| self.lower_impl(i, depth + 1));
                let unchanged = match (&new_index, index) {
                    (Some(new), Some(old)) => Rc::ptr_eq(new, old),
                    (None, None) => true,
                    _ => false,
                };
                if unchanged {
                    uop.clone()
                } else {
                    Rc::new(UOp::Load {
                        dtype: dtype.clone(),
                        buffer: buffer.clone(),
                        index: new_index,
                    })
                }
            }

            UOp::Store {
                dtype,
                buffer,
                index,
                value,
            } => {
                let new_index = index.as_ref().map(|i| self.lower_impl(i, depth + 1));
                let new_value = self.lower_impl(value, depth + 1);
                let index_unchanged = match (&new_index, index) {
                    (Some(new), Some(old)) => Rc::ptr_eq(new, old),
                    (None, None) => true,
                    _ => false,
                };
                if index_unchanged && Rc::ptr_eq(&new_value, value) {
                    uop.clone()
                } else {
                    Rc::new(UOp::Store {
                        dtype: dtype.clone(),
                        buffer: buffer.clone(),
                        index: new_index,
                        value: new_value,
                    })
                }
            }

            UOp::Sequence { dtype, ops } => {
                let new_ops: Vec<Rc<UOp>> =
                    ops.iter().map(|o| self.lower_impl(o, depth + 1)).collect();
                let unchanged = new_ops.len() == ops.len()
                    && new_ops
                        .iter()
                        .zip(ops.iter())
                        .all(|(a, b)| Rc::ptr_eq(a, b));
                if unchanged {
                    uop.clone()
                } else {
                    Rc::new(UOp::Sequence {
                        dtype: dtype.clone(),
                        ops: new_ops,
                    })
                }
            }

            UOp::Add { dtype, lhs, rhs } => {
                let new_lhs = self.lower_impl(lhs, depth + 1);
                let new_rhs = self.lower_impl(rhs, depth + 1);
                if Rc::ptr_eq(&new_lhs, lhs) && Rc::ptr_eq(&new_rhs, rhs) {
                    uop.clone()
                } else {
                    Rc::new(UOp::Add {
                        dtype: dtype.clone(),
                        lhs: new_lhs,
                        rhs: new_rhs,
                    })
                }
            }

            UOp::Mul { dtype, lhs, rhs } => {
                let new_lhs = self.lower_impl(lhs, depth + 1);
                let new_rhs = self.lower_impl(rhs, depth + 1);
                if Rc::ptr_eq(&new_lhs, lhs) && Rc::ptr_eq(&new_rhs, rhs) {
                    uop.clone()
                } else {
                    Rc::new(UOp::Mul {
                        dtype: dtype.clone(),
                        lhs: new_lhs,
                        rhs: new_rhs,
                    })
                }
            }

            UOp::Max { dtype, lhs, rhs } => {
                let new_lhs = self.lower_impl(lhs, depth + 1);
                let new_rhs = self.lower_impl(rhs, depth + 1);
                if Rc::ptr_eq(&new_lhs, lhs) && Rc::ptr_eq(&new_rhs, rhs) {
                    uop.clone()
                } else {
                    Rc::new(UOp::Max {
                        dtype: dtype.clone(),
                        lhs: new_lhs,
                        rhs: new_rhs,
                    })
                }
            }

            UOp::Recip { dtype, arg } => {
                let new_arg = self.lower_impl(arg, depth + 1);
                if Rc::ptr_eq(&new_arg, arg) {
                    uop.clone()
                } else {
                    Rc::new(UOp::Recip {
                        dtype: dtype.clone(),
                        arg: new_arg,
                    })
                }
            }

            UOp::Sqrt { dtype, arg } => {
                let new_arg = self.lower_impl(arg, depth + 1);
                if Rc::ptr_eq(&new_arg, arg) {
                    uop.clone()
                } else {
                    Rc::new(UOp::Sqrt {
                        dtype: dtype.clone(),
                        arg: new_arg,
                    })
                }
            }

            // 残りのバリアントは子ノードのlowerのみ
            _ => uop.clone(),
        }
    }

    /// Element-wise演算をlower
    fn lower_elementwise(
        &self,
        op: &ElementwiseOp,
        inputs: &[Rc<UOp>],
        dtype: DType,
        depth: usize,
    ) -> Rc<UOp> {
        // 入力をlower
        let lowered_inputs: Vec<Rc<UOp>> = inputs
            .iter()
            .map(|i| self.lower_impl(i, depth + 1))
            .collect();

        // 入力からshapeを推定（簡易版、実際は入力のshapeから計算）
        let size = self.estimate_size(&lowered_inputs);

        // GPUスレッドIDを使った並列ループを生成
        let tid = helper::thread_idx(0, DType::F32);

        // 各入力からのロード
        let loaded_values: Vec<Rc<UOp>> = lowered_inputs
            .iter()
            .enumerate()
            .map(|(i, _input)| {
                let buffer_name = format!("input{}", i);
                helper::load(buffer_name, Some(tid.clone()), dtype.clone())
            })
            .collect();

        // 演算の適用
        let result_value = self.apply_elementwise_op(*op, &loaded_values, dtype.clone());

        // 結果の格納
        let store = helper::store("output", Some(tid.clone()), result_value);

        // 並列ループで包む
        helper::loop_op("tid", 0, size, store, true)
    }

    /// Reduce演算をlower
    fn lower_reduce(
        &self,
        op: ReduceOp,
        input: &Rc<UOp>,
        axis: usize,
        input_shape: &[usize],
        dtype: DType,
        depth: usize,
    ) -> Rc<UOp> {
        // 入力をlower
        let _lowered_input = self.lower_impl(input, depth + 1);

        // Reduce軸とその他の軸を分離
        let reduce_size = input_shape.get(axis).copied().unwrap_or(1);
        let outer_size = input_shape.iter().take(axis).product::<usize>().max(1);
        let inner_size = input_shape.iter().skip(axis + 1).product::<usize>().max(1);
        let total_outer = outer_size * inner_size;

        // GPUスレッドID（各出力要素）
        let oidx = helper::thread_idx(0, DType::F32);

        // アキュムレータの初期化
        let init_value = match op {
            ReduceOp::Sum => helper::const_val(0.0, dtype.clone()),
            ReduceOp::Max => helper::const_val(f64::NEG_INFINITY, dtype.clone()),
            ReduceOp::Min => helper::const_val(f64::INFINITY, dtype.clone()),
        };

        // Reduceループのインデックス
        let ridx = helper::var("ridx", DType::F32);

        // 入力インデックスの計算（簡略化版）
        let input_idx = ridx.clone();

        // 入力からロード
        let loaded = helper::load("input", Some(input_idx), dtype.clone());

        // アキュムレータとの演算
        let acc = helper::var("acc", dtype.clone());
        let updated_acc = match op {
            ReduceOp::Sum => helper::add(acc.clone(), loaded),
            ReduceOp::Max | ReduceOp::Min => helper::max(acc.clone(), loaded),
        };

        // アキュムレータの更新（Store）
        let acc_update = helper::store("acc", None, updated_acc);

        // Reduceループ
        let reduce_loop = helper::loop_op("ridx", 0, reduce_size, acc_update, false);

        // 結果の格納
        let final_acc = helper::load("acc", None, dtype.clone());
        let store_result = helper::store("output", Some(oidx.clone()), final_acc);

        // 初期化、reduceループ、格納を順次実行
        let init_acc = helper::store("acc", None, init_value);
        let sequence = helper::sequence(vec![init_acc, reduce_loop, store_result]);

        // 並列ループで包む（各出力要素）
        helper::loop_op("oidx", 0, total_outer, sequence, true)
    }

    /// Element-wise演算を適用
    fn apply_elementwise_op(&self, op: ElementwiseOp, inputs: &[Rc<UOp>], dtype: DType) -> Rc<UOp> {
        match op {
            ElementwiseOp::Add => {
                if inputs.len() != 2 {
                    panic!("Add requires 2 inputs");
                }
                helper::add(inputs[0].clone(), inputs[1].clone())
            }
            ElementwiseOp::Mul => {
                if inputs.len() != 2 {
                    panic!("Mul requires 2 inputs");
                }
                helper::mul(inputs[0].clone(), inputs[1].clone())
            }
            ElementwiseOp::Neg => {
                if inputs.len() != 1 {
                    panic!("Neg requires 1 input");
                }
                helper::mul(helper::const_val(-1.0, dtype), inputs[0].clone())
            }
            ElementwiseOp::Recip => {
                if inputs.len() != 1 {
                    panic!("Recip requires 1 input");
                }
                helper::recip(inputs[0].clone())
            }
            ElementwiseOp::Sqrt => {
                if inputs.len() != 1 {
                    panic!("Sqrt requires 1 input");
                }
                helper::sqrt(inputs[0].clone())
            }
            ElementwiseOp::Max => {
                if inputs.len() != 2 {
                    panic!("Max requires 2 inputs");
                }
                helper::max(inputs[0].clone(), inputs[1].clone())
            }
            ElementwiseOp::Div => {
                if inputs.len() != 2 {
                    panic!("Div requires 2 inputs");
                }
                // a / b = a * recip(b)
                let recip_b = helper::recip(inputs[1].clone());
                helper::mul(inputs[0].clone(), recip_b)
            }
            ElementwiseOp::Exp | ElementwiseOp::Log => {
                // 今回は未実装
                unimplemented!("Exp/Log not implemented yet")
            }
        }
    }

    /// 入力からサイズを推定（簡易版）
    fn estimate_size(&self, _inputs: &[Rc<UOp>]) -> usize {
        // 実際には入力のshapeから計算
        // ここでは仮の値
        1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helper;

    #[test]
    fn test_lower_elementwise() {
        let lowerer = Lowerer::new(256);

        let a = helper::input("a", vec![10], DType::F32);
        let b = helper::input("b", vec![10], DType::F32);
        let add = helper::elementwise(ElementwiseOp::Add, vec![a, b], DType::F32);

        let lowered = lowerer.lower(&add);

        println!("Lowered elementwise:\n{}", lowered.to_debug_string(0));

        // Loopノードが生成されているか確認
        match &*lowered {
            UOp::Loop { var, parallel, .. } => {
                assert_eq!(var, "tid");
                assert!(parallel);
            }
            _ => panic!("Expected Loop"),
        }
    }

    #[test]
    fn test_lower_reduce() {
        let lowerer = Lowerer::new(256);

        let a = helper::input("a", vec![10, 20], DType::F32);
        let sum = helper::reduce(ReduceOp::Sum, a, 1, vec![10, 20]);

        let lowered = lowerer.lower(&sum);

        println!("Lowered reduce:\n{}", lowered.to_debug_string(0));

        // Loopノードが生成されているか確認
        match &*lowered {
            UOp::Loop { var, parallel, .. } => {
                assert_eq!(var, "oidx");
                assert!(parallel);
            }
            _ => panic!("Expected Loop"),
        }
    }
}
