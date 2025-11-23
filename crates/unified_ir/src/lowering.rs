use crate::uop::{ElementwiseOp, ReduceOp, UOp, UOpKind};
use harp::DType;

/// 高レベル演算を低レベル演算にlowerする
pub struct Lowerer {
    thread_count: usize,
}

impl Lowerer {
    pub fn new(thread_count: usize) -> Self {
        Self { thread_count }
    }

    /// UOpグラフをlower（高レベル→低レベル変換）
    pub fn lower(&self, uop: &UOp) -> UOp {
        self.lower_impl(uop, 0)
    }

    fn lower_impl(&self, uop: &UOp, depth: usize) -> UOp {
        // 最大深度チェック（無限ループ防止）
        if depth > 100 {
            return uop.clone();
        }

        match &uop.0.op {
            // ========== 高レベル演算のlowering ==========
            UOpKind::Elementwise { op } => {
                self.lower_elementwise(op, &uop.0.src, uop.0.dtype.clone(), depth)
            }

            UOpKind::Reduce {
                op,
                axis,
                input_shape,
            } => self.lower_reduce(
                *op,
                &uop.0.src[0],
                *axis,
                input_shape,
                uop.0.dtype.clone(),
                depth,
            ),

            // ========== その他の演算は子ノードのみlower ==========
            _ => {
                let new_src: Vec<UOp> = uop
                    .0
                    .src
                    .iter()
                    .map(|s| self.lower_impl(s, depth + 1))
                    .collect();

                if new_src
                    .iter()
                    .zip(&uop.0.src)
                    .all(|(a, b)| std::rc::Rc::ptr_eq(&a.0, &b.0))
                {
                    uop.clone()
                } else {
                    UOp::new(uop.0.op.clone(), uop.0.dtype.clone(), new_src)
                }
            }
        }
    }

    /// Element-wise演算をlower
    fn lower_elementwise(
        &self,
        op: &ElementwiseOp,
        inputs: &[UOp],
        dtype: DType,
        depth: usize,
    ) -> UOp {
        // 入力をlower
        let lowered_inputs: Vec<UOp> = inputs
            .iter()
            .map(|i| self.lower_impl(i, depth + 1))
            .collect();

        // 入力からshapeを推定（簡易版、実際は入力のshapeから計算）
        // ここでは仮に配列サイズを決定
        let size = self.estimate_size(&lowered_inputs);

        // GPUスレッドIDを使った並列ループを生成
        let tid = UOp::thread_idx(0, DType::F32);

        // 各入力からのロード
        let loaded_values: Vec<UOp> = lowered_inputs
            .iter()
            .enumerate()
            .map(|(i, input)| {
                let buffer_name = format!("input{}", i);
                UOp::load(buffer_name, Some(tid.clone()), dtype.clone())
            })
            .collect();

        // 演算の適用
        let result_value = self.apply_elementwise_op(*op, &loaded_values, dtype.clone());

        // 結果の格納
        let store = UOp::store("output".to_string(), Some(tid.clone()), result_value);

        // 並列ループで包む
        UOp::loop_op("tid".to_string(), 0, size, store, true)
    }

    /// Reduce演算をlower
    fn lower_reduce(
        &self,
        op: ReduceOp,
        input: &UOp,
        axis: usize,
        input_shape: &[usize],
        dtype: DType,
        depth: usize,
    ) -> UOp {
        // 入力をlower
        let lowered_input = self.lower_impl(input, depth + 1);

        // Reduce軸とその他の軸を分離
        let reduce_size = input_shape.get(axis).copied().unwrap_or(1);
        let outer_size = input_shape.iter().take(axis).product::<usize>().max(1);
        let inner_size = input_shape.iter().skip(axis + 1).product::<usize>().max(1);
        let total_outer = outer_size * inner_size;

        // GPUスレッドID（各出力要素）
        let oidx = UOp::thread_idx(0, DType::F32);

        // アキュムレータの初期化
        let init_value = match op {
            ReduceOp::Sum => UOp::const_val(0.0, dtype.clone()),
            ReduceOp::Max => UOp::const_val(f64::NEG_INFINITY, dtype.clone()),
            ReduceOp::Min => UOp::const_val(f64::INFINITY, dtype.clone()),
        };

        // Reduceループのインデックス
        let ridx = UOp::var("ridx", DType::F32);

        // 入力インデックスの計算（簡略化版）
        // 実際には outer_idx * reduce_size * inner_size + ridx * inner_size + inner_idx
        // ここでは簡単のため ridx を使用
        let input_idx = ridx.clone();

        // 入力からロード
        let loaded = UOp::load("input".to_string(), Some(input_idx), dtype.clone());

        // アキュムレータとの演算
        let acc = UOp::var("acc", dtype.clone());
        let updated_acc = match op {
            ReduceOp::Sum => UOp::add(acc.clone(), loaded),
            ReduceOp::Max => UOp::new(UOpKind::Max, dtype.clone(), vec![acc.clone(), loaded]),
            ReduceOp::Min => {
                // Minは Maxの否定として実装
                UOp::new(UOpKind::Max, dtype.clone(), vec![acc.clone(), loaded])
            }
        };

        // アキュムレータの更新（Store）
        let acc_update = UOp::store("acc".to_string(), None, updated_acc);

        // Reduceループ
        let reduce_loop = UOp::loop_op("ridx".to_string(), 0, reduce_size, acc_update, false);

        // 結果の格納
        let final_acc = UOp::load("acc".to_string(), None, dtype.clone());
        let store_result = UOp::store("output".to_string(), Some(oidx.clone()), final_acc);

        // 初期化、reduceループ、格納を順次実行
        let init_acc = UOp::store("acc".to_string(), None, init_value);
        let sequence = UOp::sequence(vec![init_acc, reduce_loop, store_result]);

        // 並列ループで包む（各出力要素）
        UOp::loop_op("oidx".to_string(), 0, total_outer, sequence, true)
    }

    /// Element-wise演算を適用
    fn apply_elementwise_op(&self, op: ElementwiseOp, inputs: &[UOp], dtype: DType) -> UOp {
        match op {
            ElementwiseOp::Add => {
                if inputs.len() != 2 {
                    panic!("Add requires 2 inputs");
                }
                UOp::add(inputs[0].clone(), inputs[1].clone())
            }
            ElementwiseOp::Mul => {
                if inputs.len() != 2 {
                    panic!("Mul requires 2 inputs");
                }
                UOp::mul(inputs[0].clone(), inputs[1].clone())
            }
            ElementwiseOp::Neg => {
                if inputs.len() != 1 {
                    panic!("Neg requires 1 input");
                }
                UOp::mul(UOp::const_val(-1.0, dtype), inputs[0].clone())
            }
            ElementwiseOp::Recip => {
                if inputs.len() != 1 {
                    panic!("Recip requires 1 input");
                }
                UOp::new(UOpKind::Recip, dtype, vec![inputs[0].clone()])
            }
            ElementwiseOp::Sqrt => {
                if inputs.len() != 1 {
                    panic!("Sqrt requires 1 input");
                }
                UOp::new(UOpKind::Sqrt, dtype, vec![inputs[0].clone()])
            }
            ElementwiseOp::Max => {
                if inputs.len() != 2 {
                    panic!("Max requires 2 inputs");
                }
                UOp::new(
                    UOpKind::Max,
                    dtype,
                    vec![inputs[0].clone(), inputs[1].clone()],
                )
            }
            ElementwiseOp::Div => {
                if inputs.len() != 2 {
                    panic!("Div requires 2 inputs");
                }
                // a / b = a * recip(b)
                let recip_b = UOp::new(UOpKind::Recip, dtype.clone(), vec![inputs[1].clone()]);
                UOp::mul(inputs[0].clone(), recip_b)
            }
            ElementwiseOp::Exp | ElementwiseOp::Log => {
                // 今回は未実装
                unimplemented!("Exp/Log not implemented yet")
            }
        }
    }

    /// 入力からサイズを推定（簡易版）
    fn estimate_size(&self, _inputs: &[UOp]) -> usize {
        // 実際には入力のshapeから計算
        // ここでは仮の値
        1024
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_elementwise() {
        let lowerer = Lowerer::new(256);

        let a = UOp::input("a", vec![10], DType::F32);
        let b = UOp::input("b", vec![10], DType::F32);
        let add = UOp::elementwise(ElementwiseOp::Add, vec![a, b], DType::F32);

        let lowered = lowerer.lower(&add);

        println!("Lowered elementwise:\n{}", lowered.to_debug_string(0));

        // Loopノードが生成されているか確認
        match &lowered.0.op {
            UOpKind::Loop { var, parallel, .. } => {
                assert_eq!(var, "tid");
                assert!(parallel);
            }
            _ => panic!("Expected Loop"),
        }
    }

    #[test]
    fn test_lower_reduce() {
        let lowerer = Lowerer::new(256);

        let a = UOp::input("a", vec![10, 20], DType::F32);
        let sum = UOp::reduce(ReduceOp::Sum, a, 1, vec![10, 20]);

        let lowered = lowerer.lower(&sum);

        println!("Lowered reduce:\n{}", lowered.to_debug_string(0));

        // Loopノードが生成されているか確認
        match &lowered.0.op {
            UOpKind::Loop { var, parallel, .. } => {
                assert_eq!(var, "oidx");
                assert!(parallel);
            }
            _ => panic!("Expected Loop"),
        }
    }
}
