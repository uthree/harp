//! メモリ操作の勾配関数

use super::{GradFn, Tensor};

#[derive(Debug)]
pub struct PadBackward {
    pub padding: Vec<(usize, usize)>,
}

impl GradFn for PadBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Pad requires 1 input");

        let input = &inputs[0];
        let input_shape = input.data.view.shape();

        // パディングを除去する範囲を計算
        // 各軸について: [padding_before, padding_before + original_size]
        let ranges: Vec<(usize, usize)> = self
            .padding
            .iter()
            .enumerate()
            .map(|(i, (before, _after))| {
                let size = input_shape[i].expect_usize("PadBackward requires constant input shape");
                (*before, before + size)
            })
            .collect();

        vec![Some(grad_output.slice(ranges))]
    }
}

/// Slice演算の勾配: 元のサイズのゼロテンソルを作り、slice部分に勾配を配置
///
/// Padの逆操作として実装
#[derive(Debug)]
pub struct SliceBackward {
    pub ranges: Vec<(usize, usize)>,
}

impl GradFn for SliceBackward {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>> {
        assert_eq!(inputs.len(), 1, "Slice requires 1 input");

        let input = &inputs[0];
        let input_shape = input.data.view.shape();

        // Sliceの勾配は、元のサイズに戻すためにパディングを追加
        // 各軸について: padding_before = ranges[i].0, padding_after = input_shape[i] - ranges[i].1

        let padding: Vec<(usize, usize)> = self
            .ranges
            .iter()
            .enumerate()
            .map(|(i, (start, end))| {
                let input_size =
                    input_shape[i].expect_usize("SliceBackward requires constant input shape");
                let padding_before = *start;
                let padding_after = input_size - end;
                (padding_before, padding_after)
            })
            .collect();

        vec![Some(grad_output.pad(padding, 0.0))]
    }
}
