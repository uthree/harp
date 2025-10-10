use crate::graph::ops::cumulative::CumulativeOps;
use crate::graph::ops::reduce::ReduceOps;
use crate::graph::GraphNode;
use std::rc::Rc;

/// Unique identifier for tensors in the autograd graph
pub type TensorId = usize;

/// Trait for defining gradient functions.
/// Each operation implements this to specify how gradients flow backward.
pub trait GradFn: std::fmt::Debug {
    /// Compute gradients for inputs given the gradient of the output.
    ///
    /// # Arguments
    /// * `grad_output` - The gradient flowing back from the output
    /// * `inputs` - The input GraphNodes that were used in the forward pass
    ///
    /// # Returns
    /// A vector of gradients, one for each input. If an input doesn't need gradients,
    /// the corresponding entry can be None.
    fn backward(&self, grad_output: GraphNode, inputs: &[GraphNode]) -> Vec<Option<GraphNode>>;

    /// Get a human-readable name for this gradient function (for debugging)
    fn name(&self) -> &'static str;
}

/// Metadata for gradient tracking.
/// Stores information needed to compute gradients during backward pass.
#[derive(Clone)]
pub struct TensorMeta {
    /// The GraphNode representing this tensor in the computation graph
    pub graph_node: GraphNode,

    /// The gradient function that computes gradients for this operation
    pub grad_fn: Option<Rc<dyn GradFn>>,

    /// Input tensors that this tensor depends on
    /// Stored as (TensorId, GraphNode) pairs
    pub inputs: Vec<(TensorId, GraphNode)>,

    /// Whether this tensor requires gradient computation
    pub requires_grad: bool,
}

impl TensorMeta {
    /// Create a new TensorMeta for a leaf tensor (no gradient function)
    pub fn leaf(graph_node: GraphNode, requires_grad: bool) -> Self {
        TensorMeta {
            graph_node,
            grad_fn: None,
            inputs: Vec::new(),
            requires_grad,
        }
    }

    /// Create a new TensorMeta for a non-leaf tensor (has gradient function)
    pub fn non_leaf(
        graph_node: GraphNode,
        grad_fn: Rc<dyn GradFn>,
        inputs: Vec<(TensorId, GraphNode)>,
    ) -> Self {
        TensorMeta {
            graph_node,
            grad_fn: Some(grad_fn),
            inputs,
            requires_grad: true, // Non-leaf tensors always require grad
        }
    }
}

// Example gradient functions for basic operations

/// Gradient function for addition: grad flows equally to both inputs
#[derive(Debug)]
pub struct AddBackward;

impl GradFn for AddBackward {
    fn backward(&self, grad_output: GraphNode, _inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = x + y:
        // dL/dx = dL/dz * dz/dx = dL/dz * 1 = dL/dz
        // dL/dy = dL/dz * dz/dy = dL/dz * 1 = dL/dz
        vec![Some(grad_output.clone()), Some(grad_output)]
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// Gradient function for multiplication
#[derive(Debug)]
pub struct MulBackward;

impl GradFn for MulBackward {
    fn backward(&self, grad_output: GraphNode, inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = x * y:
        // dL/dx = dL/dz * dz/dx = dL/dz * y
        // dL/dy = dL/dz * dz/dy = dL/dz * x
        assert_eq!(inputs.len(), 2, "MulBackward expects 2 inputs");
        let x = &inputs[0];
        let y = &inputs[1];

        vec![
            Some(grad_output.clone() * y.clone()),
            Some(grad_output * x.clone()),
        ]
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

/// Gradient function for negation
#[derive(Debug)]
pub struct NegBackward;

impl GradFn for NegBackward {
    fn backward(&self, grad_output: GraphNode, _inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = -x:
        // dL/dx = dL/dz * dz/dx = dL/dz * (-1) = -dL/dz
        vec![Some(-grad_output)]
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

/// Gradient function for reciprocal
#[derive(Debug)]
pub struct RecipBackward;

impl GradFn for RecipBackward {
    fn backward(&self, grad_output: GraphNode, inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = 1/x:
        // dL/dx = dL/dz * dz/dx = dL/dz * (-1/x^2)
        assert_eq!(inputs.len(), 1, "RecipBackward expects 1 input");
        let x = &inputs[0];

        let grad_x = (-grad_output) * (x.clone() * x.clone()).recip();

        vec![Some(grad_x)]
    }

    fn name(&self) -> &'static str {
        "RecipBackward"
    }
}

/// Gradient function for sum reduction
#[derive(Debug)]
pub struct SumBackward {
    pub axis: usize,
}

impl GradFn for SumBackward {
    fn backward(&self, grad_output: GraphNode, inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = sum(x, axis):
        // dL/dx = broadcast(dL/dz, original_shape)
        // The gradient needs to be broadcasted back to the original shape
        assert_eq!(inputs.len(), 1, "SumBackward expects 1 input");
        let x = &inputs[0];

        // unsqueezeで縮約した軸を復元し、expandで元の形状にブロードキャスト
        let grad_x = grad_output
            .unsqueeze(self.axis)
            .expand(x.view.shape().to_vec());

        vec![Some(grad_x)]
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

/// Gradient function for product reduction
#[derive(Debug)]
pub struct ProductBackward {
    pub axis: usize,
}

impl GradFn for ProductBackward {
    fn backward(&self, grad_output: GraphNode, inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = product(x, axis):
        // dL/dx = dL/dz * (z / x)
        // where z is the product result and needs to be broadcasted
        assert_eq!(inputs.len(), 1, "ProductBackward expects 1 input");
        let x = &inputs[0];

        // 出力を計算（forward passの結果を再計算）
        let output = x.clone().product(self.axis);

        // grad_outputをunsqueeze + expand
        let grad_expanded = grad_output
            .unsqueeze(self.axis)
            .expand(x.view.shape().to_vec());

        // outputをunsqueeze + expand
        let output_expanded = output.unsqueeze(self.axis).expand(x.view.shape().to_vec());

        // grad_x = grad_output * (output / x)
        let grad_x = grad_expanded * (output_expanded * x.clone().recip());

        vec![Some(grad_x)]
    }

    fn name(&self) -> &'static str {
        "ProductBackward"
    }
}

/// Gradient function for max reduction
#[derive(Debug)]
pub struct MaxBackward {
    pub axis: usize,
}

impl GradFn for MaxBackward {
    fn backward(&self, grad_output: GraphNode, inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = max(x, axis):
        // dL/dx = dL/dz * (x == z).cast()
        // Gradient flows only to the maximum elements
        assert_eq!(inputs.len(), 1, "MaxBackward expects 1 input");
        let x = &inputs[0];

        // 出力を計算（forward passの結果を再計算）
        let output = x.clone().max(self.axis);

        // grad_outputとoutputをunsqueeze + expand
        let grad_expanded = grad_output
            .unsqueeze(self.axis)
            .expand(x.view.shape().to_vec());
        let _output_expanded = output.unsqueeze(self.axis).expand(x.view.shape().to_vec());

        // x == output の mask を作成
        // 注: 現在のGraphNodeでは直接比較演算がないため、
        // (x - output).abs() < epsilon のような近似を使うか、
        // または単純に (x - output) == 0 を仮定
        // ここでは、より安全な方法として、等しい場合は勾配を流す
        // mask = (x == output) を実装できないので、
        // 代わりに x - output が 0 に近いかを確認する必要がある
        //
        // しかし、GraphNodeレベルでは比較演算がないため、
        // バックエンドに任せる形で実装する必要がある
        //
        // 一旦、簡易実装として勾配をそのまま返す
        // （正しい実装にはGraphNodeに比較演算の追加が必要）
        let grad_x = grad_expanded;

        vec![Some(grad_x)]
    }

    fn name(&self) -> &'static str {
        "MaxBackward"
    }
}

/// Gradient function for cumulative sum
#[derive(Debug)]
pub struct CumsumBackward {
    pub axis: usize,
}

impl GradFn for CumsumBackward {
    fn backward(&self, grad_output: GraphNode, _inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = cumsum(x, axis):
        // dL/dx[i] = sum(dL/dz[i:]) (reverse cumulative sum)
        //
        // cumsum: y[i] = sum(x[0:i+1])
        // gradient: dx[i] = sum(dy[i:])
        //
        // This can be implemented as: flip -> cumsum -> flip
        let grad_x = grad_output
            .flip(self.axis)
            .cumsum(self.axis)
            .flip(self.axis);

        vec![Some(grad_x)]
    }

    fn name(&self) -> &'static str {
        "CumsumBackward"
    }
}

/// Gradient function for cumulative product
#[derive(Debug)]
pub struct CumprodBackward {
    pub axis: usize,
}

impl GradFn for CumprodBackward {
    fn backward(&self, grad_output: GraphNode, inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = cumprod(x, axis):
        // This is complex to implement correctly, especially with zeros
        // A simplified approach: dy[i] = grad_output[i] * (output[i] / x[i])
        // then apply reverse cumsum
        //
        // TODO: Implement proper gradient computation
        // For now, return a simplified version
        assert_eq!(inputs.len(), 1, "CumprodBackward expects 1 input");
        let x = &inputs[0];

        // 出力を計算（forward passの結果を再計算）
        let output = x.clone().cumprod(self.axis);

        // grad_x = grad_output * (output / x)
        // This is a simplified version and may not be entirely correct
        let grad_x = grad_output * (output * x.clone().recip());

        vec![Some(grad_x)]
    }

    fn name(&self) -> &'static str {
        "CumprodBackward"
    }
}

/// Gradient function for cumulative max
#[derive(Debug)]
pub struct CummaxBackward {
    pub axis: usize,
}

impl GradFn for CummaxBackward {
    fn backward(&self, grad_output: GraphNode, _inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = cummax(x, axis):
        // Gradient flows only to positions where the maximum was updated
        //
        // TODO: Implement proper gradient computation with comparison operations
        // For now, return a simplified version
        //
        // 簡易実装: 勾配をそのまま返す
        // （正しい実装には比較演算が必要）
        let grad_x = grad_output;

        vec![Some(grad_x)]
    }

    fn name(&self) -> &'static str {
        "CummaxBackward"
    }
}

/// Gradient function for unfold
#[derive(Debug)]
pub struct UnfoldBackward {
    pub dim: usize,
    pub window_size: usize,
    pub stride: usize,
    pub dilation: usize,
}

/// Gradient function for fold
#[derive(Debug)]
pub struct FoldBackward {
    pub dim: usize,
    pub window_size: usize,
    pub stride: usize,
    pub dilation: usize,
}

impl GradFn for UnfoldBackward {
    fn backward(&self, grad_output: GraphNode, inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = unfold(x, dim, window_size, stride, dilation):
        // unfoldは値を複製したViewを作成する操作なので、
        // 逆伝搬では複製された値の勾配を元の位置に加算で縮約する必要がある
        //
        // 入力: [B, C, L]
        // 出力: [B, C, L', K] where L' = (L - (K-1)*D - 1) / S + 1
        // 勾配: [B, C, L', K] -> [B, C, L]
        //
        // これはfold操作で実現できる:
        // grad_output[..., i, k] は入力の位置 i*stride + k*dilation に対応し、
        // foldがこれらを適切に加算してくれる

        assert_eq!(inputs.len(), 1, "UnfoldBackward expects 1 input");
        let x = &inputs[0];
        let input_shape = x.view.shape();
        let output_size = match &input_shape[self.dim] {
            crate::graph::shape::Expr::Const(c) => *c as usize,
            _ => panic!("output size must be constant"),
        };

        // foldを使って元の形状に戻す
        let grad_x = grad_output.fold(
            self.dim,
            self.window_size,
            self.stride,
            self.dilation,
            output_size,
        );

        vec![Some(grad_x)]
    }

    fn name(&self) -> &'static str {
        "UnfoldBackward"
    }
}

impl GradFn for FoldBackward {
    fn backward(&self, grad_output: GraphNode, _inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = fold(x, dim, window_size, stride, dilation, output_size):
        // foldは重複する位置の値を加算する操作なので、
        // 逆伝搬ではunfoldを使って勾配を各ウィンドウ位置に複製する
        //
        // 入力: [B, C, L', K]
        // 出力: [B, C, L]
        // 勾配: [B, C, L] -> [B, C, L', K]
        //
        // これはunfold操作で実現できる:
        // grad_output[..., i*stride + k*dilation] は grad_input[..., i, k] に対応

        let grad_x = grad_output.unfold(self.dim, self.window_size, self.stride, self.dilation);

        vec![Some(grad_x)]
    }

    fn name(&self) -> &'static str {
        "FoldBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::Graph;

    #[test]
    fn test_add_backward() {
        let mut graph = Graph::new();
        let grad = graph.input(DType::F32, vec![2.into(), 3.into()]);
        let x = graph.input(DType::F32, vec![2.into(), 3.into()]);
        let y = graph.input(DType::F32, vec![2.into(), 3.into()]);

        let grad_fn = AddBackward;
        let grads = grad_fn.backward(grad.clone(), &[x, y]);

        assert_eq!(grads.len(), 2);
        assert!(grads[0].is_some());
        assert!(grads[1].is_some());
    }

    #[test]
    fn test_mul_backward() {
        let mut graph = Graph::new();
        let grad = graph.input(DType::F32, vec![2.into(), 3.into()]);
        let x = graph.input(DType::F32, vec![2.into(), 3.into()]);
        let y = graph.input(DType::F32, vec![2.into(), 3.into()]);

        let grad_fn = MulBackward;
        let grads = grad_fn.backward(grad.clone(), &[x.clone(), y.clone()]);

        assert_eq!(grads.len(), 2);
        assert!(grads[0].is_some());
        assert!(grads[1].is_some());
    }
}
