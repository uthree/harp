//! Multi-Head Attention Layer
//!
//! Implements the multi-head attention mechanism as described in
//! "Attention Is All You Need" (Vaswani et al., 2017).

use eclat::tensor::Tensor;
use eclat::tensor::dim::{D1, D2, D3, D4};

use super::{Module, Parameter, ParameterBase};
use crate::functional;

/// Multi-Head Attention layer.
///
/// Allows the model to jointly attend to information from different
/// representation subspaces at different positions.
///
/// ```text
/// MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
/// where head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)
/// ```
///
/// # Example
///
/// ```ignore
/// use eclat_nn::layers::MultiheadAttention;
/// use eclat::tensor::{Tensor, dim::D3};
///
/// // Create attention layer with embed_dim=512, 8 heads
/// let attention = MultiheadAttention::new(512, 8, true);
///
/// // Input: [batch, seq_len, embed_dim]
/// let query: Tensor<D3, f32> = Tensor::input([32, 10, 512]);
/// let key: Tensor<D3, f32> = Tensor::input([32, 10, 512]);
/// let value: Tensor<D3, f32> = Tensor::input([32, 10, 512]);
///
/// let output = attention.forward(&query, &key, &value, None);
/// // output shape: [32, 10, 512]
/// ```
pub struct MultiheadAttention {
    /// Total embedding dimension
    embed_dim: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head (embed_dim / num_heads)
    head_dim: usize,
    /// Query projection weight: [embed_dim, embed_dim]
    w_q: Parameter<D2>,
    /// Key projection weight: [embed_dim, embed_dim]
    w_k: Parameter<D2>,
    /// Value projection weight: [embed_dim, embed_dim]
    w_v: Parameter<D2>,
    /// Output projection weight: [embed_dim, embed_dim]
    w_o: Parameter<D2>,
    /// Query projection bias: [embed_dim]
    b_q: Option<Parameter<D1>>,
    /// Key projection bias: [embed_dim]
    b_k: Option<Parameter<D1>>,
    /// Value projection bias: [embed_dim]
    b_v: Option<Parameter<D1>>,
    /// Output projection bias: [embed_dim]
    b_o: Option<Parameter<D1>>,
    /// Training mode flag
    training: bool,
}

impl MultiheadAttention {
    /// Creates a new MultiheadAttention layer.
    ///
    /// # Arguments
    /// * `embed_dim` - Total dimension of the model (must be divisible by num_heads)
    /// * `num_heads` - Number of parallel attention heads
    /// * `bias` - Whether to include bias in projections
    ///
    /// # Panics
    /// Panics if embed_dim is not divisible by num_heads.
    pub fn new(embed_dim: usize, num_heads: usize, bias: bool) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim ({}) must be divisible by num_heads ({})",
            embed_dim,
            num_heads
        );

        let head_dim = embed_dim / num_heads;

        // Initialize projection weights
        let w_q = Parameter::new("w_q", &[embed_dim, embed_dim]);
        let w_k = Parameter::new("w_k", &[embed_dim, embed_dim]);
        let w_v = Parameter::new("w_v", &[embed_dim, embed_dim]);
        let w_o = Parameter::new("w_o", &[embed_dim, embed_dim]);

        // Initialize biases if requested
        let (b_q, b_k, b_v, b_o) = if bias {
            (
                Some(Parameter::new("b_q", &[embed_dim])),
                Some(Parameter::new("b_k", &[embed_dim])),
                Some(Parameter::new("b_v", &[embed_dim])),
                Some(Parameter::new("b_o", &[embed_dim])),
            )
        } else {
            (None, None, None, None)
        };

        Self {
            embed_dim,
            num_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
            b_q,
            b_k,
            b_v,
            b_o,
            training: true,
        }
    }

    /// Returns the embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Returns the number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Returns the dimension per head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Forward pass of multi-head attention.
    ///
    /// # Arguments
    /// * `query` - Query tensor of shape [batch, seq_len_q, embed_dim]
    /// * `key` - Key tensor of shape [batch, seq_len_k, embed_dim]
    /// * `value` - Value tensor of shape [batch, seq_len_v, embed_dim]
    /// * `attn_mask` - Optional attention mask of shape [batch, num_heads, seq_len_q, seq_len_k]
    ///                 or broadcastable shape. Values of 1.0 indicate positions to attend,
    ///                 0.0 to mask out.
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len_q, embed_dim]
    ///
    /// # Note
    /// seq_len_k must equal seq_len_v
    pub fn forward(
        &self,
        query: &Tensor<D3, f32>,
        key: &Tensor<D3, f32>,
        value: &Tensor<D3, f32>,
        attn_mask: Option<&Tensor<D4, f32>>,
    ) -> Tensor<D3, f32> {
        let batch_size = query.shape()[0];
        let seq_len_q = query.shape()[1];
        let seq_len_k = key.shape()[1];

        // Project Q, K, V: [B, L, E] -> [B, L, E]
        let q = self.project(query, &self.w_q, self.b_q.as_ref());
        let k = self.project(key, &self.w_k, self.b_k.as_ref());
        let v = self.project(value, &self.w_v, self.b_v.as_ref());

        // Reshape to multi-head format: [B, L, E] -> [B, L, H, D] -> [B, H, L, D]
        let q_heads = self.split_heads(&q, batch_size, seq_len_q);
        let k_heads = self.split_heads(&k, batch_size, seq_len_k);
        let v_heads = self.split_heads(&v, batch_size, seq_len_k);

        // Apply scaled dot-product attention
        let attn_output =
            functional::scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, attn_mask);

        // Merge heads: [B, H, L, D] -> [B, L, H, D] -> [B, L, E]
        let merged = self.merge_heads(&attn_output, batch_size, seq_len_q);

        // Output projection: [B, L, E] -> [B, L, E]
        self.project(&merged, &self.w_o, self.b_o.as_ref())
    }

    /// Projects input using weight and optional bias.
    /// input: [B, L, E], weight: [E, E], bias: [E] -> output: [B, L, O]
    ///
    /// Uses reshape to convert 3D to 2D for linear, then back to 3D.
    fn project(
        &self,
        input: &Tensor<D3, f32>,
        weight: &Parameter<D2>,
        bias: Option<&Parameter<D1>>,
    ) -> Tensor<D3, f32> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let in_features = input.shape()[2];
        let out_features = weight.tensor().shape()[0];

        // Reshape [B, L, E] -> [B*L, E]
        let flat: Tensor<D2, f32> = input.reshape([batch_size * seq_len, in_features]);

        // Apply linear: [B*L, E] -> [B*L, O]
        let b = bias.map(|b| b.tensor());
        let projected = functional::linear(&flat, &weight.tensor(), b.as_deref());

        // Reshape [B*L, O] -> [B, L, O]
        projected.reshape([batch_size, seq_len, out_features])
    }

    /// Splits the last dimension into multiple heads.
    /// input: [B, L, E] -> output: [B, H, L, D]
    fn split_heads(
        &self,
        input: &Tensor<D3, f32>,
        batch_size: usize,
        seq_len: usize,
    ) -> Tensor<D4, f32> {
        // [B, L, E] -> [B, L, H, D]
        let reshaped: Tensor<D4, f32> =
            input.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);
        // [B, L, H, D] -> [B, H, L, D]
        reshaped.permute(&[0, 2, 1, 3])
    }

    /// Merges heads back into embed_dim.
    /// input: [B, H, L, D] -> output: [B, L, E]
    fn merge_heads(
        &self,
        input: &Tensor<D4, f32>,
        batch_size: usize,
        seq_len: usize,
    ) -> Tensor<D3, f32> {
        // [B, H, L, D] -> [B, L, H, D]
        let permuted = input.permute(&[0, 2, 1, 3]);
        // Make contiguous after permute for reshape
        let contiguous = permuted.contiguous();
        // [B, L, H, D] -> [B, L, E]
        contiguous.reshape([batch_size, seq_len, self.embed_dim])
    }
}

impl Module for MultiheadAttention {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        let mut params: Vec<Box<dyn ParameterBase>> = vec![
            Box::new(self.w_q.clone()),
            Box::new(self.w_k.clone()),
            Box::new(self.w_v.clone()),
            Box::new(self.w_o.clone()),
        ];

        if let Some(ref b) = self.b_q {
            params.push(Box::new(b.clone()));
        }
        if let Some(ref b) = self.b_k {
            params.push(Box::new(b.clone()));
        }
        if let Some(ref b) = self.b_v {
            params.push(Box::new(b.clone()));
        }
        if let Some(ref b) = self.b_o {
            params.push(Box::new(b.clone()));
        }

        params
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for MultiheadAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiheadAttention")
            .field("embed_dim", &self.embed_dim)
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .field("bias", &self.b_q.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multihead_attention_creation() {
        let mha = MultiheadAttention::new(512, 8, true);
        assert_eq!(mha.embed_dim(), 512);
        assert_eq!(mha.num_heads(), 8);
        assert_eq!(mha.head_dim(), 64);
    }

    #[test]
    fn test_multihead_attention_no_bias() {
        let mha = MultiheadAttention::new(256, 4, false);
        let params = mha.parameters();
        // 4 weights only (no biases)
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_multihead_attention_with_bias() {
        let mha = MultiheadAttention::new(256, 4, true);
        let params = mha.parameters();
        // 4 weights + 4 biases
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_multihead_attention_forward() {
        let mha = MultiheadAttention::new(64, 4, true);

        let batch_size = 2;
        let seq_len = 8;
        let embed_dim = 64;

        let query: Tensor<D3, f32> = Tensor::input([batch_size, seq_len, embed_dim]);
        let key: Tensor<D3, f32> = Tensor::input([batch_size, seq_len, embed_dim]);
        let value: Tensor<D3, f32> = Tensor::input([batch_size, seq_len, embed_dim]);

        let output = mha.forward(&query, &key, &value, None);
        assert_eq!(output.shape(), vec![batch_size, seq_len, embed_dim]);
    }

    #[test]
    fn test_multihead_attention_cross_attention() {
        // Cross-attention: query from decoder, key/value from encoder
        let mha = MultiheadAttention::new(64, 4, true);

        let batch_size = 2;
        let seq_len_q = 10; // decoder sequence length
        let seq_len_kv = 20; // encoder sequence length
        let embed_dim = 64;

        let query: Tensor<D3, f32> = Tensor::input([batch_size, seq_len_q, embed_dim]);
        let key: Tensor<D3, f32> = Tensor::input([batch_size, seq_len_kv, embed_dim]);
        let value: Tensor<D3, f32> = Tensor::input([batch_size, seq_len_kv, embed_dim]);

        let output = mha.forward(&query, &key, &value, None);
        assert_eq!(output.shape(), vec![batch_size, seq_len_q, embed_dim]);
    }

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn test_multihead_attention_invalid_dims() {
        // Should panic: 512 is not divisible by 7
        let _ = MultiheadAttention::new(512, 7, true);
    }

    #[test]
    fn test_multihead_attention_num_parameters() {
        let mha = MultiheadAttention::new(64, 4, true);
        // 4 weights of size 64*64 = 4096 each = 16384
        // 4 biases of size 64 each = 256
        // Total = 16384 + 256 = 16640
        assert_eq!(mha.num_parameters(), 16640);
    }
}
