//! Attention mechanisms
//!
//! This module provides attention-related functional operations.

use eclat::tensor::Tensor;
use eclat::tensor::dim::D4;

use super::softmax;

/// Batched matrix multiplication for 4D tensors.
///
/// Computes `A @ B` where:
/// - A: [batch, heads, M, K]
/// - B: [batch, heads, K, N]
/// - Output: [batch, heads, M, N]
///
/// # Arguments
/// * `a` - First tensor of shape [batch, heads, M, K]
/// * `b` - Second tensor of shape [batch, heads, K, N]
///
/// # Returns
/// Output tensor of shape [batch, heads, M, N]
fn batched_matmul_d4(a: &Tensor<D4, f32>, b: &Tensor<D4, f32>) -> Tensor<D4, f32> {
    // A: [B, H, M, K] -> [B, H, M, K, 1]
    let a_expanded: Tensor<eclat::tensor::dim::D5, f32> = a.unsqueeze(4);

    // B: [B, H, K, N] -> [B, H, 1, K, N]
    let b_expanded: Tensor<eclat::tensor::dim::D5, f32> = b.unsqueeze(2);

    // Broadcast multiply: [B, H, M, K, 1] * [B, H, 1, K, N] -> [B, H, M, K, N]
    let product = &a_expanded * &b_expanded;

    // Sum over K axis (axis 3): [B, H, M, K, N] -> [B, H, M, N]
    product.sum(3)
}

/// Computes scaled dot-product attention.
///
/// ```text
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
/// ```
///
/// # Arguments
/// * `query` - Query tensor of shape [batch, num_heads, seq_len_q, head_dim]
/// * `key` - Key tensor of shape [batch, num_heads, seq_len_k, head_dim]
/// * `value` - Value tensor of shape [batch, num_heads, seq_len_v, head_dim]
/// * `attn_mask` - Optional attention mask of shape [batch, num_heads, seq_len_q, seq_len_k]
///                 Values of true/1.0 indicate positions to attend, false/0.0 to mask out.
///                 Pass None for no masking.
///
/// # Returns
/// * Output tensor of shape [batch, num_heads, seq_len_q, head_dim]
///
/// # Note
/// - seq_len_k must equal seq_len_v
/// - The scaling factor is `1 / sqrt(head_dim)`
pub fn scaled_dot_product_attention(
    query: &Tensor<D4, f32>,
    key: &Tensor<D4, f32>,
    value: &Tensor<D4, f32>,
    attn_mask: Option<&Tensor<D4, f32>>,
) -> Tensor<D4, f32> {
    // query: [B, H, L_q, D]
    // key: [B, H, L_k, D]
    // value: [B, H, L_v, D]

    // Get head_dim for scaling
    let head_dim = query.shape()[3];
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Transpose key: [B, H, L_k, D] -> [B, H, D, L_k]
    let key_t = key.permute(&[0, 1, 3, 2]);

    // Compute attention scores: Q @ K^T
    // [B, H, L_q, D] @ [B, H, D, L_k] -> [B, H, L_q, L_k]
    let scores = batched_matmul_d4(query, &key_t);

    // Scale scores
    let scaled_scores = scores.scale(scale);

    // Apply attention mask if provided
    let masked_scores = match attn_mask {
        Some(mask) => {
            // mask: 1.0 = attend, 0.0 = mask
            // We need to convert 0s to large negative values
            let neg_inf = Tensor::<D4, f32>::ones_like(&scaled_scores).scale(-1e9);
            let ones = Tensor::<D4, f32>::ones_like(mask);
            let mask_bool = mask.ge(&ones.scale(0.5)); // mask >= 0.5 -> true
            scaled_scores.where_cond(&mask_bool, &neg_inf)
        }
        None => scaled_scores,
    };

    // Softmax over last axis (seq_len_k)
    let attn_weights = softmax(&masked_scores, 3);

    // Apply attention to values: attn_weights @ V
    // [B, H, L_q, L_k] @ [B, H, L_v, D] -> [B, H, L_q, D]
    // Note: L_k == L_v, so we need V in shape [B, H, L_k, D]
    batched_matmul_d4(&attn_weights, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_matmul_d4() {
        let a: Tensor<D4, f32> = Tensor::input([2, 4, 8, 16]); // [B, H, M, K]
        let b: Tensor<D4, f32> = Tensor::input([2, 4, 16, 10]); // [B, H, K, N]
        let c = batched_matmul_d4(&a, &b);
        assert_eq!(c.shape(), vec![2, 4, 8, 10]); // [B, H, M, N]
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        let batch_size = 2;
        let num_heads = 4;
        let seq_len = 8;
        let head_dim = 16;

        let query: Tensor<D4, f32> = Tensor::input([batch_size, num_heads, seq_len, head_dim]);
        let key: Tensor<D4, f32> = Tensor::input([batch_size, num_heads, seq_len, head_dim]);
        let value: Tensor<D4, f32> = Tensor::input([batch_size, num_heads, seq_len, head_dim]);

        let output = scaled_dot_product_attention(&query, &key, &value, None);
        assert_eq!(
            output.shape(),
            vec![batch_size, num_heads, seq_len, head_dim]
        );
    }

    #[test]
    fn test_scaled_dot_product_attention_with_mask() {
        let batch_size = 2;
        let num_heads = 4;
        let seq_len = 8;
        let head_dim = 16;

        let query: Tensor<D4, f32> = Tensor::input([batch_size, num_heads, seq_len, head_dim]);
        let key: Tensor<D4, f32> = Tensor::input([batch_size, num_heads, seq_len, head_dim]);
        let value: Tensor<D4, f32> = Tensor::input([batch_size, num_heads, seq_len, head_dim]);
        let mask: Tensor<D4, f32> = Tensor::input([batch_size, num_heads, seq_len, seq_len]);

        let output = scaled_dot_product_attention(&query, &key, &value, Some(&mask));
        assert_eq!(
            output.shape(),
            vec![batch_size, num_heads, seq_len, head_dim]
        );
    }
}
