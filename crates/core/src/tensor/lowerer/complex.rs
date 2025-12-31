//! Complex number lowering helpers
//!
//! This module provides helper functions for lowering complex tensor operations
//! to operations on real and imaginary parts.
//!
//! ## Memory Layout
//!
//! Complex numbers use an interleaved memory layout:
//! ```text
//! [re0, im0, re1, im1, re2, im2, ...]
//! ```
//!
//! For a complex value at logical index `i`:
//! - Real part is at memory offset `2 * i`
//! - Imaginary part is at memory offset `2 * i + 1`
//!
//! ## Lowering Strategy
//!
//! Complex operations are decomposed into real operations during lowering:
//!
//! - `Real(z)` → Load from `2 * offset`
//! - `Imag(z)` → Load from `2 * offset + 1`
//! - `Conj(z)` → `MakeComplex(Real(z), -Imag(z))`
//! - `MakeComplex(re, im)` → Store `re` to `2 * offset`, `im` to `2 * offset + 1`
//!
//! ## Arithmetic Decomposition
//!
//! - `(a+bi) + (c+di) = (a+c) + (b+d)i`
//! - `(a+bi) * (c+di) = (ac-bd) + (ad+bc)i`
//! - `(a+bi) / (c+di) = ((ac+bd) + (bc-ad)i) / (c² + d²)`
//! - `-(a+bi) = (-a) + (-b)i`
//! - `1/(a+bi) = (a - bi) / (a² + b²)`

use crate::ast::helper::*;
use crate::ast::{AstNode, DType, Literal};

// ============================================================================
// Complex memory offset helpers
// ============================================================================

/// Get the memory offset for the real part of a complex value at logical index
///
/// For interleaved layout: offset_real = 2 * logical_offset
pub fn real_offset(logical_offset: AstNode) -> AstNode {
    logical_offset * const_int(2)
}

/// Get the memory offset for the imaginary part of a complex value at logical index
///
/// For interleaved layout: offset_imag = 2 * logical_offset + 1
pub fn imag_offset(logical_offset: AstNode) -> AstNode {
    logical_offset * const_int(2) + const_int(1)
}

// ============================================================================
// Complex load/store helpers
// ============================================================================

/// Load the real part of a complex value from a buffer
///
/// # Arguments
/// * `buffer` - Buffer variable name
/// * `logical_offset` - Logical index of the complex value
/// * `element_dtype` - DType of the real element (F32 or F64)
pub fn load_complex_real(buffer: &str, logical_offset: AstNode, element_dtype: DType) -> AstNode {
    load(var(buffer), real_offset(logical_offset), element_dtype)
}

/// Load the imaginary part of a complex value from a buffer
///
/// # Arguments
/// * `buffer` - Buffer variable name
/// * `logical_offset` - Logical index of the complex value
/// * `element_dtype` - DType of the imaginary element (F32 or F64)
pub fn load_complex_imag(buffer: &str, logical_offset: AstNode, element_dtype: DType) -> AstNode {
    load(var(buffer), imag_offset(logical_offset), element_dtype)
}

/// Store a complex value (real and imaginary parts) to a buffer
///
/// # Arguments
/// * `buffer` - Buffer variable name
/// * `logical_offset` - Logical index of the complex value
/// * `real_value` - Real part to store
/// * `imag_value` - Imaginary part to store
pub fn store_complex(
    buffer: &str,
    logical_offset: AstNode,
    real_value: AstNode,
    imag_value: AstNode,
) -> Vec<AstNode> {
    vec![
        store(var(buffer), real_offset(logical_offset.clone()), real_value),
        store(var(buffer), imag_offset(logical_offset), imag_value),
    ]
}

// ============================================================================
// Complex arithmetic decomposition
// ============================================================================

/// Decompose complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i
///
/// # Returns
/// (real_result, imag_result)
pub fn lower_complex_add(
    a_re: AstNode,
    a_im: AstNode,
    b_re: AstNode,
    b_im: AstNode,
) -> (AstNode, AstNode) {
    (a_re + b_re, a_im + b_im)
}

/// Decompose complex subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i
///
/// # Returns
/// (real_result, imag_result)
pub fn lower_complex_sub(
    a_re: AstNode,
    a_im: AstNode,
    b_re: AstNode,
    b_im: AstNode,
) -> (AstNode, AstNode) {
    (a_re - b_re, a_im - b_im)
}

/// Decompose complex multiplication: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
///
/// # Returns
/// (real_result, imag_result)
pub fn lower_complex_mul(
    a_re: AstNode,
    a_im: AstNode,
    b_re: AstNode,
    b_im: AstNode,
) -> (AstNode, AstNode) {
    // (a+bi) * (c+di) = (ac - bd) + (ad + bc)i
    let ac = a_re.clone() * b_re.clone();
    let bd = a_im.clone() * b_im.clone();
    let ad = a_re * b_im;
    let bc = a_im * b_re;

    (ac - bd, ad + bc)
}

/// Decompose complex division: (a+bi) / (c+di) = ((ac+bd) + (bc-ad)i) / (c² + d²)
///
/// # Returns
/// (real_result, imag_result)
pub fn lower_complex_div(
    a_re: AstNode,
    a_im: AstNode,
    b_re: AstNode,
    b_im: AstNode,
) -> (AstNode, AstNode) {
    // (a+bi) / (c+di) = (a+bi) * conj(c+di) / |c+di|²
    //                 = ((ac + bd) + (bc - ad)i) / (c² + d²)
    let norm_sq = b_re.clone() * b_re.clone() + b_im.clone() * b_im.clone();

    let ac = a_re.clone() * b_re.clone();
    let bd = a_im.clone() * b_im.clone();
    let bc = a_im * b_re;
    let ad = a_re * b_im;

    let real_num = ac + bd;
    let imag_num = bc - ad;

    (real_num / norm_sq.clone(), imag_num / norm_sq)
}

/// Decompose complex negation: -(a+bi) = (-a) + (-b)i
///
/// # Returns
/// (real_result, imag_result)
pub fn lower_complex_neg(a_re: AstNode, a_im: AstNode) -> (AstNode, AstNode) {
    (-a_re, -a_im)
}

/// Decompose complex reciprocal: 1/(a+bi) = (a - bi) / (a² + b²)
///
/// # Returns
/// (real_result, imag_result)
pub fn lower_complex_recip(a_re: AstNode, a_im: AstNode) -> (AstNode, AstNode) {
    let norm_sq = a_re.clone() * a_re.clone() + a_im.clone() * a_im.clone();

    (a_re / norm_sq.clone(), -a_im / norm_sq)
}

/// Decompose complex conjugate: conj(a+bi) = a - bi
///
/// # Returns
/// (real_result, imag_result)
pub fn lower_complex_conj(a_re: AstNode, a_im: AstNode) -> (AstNode, AstNode) {
    (a_re, -a_im)
}

// ============================================================================
// Type helpers
// ============================================================================

/// Check if a DType is a complex type
pub fn is_complex_dtype(dtype: &DType) -> bool {
    matches!(dtype, DType::Complex32 | DType::Complex64)
}

/// Get the element (real) dtype for a complex dtype
pub fn element_dtype(dtype: &DType) -> DType {
    match dtype {
        DType::Complex32 => DType::F32,
        DType::Complex64 => DType::F64,
        _ => dtype.clone(),
    }
}

/// Get the complex zero literal for a given complex dtype
pub fn complex_zero(dtype: &DType) -> AstNode {
    match dtype {
        DType::Complex32 => AstNode::Const(Literal::Complex32(0.0, 0.0)),
        DType::Complex64 => AstNode::Const(Literal::Complex64(0.0, 0.0)),
        _ => panic!("complex_zero called with non-complex dtype: {:?}", dtype),
    }
}

/// Get the complex one literal for a given complex dtype
pub fn complex_one(dtype: &DType) -> AstNode {
    match dtype {
        DType::Complex32 => AstNode::Const(Literal::Complex32(1.0, 0.0)),
        DType::Complex64 => AstNode::Const(Literal::Complex64(1.0, 0.0)),
        _ => panic!("complex_one called with non-complex dtype: {:?}", dtype),
    }
}

/// Get the complex i literal for a given complex dtype
pub fn complex_i(dtype: &DType) -> AstNode {
    match dtype {
        DType::Complex32 => AstNode::Const(Literal::Complex32(0.0, 1.0)),
        DType::Complex64 => AstNode::Const(Literal::Complex64(0.0, 1.0)),
        _ => panic!("complex_i called with non-complex dtype: {:?}", dtype),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_offset() {
        let offset = real_offset(const_int(5));
        // Should be 5 * 2 = 10
        match offset {
            AstNode::Mul(_, _) => {} // 5 * 2
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_imag_offset() {
        let offset = imag_offset(const_int(5));
        // Should be 5 * 2 + 1 = 11
        match offset {
            AstNode::Add(_, _) => {} // (5 * 2) + 1
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_is_complex_dtype() {
        assert!(is_complex_dtype(&DType::Complex32));
        assert!(is_complex_dtype(&DType::Complex64));
        assert!(!is_complex_dtype(&DType::F32));
        assert!(!is_complex_dtype(&DType::F64));
    }

    #[test]
    fn test_element_dtype() {
        assert_eq!(element_dtype(&DType::Complex32), DType::F32);
        assert_eq!(element_dtype(&DType::Complex64), DType::F64);
        assert_eq!(element_dtype(&DType::F32), DType::F32);
    }

    #[test]
    fn test_complex_add_decomposition() {
        // (1+2i) + (3+4i) = (4+6i)
        let a_re = const_f32(1.0);
        let a_im = const_f32(2.0);
        let b_re = const_f32(3.0);
        let b_im = const_f32(4.0);

        let (result_re, result_im) = lower_complex_add(a_re, a_im, b_re, b_im);

        // Check structure (actual values would be computed at runtime)
        match result_re {
            AstNode::Add(_, _) => {}
            _ => panic!("Expected Add node for real part"),
        }
        match result_im {
            AstNode::Add(_, _) => {}
            _ => panic!("Expected Add node for imag part"),
        }
    }

    #[test]
    fn test_complex_mul_decomposition() {
        // (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
        let a_re = const_f32(1.0);
        let a_im = const_f32(2.0);
        let b_re = const_f32(3.0);
        let b_im = const_f32(4.0);

        let (result_re, result_im) = lower_complex_mul(a_re, a_im, b_re, b_im);

        // Real part: ac - bd (Add with Neg, since Sub is implemented as a + (-b))
        match result_re {
            AstNode::Add(_, _) => {}
            _ => panic!("Expected Add node for real part (Sub is Add + Neg)"),
        }
        // Imag part: ad + bc (Add node)
        match result_im {
            AstNode::Add(_, _) => {}
            _ => panic!("Expected Add node for imag part"),
        }
    }

    #[test]
    fn test_store_complex() {
        let stmts = store_complex("output", const_int(5), const_f32(1.0), const_f32(2.0));
        assert_eq!(stmts.len(), 2);

        // First statement stores real part
        match &stmts[0] {
            AstNode::Store { .. } => {}
            _ => panic!("Expected Store node"),
        }

        // Second statement stores imag part
        match &stmts[1] {
            AstNode::Store { .. } => {}
            _ => panic!("Expected Store node"),
        }
    }
}
