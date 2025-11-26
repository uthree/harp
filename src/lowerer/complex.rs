// Complex number lowering - decomposes complex operations into two F32 operations
//
// Complex numbers are represented as interleaved F32 values in a single buffer:
// [re0, im0, re1, im1, re2, im2, ...]
//
// This layout provides:
// - Better cache locality (real and imaginary parts are adjacent)
// - Simpler buffer management (single buffer per complex tensor)
// - Natural alignment for SIMD operations on complex pairs

use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, VarDecl, VarKind, helper::*};
use crate::graph::{GraphNode, ops::ElementwiseOp};

use super::Lowerer;

impl Lowerer {
    /// Check if a node has Complex dtype
    pub(super) fn is_complex_node(node: &GraphNode) -> bool {
        matches!(node.dtype, crate::graph::DType::Complex)
    }

    /// Create input parameter for complex type (single F32* buffer with interleaved layout)
    pub(super) fn create_complex_input_param(&self, index: usize) -> VarDecl {
        VarDecl {
            name: format!("input{}", index),
            dtype: AstDType::Ptr(Box::new(AstDType::F32)),
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
        }
    }

    /// Create output parameter for complex type (single F32* buffer with interleaved layout)
    pub(super) fn create_complex_output_param(&self) -> VarDecl {
        VarDecl {
            name: "output".to_string(),
            dtype: AstDType::Ptr(Box::new(AstDType::F32)),
            mutability: Mutability::Mutable,
            kind: VarKind::Normal,
        }
    }

    /// Lower complex elementwise kernel
    pub(super) fn lower_complex_elementwise_kernel(
        &mut self,
        node: &GraphNode,
        node_id: usize,
        op: &ElementwiseOp,
    ) -> Result<AstNode, String> {
        log::debug!("Lowering complex elementwise operation: {:?}", op);

        let shape = node.view.shape();
        let ndim = shape.len();

        // Generate parameters: complex input buffers, complex output buffers, shape vars
        let mut params = Vec::new();

        // Input buffers (complex -> single F32 buffer with interleaved layout)
        let mut input_idx = 0;
        for src in node.src.iter() {
            // 定数ノード（Const, ComplexConst）はバッファを持たないのでスキップ
            if matches!(
                src.op,
                crate::graph::GraphOp::Const(_) | crate::graph::GraphOp::ComplexConst { .. }
            ) {
                continue;
            }
            if Self::is_complex_node(src) {
                params.push(self.create_complex_input_param(input_idx));
            } else {
                params.push(self.create_input_param(input_idx, &src.dtype)?);
            }
            input_idx += 1;
        }

        // Output buffer (complex -> single F32 buffer with interleaved layout)
        if Self::is_complex_node(node) {
            params.push(self.create_complex_output_param());
        } else {
            params.push(self.create_output_param(&node.dtype)?);
        }

        params.extend(self.extract_shape_params(shape));

        // Generate loop body
        let (body_statements, body_scope) =
            self.generate_complex_elementwise_loops(node, op, ndim)?;

        Ok(self.create_kernel_function(node_id, params, body_statements, body_scope))
    }

    /// Generate loops for complex elementwise operation
    fn generate_complex_elementwise_loops(
        &mut self,
        node: &GraphNode,
        op: &ElementwiseOp,
        ndim: usize,
    ) -> Result<(Vec<AstNode>, Scope), String> {
        let mut scope = Scope::new();

        if ndim == 0 {
            // Scalar operation (no loops)
            let statements = self.generate_complex_elementwise_body(node, op, &[], &mut scope)?;
            return Ok((statements, scope));
        }

        // Generate nested loops
        let mut body_statements = self.generate_complex_elementwise_body(
            node,
            op,
            &(0..ndim).collect::<Vec<_>>(),
            &mut scope,
        )?;

        // Create loops in reverse order (inner to outer)
        for axis in (0..ndim).rev() {
            let loop_var = format!("ridx{}", axis);
            let shape_expr: AstNode = node.view.shape()[axis].clone().into();

            let loop_body = block(body_statements, scope.clone());
            scope = Scope::new();

            body_statements = vec![range(
                loop_var,
                const_int(0),
                const_int(1),
                shape_expr,
                loop_body,
            )];
        }

        Ok((body_statements, scope))
    }

    /// Generate body for complex elementwise operation
    fn generate_complex_elementwise_body(
        &mut self,
        node: &GraphNode,
        op: &ElementwiseOp,
        axes: &[usize],
        _scope: &mut Scope,
    ) -> Result<Vec<AstNode>, String> {
        // Load input values (real and imaginary parts from interleaved layout)
        let mut input_real_parts = Vec::new();
        let mut input_imag_parts = Vec::new();
        let mut input_idx = 0;

        for src in node.src.iter() {
            // Handle complex constants
            if let crate::graph::GraphOp::ComplexConst { re, im } = &src.op {
                input_real_parts.push(const_f32(*re));
                input_imag_parts.push(const_f32(*im));
                continue;
            }

            // Handle regular constants (real-valued)
            if let crate::graph::GraphOp::Const(lit) = &src.op {
                let const_node = AstNode::Const(lit.clone());
                input_real_parts.push(const_node.clone());
                input_imag_parts.push(const_f32(0.0)); // Imaginary part is 0
                continue;
            }

            let base_offset = self.compute_offset_from_view(src, axes);

            if Self::is_complex_node(src) {
                // Complex input: load from interleaved buffer
                // real part at offset * 2, imaginary part at offset * 2 + 1
                let input_ptr = var(format!("input{}", input_idx));
                let offset_re = base_offset.clone() * const_int(2);
                let offset_im = base_offset * const_int(2) + const_int(1);

                input_real_parts.push(load(input_ptr.clone(), offset_re, AstDType::F32));
                input_imag_parts.push(load(input_ptr, offset_im, AstDType::F32));
            } else {
                // Real input: imaginary part is 0
                let input_ptr = var(format!("input{}", input_idx));
                let src_dtype = self.graph_dtype_to_ast(&src.dtype)?;
                input_real_parts.push(load(input_ptr, base_offset, src_dtype));
                input_imag_parts.push(const_f32(0.0));
            }
            input_idx += 1;
        }

        // Apply complex operation
        let (result_re, result_im) =
            self.apply_complex_elementwise_op(op, &input_real_parts, &input_imag_parts)?;

        // Store results
        let base_output_offset = self.compute_offset_from_view(node, axes);

        if Self::is_complex_node(node) {
            // Complex output: store to interleaved buffer
            // real part at offset * 2, imaginary part at offset * 2 + 1
            let output_ptr = var("output");
            let offset_re = base_output_offset.clone() * const_int(2);
            let offset_im = base_output_offset * const_int(2) + const_int(1);

            let store_re = store(output_ptr.clone(), offset_re, result_re);
            let store_im = store(output_ptr, offset_im, result_im);
            Ok(vec![store_re, store_im])
        } else {
            // Real output (only possible for operations that produce real from complex, not implemented yet)
            Ok(vec![store(var("output"), base_output_offset, result_re)])
        }
    }

    /// Apply complex elementwise operation
    /// Returns (real_result, imag_result)
    fn apply_complex_elementwise_op(
        &self,
        op: &ElementwiseOp,
        real_parts: &[AstNode],
        imag_parts: &[AstNode],
    ) -> Result<(AstNode, AstNode), String> {
        match op {
            ElementwiseOp::Add => {
                // (a+bi) + (c+di) = (a+c) + (b+d)i
                if real_parts.len() != 2 {
                    return Err("Complex Add requires 2 operands".to_string());
                }
                let result_re = real_parts[0].clone() + real_parts[1].clone();
                let result_im = imag_parts[0].clone() + imag_parts[1].clone();
                Ok((result_re, result_im))
            }
            ElementwiseOp::Mul => {
                // (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
                if real_parts.len() != 2 {
                    return Err("Complex Mul requires 2 operands".to_string());
                }
                let a = real_parts[0].clone();
                let b = imag_parts[0].clone();
                let c = real_parts[1].clone();
                let d = imag_parts[1].clone();

                // ac - bd
                let result_re = a.clone() * c.clone() + const_f32(-1.0) * b.clone() * d.clone();
                // ad + bc
                let result_im = a * d + b * c;
                Ok((result_re, result_im))
            }
            ElementwiseOp::Neg => {
                // -(a+bi) = -a + (-b)i
                if real_parts.len() != 1 {
                    return Err("Complex Neg requires 1 operand".to_string());
                }
                let result_re = const_f32(-1.0) * real_parts[0].clone();
                let result_im = const_f32(-1.0) * imag_parts[0].clone();
                Ok((result_re, result_im))
            }
            ElementwiseOp::Recip => {
                // 1/(a+bi) = (a-bi)/(a²+b²)
                if real_parts.len() != 1 {
                    return Err("Complex Recip requires 1 operand".to_string());
                }
                let a = real_parts[0].clone();
                let b = imag_parts[0].clone();

                // denom = a² + b²
                let denom = a.clone() * a.clone() + b.clone() * b.clone();
                let denom_recip = recip(denom);

                // real = a / denom
                let result_re = a * denom_recip.clone();
                // imag = -b / denom
                let result_im = const_f32(-1.0) * b * denom_recip;
                Ok((result_re, result_im))
            }
            _ => Err(format!("Complex {:?} operation is not yet implemented", op)),
        }
    }
}
