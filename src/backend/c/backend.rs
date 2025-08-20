use crate::backend::{
    c::{CCompiler, CRenderer},
    generic::GenericBackend,
};

pub type CBackend = GenericBackend<CCompiler, CRenderer>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::DType,
        backend::{Backend, Buffer, Kernel, c::CBuffer},
        graph::{Graph, TensorSignature, shape::expr::Expr as ShapeExpr},
    };

    #[test]
    fn test_c_backend_e2e() {
        let _ = env_logger::builder().is_test(true).try_init();
        // 1. Build a graph
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(1)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);
        let c = a + b;
        graph.outputs.push(c);
        graph.signature.outputs.push(TensorSignature {
            dtype: dtype.clone(),
            shape: shape.clone(),
        });

        // 2. Use CBackend to compile
        let mut backend = CBackend::new();
        let mut kernel = backend.compile(&graph);

        // 4. Prepare buffers and run the kernel
        let a_data = vec![5.0f32];
        let b_data = vec![3.0f32];
        let shape_usize = vec![1];

        let a_buffer = CBuffer::from_slice(&a_data, &shape_usize, dtype.clone());
        let b_buffer = CBuffer::from_slice(&b_data, &shape_usize, dtype.clone());
        let out_buffer = CBuffer::allocate(dtype.clone(), shape_usize);

        let buffers = vec![out_buffer, a_buffer, b_buffer];
        let result_buffers = kernel.call(buffers, &[]);

        // 5. Check the result
        let result_data = result_buffers[0].to_vec::<f32>();
        assert_eq!(result_data, vec![8.0f32]);
    }
}
