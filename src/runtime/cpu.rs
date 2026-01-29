//! CPU backend implementation using an interpreter.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use crate::device::{Buffer, BufferMap, Device, DeviceError, Result};
use crate::dtype::{DType, Scalar};
use crate::ops::Ops;
use crate::shape::Shape;
use crate::uop::{UOp, UOpArg};

/// CPU buffer implementation.
pub struct CpuBuffer {
    data: Vec<u8>,
    dtype: DType,
}

impl CpuBuffer {
    pub fn new(numel: usize, dtype: DType) -> Self {
        let size = numel * dtype.size_bytes();
        CpuBuffer {
            data: vec![0; size],
            dtype,
        }
    }

    pub fn from_data(data: Vec<u8>, dtype: DType) -> Self {
        CpuBuffer { data, dtype }
    }

    /// Returns raw data pointer for reading.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns raw data pointer for writing.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Gets a value at the given index.
    pub fn get<T: Scalar>(&self, idx: usize) -> T {
        let offset = idx * T::DTYPE.size_bytes();
        T::from_bytes(&self.data[offset..])
    }

    /// Sets a value at the given index.
    pub fn set<T: Scalar>(&mut self, idx: usize, value: T) {
        let offset = idx * T::DTYPE.size_bytes();
        let bytes = value.to_bytes();
        self.data[offset..offset + bytes.len()].copy_from_slice(&bytes);
    }
}

impl Buffer for CpuBuffer {
    fn size(&self) -> usize {
        self.data.len()
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn copy_from_host(&mut self, data: &[u8]) {
        self.data.copy_from_slice(data);
    }

    fn copy_to_host(&self) -> Vec<u8> {
        self.data.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// CPU device implementation.
pub struct CpuDevice;

impl CpuDevice {
    pub fn new() -> Self {
        CpuDevice
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for CpuDevice {
    fn name(&self) -> &str {
        "CPU"
    }

    fn alloc(&self, numel: usize, dtype: DType) -> Result<Box<dyn Buffer>> {
        Ok(Box::new(CpuBuffer::new(numel, dtype)))
    }

    fn realize(&self, uop: &UOp, buffers: &mut BufferMap) -> Result<Arc<dyn Buffer>> {
        let mut interpreter = CpuInterpreter::new(buffers);
        interpreter.eval(uop)
    }
}

/// CPU interpreter for evaluating UOp graphs.
struct CpuInterpreter<'a> {
    buffers: &'a mut BufferMap,
    cache: HashMap<usize, Arc<dyn Buffer>>,
}

impl<'a> CpuInterpreter<'a> {
    fn new(buffers: &'a mut BufferMap) -> Self {
        CpuInterpreter {
            buffers,
            cache: HashMap::new(),
        }
    }

    fn eval(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        // Check cache first
        let ptr = uop.ptr_id();
        if let Some(cached) = self.cache.get(&ptr) {
            return Ok(cached.clone());
        }

        let result = self.eval_inner(uop)?;

        // Cache result
        self.cache.insert(ptr, result.clone());

        Ok(result)
    }

    fn eval_inner(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        match uop.op() {
            Ops::Const => self.eval_const(uop),
            Ops::Load => self.eval_load(uop),
            Ops::Neg => self.eval_unary(uop, |x: f64| -x),
            Ops::Exp => self.eval_unary(uop, f64::exp),
            Ops::Log => self.eval_unary(uop, f64::ln),
            Ops::Sqrt => self.eval_unary(uop, f64::sqrt),
            Ops::Sin => self.eval_unary(uop, f64::sin),
            Ops::Cos => self.eval_unary(uop, f64::cos),
            Ops::Recip => self.eval_unary(uop, |x: f64| 1.0 / x),
            Ops::Add => self.eval_binary(uop, |a: f64, b: f64| a + b),
            Ops::Sub => self.eval_binary(uop, |a: f64, b: f64| a - b),
            Ops::Mul => self.eval_binary(uop, |a: f64, b: f64| a * b),
            Ops::Div => self.eval_binary(uop, |a: f64, b: f64| a / b),
            Ops::Max => self.eval_binary(uop, f64::max),
            Ops::CmpLt => self.eval_cmp(uop, |a: f64, b: f64| a < b),
            Ops::CmpEq => self.eval_cmp(uop, |a: f64, b: f64| (a - b).abs() < f64::EPSILON),
            Ops::Where => self.eval_where(uop),
            Ops::Sum => self.eval_reduce(uop, 0.0, |acc, x| acc + x),
            Ops::ReduceMax => self.eval_reduce(uop, f64::NEG_INFINITY, f64::max),
            Ops::Reshape => self.eval(uop.src().first().unwrap()),
            Ops::Expand => self.eval_expand(uop),
            Ops::Permute => self.eval_permute(uop),
            Ops::Cast => self.eval_cast(uop),
            _ => Err(DeviceError::UnsupportedOperation(format!("{:?}", uop.op()))),
        }
    }

    fn eval_const(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        let numel = uop.numel();
        let dtype = uop.dtype();
        let value = match uop.arg() {
            Some(UOpArg::Scalar(v)) => *v,
            _ => {
                return Err(DeviceError::ExecutionFailed(
                    "Const missing scalar arg".into(),
                ));
            }
        };

        let mut buffer = CpuBuffer::new(numel, dtype);

        match dtype {
            DType::Float32 => {
                let v = value.to_f64() as f32;
                for i in 0..numel {
                    buffer.set(i, v);
                }
            }
            DType::Float64 => {
                let v = value.to_f64();
                for i in 0..numel {
                    buffer.set(i, v);
                }
            }
            DType::Int32 => {
                let v = value.to_f64() as i32;
                for i in 0..numel {
                    buffer.set(i, v);
                }
            }
            DType::Int64 => {
                let v = value.to_f64() as i64;
                for i in 0..numel {
                    buffer.set(i, v);
                }
            }
            DType::Bool => {
                let v = value.to_f64() != 0.0;
                for i in 0..numel {
                    buffer.set(i, v);
                }
            }
        }

        Ok(Arc::new(buffer))
    }

    fn eval_load(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        let buffer_id = match uop.arg() {
            Some(UOpArg::BufferId(id)) => *id,
            _ => {
                return Err(DeviceError::ExecutionFailed(
                    "Load missing buffer id".into(),
                ));
            }
        };

        self.buffers
            .get(buffer_id)
            .ok_or_else(|| DeviceError::BufferError(format!("Buffer {} not found", buffer_id)))
    }

    fn eval_unary<F>(&mut self, uop: &UOp, f: F) -> Result<Arc<dyn Buffer>>
    where
        F: Fn(f64) -> f64,
    {
        let src = self.eval(&uop.src()[0])?;
        let numel = uop.numel();
        let dtype = uop.dtype();
        let mut result = CpuBuffer::new(numel, dtype);

        self.apply_unary(&src, &mut result, numel, dtype, f)?;

        Ok(Arc::new(result))
    }

    fn apply_unary<F>(
        &self,
        src: &Arc<dyn Buffer>,
        dst: &mut CpuBuffer,
        numel: usize,
        dtype: DType,
        f: F,
    ) -> Result<()>
    where
        F: Fn(f64) -> f64,
    {
        let src = src.as_any().downcast_ref::<CpuBuffer>().unwrap();

        match dtype {
            DType::Float32 => {
                for i in 0..numel {
                    let v: f32 = src.get(i);
                    dst.set(i, f(v as f64) as f32);
                }
            }
            DType::Float64 => {
                for i in 0..numel {
                    let v: f64 = src.get(i);
                    dst.set(i, f(v));
                }
            }
            _ => {
                for i in 0..numel {
                    let v: f64 = read_as_f64(src, i, dtype);
                    write_from_f64(dst, i, f(v), dtype);
                }
            }
        }

        Ok(())
    }

    fn eval_binary<F>(&mut self, uop: &UOp, f: F) -> Result<Arc<dyn Buffer>>
    where
        F: Fn(f64, f64) -> f64,
    {
        let lhs = self.eval(&uop.src()[0])?;
        let rhs = self.eval(&uop.src()[1])?;

        let out_shape = uop.shape();
        let numel = out_shape.numel();
        let dtype = uop.dtype();
        let mut result = CpuBuffer::new(numel, dtype);

        let lhs_shape = uop.src()[0].shape();
        let rhs_shape = uop.src()[1].shape();

        let lhs = lhs.as_any().downcast_ref::<CpuBuffer>().unwrap();
        let rhs = rhs.as_any().downcast_ref::<CpuBuffer>().unwrap();

        for i in 0..numel {
            let indices = out_shape.multi_index(i);
            let lhs_idx = broadcast_index(&indices, out_shape, lhs_shape);
            let rhs_idx = broadcast_index(&indices, out_shape, rhs_shape);

            let a = read_as_f64(lhs, lhs_idx, uop.src()[0].dtype());
            let b = read_as_f64(rhs, rhs_idx, uop.src()[1].dtype());
            write_from_f64(&mut result, i, f(a, b), dtype);
        }

        Ok(Arc::new(result))
    }

    fn eval_cmp<F>(&mut self, uop: &UOp, f: F) -> Result<Arc<dyn Buffer>>
    where
        F: Fn(f64, f64) -> bool,
    {
        let lhs = self.eval(&uop.src()[0])?;
        let rhs = self.eval(&uop.src()[1])?;

        let out_shape = uop.shape();
        let numel = out_shape.numel();
        let mut result = CpuBuffer::new(numel, DType::Bool);

        let lhs_shape = uop.src()[0].shape();
        let rhs_shape = uop.src()[1].shape();

        let lhs = lhs.as_any().downcast_ref::<CpuBuffer>().unwrap();
        let rhs = rhs.as_any().downcast_ref::<CpuBuffer>().unwrap();

        for i in 0..numel {
            let indices = out_shape.multi_index(i);
            let lhs_idx = broadcast_index(&indices, out_shape, lhs_shape);
            let rhs_idx = broadcast_index(&indices, out_shape, rhs_shape);

            let a = read_as_f64(lhs, lhs_idx, uop.src()[0].dtype());
            let b = read_as_f64(rhs, rhs_idx, uop.src()[1].dtype());
            result.set(i, f(a, b));
        }

        Ok(Arc::new(result))
    }

    fn eval_where(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        let cond = self.eval(&uop.src()[0])?;
        let x = self.eval(&uop.src()[1])?;
        let y = self.eval(&uop.src()[2])?;

        let out_shape = uop.shape();
        let numel = out_shape.numel();
        let dtype = uop.dtype();
        let mut result = CpuBuffer::new(numel, dtype);

        let cond_shape = uop.src()[0].shape();
        let x_shape = uop.src()[1].shape();
        let y_shape = uop.src()[2].shape();

        let cond = cond.as_any().downcast_ref::<CpuBuffer>().unwrap();
        let x = x.as_any().downcast_ref::<CpuBuffer>().unwrap();
        let y = y.as_any().downcast_ref::<CpuBuffer>().unwrap();

        for i in 0..numel {
            let indices = out_shape.multi_index(i);
            let cond_idx = broadcast_index(&indices, out_shape, cond_shape);
            let x_idx = broadcast_index(&indices, out_shape, x_shape);
            let y_idx = broadcast_index(&indices, out_shape, y_shape);

            let c: bool = cond.get(cond_idx);
            let v = if c {
                read_as_f64(x, x_idx, uop.src()[1].dtype())
            } else {
                read_as_f64(y, y_idx, uop.src()[2].dtype())
            };
            write_from_f64(&mut result, i, v, dtype);
        }

        Ok(Arc::new(result))
    }

    fn eval_reduce<F>(&mut self, uop: &UOp, init: f64, f: F) -> Result<Arc<dyn Buffer>>
    where
        F: Fn(f64, f64) -> f64,
    {
        let src = self.eval(&uop.src()[0])?;
        let src_shape = uop.src()[0].shape();

        let axes = match uop.arg() {
            Some(UOpArg::Axes(a)) => a.clone(),
            _ => (0..src_shape.rank()).collect(),
        };

        let out_shape = uop.shape();
        let out_numel = out_shape.numel();
        let dtype = uop.dtype();
        let mut result = CpuBuffer::new(out_numel, dtype);

        let src = src.as_any().downcast_ref::<CpuBuffer>().unwrap();

        // Initialize with init value
        for i in 0..out_numel {
            write_from_f64(&mut result, i, init, dtype);
        }

        // Determine if keepdims was used by comparing ranks
        let keepdims = out_shape.rank() == src_shape.rank();

        // Iterate over source elements and accumulate
        let src_numel = src_shape.numel();
        for src_i in 0..src_numel {
            let src_indices = src_shape.multi_index(src_i);

            // Compute output index
            let out_indices: Vec<usize> = if keepdims {
                // When keepdims=true, use 0 for reduced dimensions
                src_indices
                    .iter()
                    .enumerate()
                    .map(|(i, &idx)| if axes.contains(&i) { 0 } else { idx })
                    .collect()
            } else {
                // When keepdims=false, remove reduced dimensions
                src_indices
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &idx)| if axes.contains(&i) { None } else { Some(idx) })
                    .collect()
            };

            let out_i = if out_indices.is_empty() {
                0
            } else {
                out_shape.flat_index(&out_indices)
            };

            let src_val = read_as_f64(src, src_i, uop.src()[0].dtype());
            let cur_val = read_as_f64(&result, out_i, dtype);
            write_from_f64(&mut result, out_i, f(cur_val, src_val), dtype);
        }

        Ok(Arc::new(result))
    }

    fn eval_expand(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        let src = self.eval(&uop.src()[0])?;
        let src_shape = uop.src()[0].shape();
        let out_shape = uop.shape();
        let numel = out_shape.numel();
        let dtype = uop.dtype();
        let mut result = CpuBuffer::new(numel, dtype);

        let src = src.as_any().downcast_ref::<CpuBuffer>().unwrap();

        for i in 0..numel {
            let indices = out_shape.multi_index(i);
            let src_idx = broadcast_index(&indices, out_shape, src_shape);
            let v = read_as_f64(src, src_idx, dtype);
            write_from_f64(&mut result, i, v, dtype);
        }

        Ok(Arc::new(result))
    }

    fn eval_permute(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        let src = self.eval(&uop.src()[0])?;
        let src_shape = uop.src()[0].shape();
        let out_shape = uop.shape();
        let numel = out_shape.numel();
        let dtype = uop.dtype();
        let mut result = CpuBuffer::new(numel, dtype);

        let axes = match uop.arg() {
            Some(UOpArg::Axes(a)) => a.clone(),
            _ => return Err(DeviceError::ExecutionFailed("Permute missing axes".into())),
        };

        let src = src.as_any().downcast_ref::<CpuBuffer>().unwrap();

        for i in 0..numel {
            let out_indices = out_shape.multi_index(i);

            // Compute source indices by reversing the permutation
            let mut src_indices = vec![0; src_shape.rank()];
            for (out_dim, &src_dim) in axes.iter().enumerate() {
                src_indices[src_dim] = out_indices[out_dim];
            }

            let src_idx = src_shape.flat_index(&src_indices);
            let v = read_as_f64(src, src_idx, dtype);
            write_from_f64(&mut result, i, v, dtype);
        }

        Ok(Arc::new(result))
    }

    fn eval_cast(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        let src = self.eval(&uop.src()[0])?;
        let src_dtype = uop.src()[0].dtype();
        let dst_dtype = uop.dtype();
        let numel = uop.numel();
        let mut result = CpuBuffer::new(numel, dst_dtype);

        let src = src.as_any().downcast_ref::<CpuBuffer>().unwrap();

        for i in 0..numel {
            let v = read_as_f64(src, i, src_dtype);
            write_from_f64(&mut result, i, v, dst_dtype);
        }

        Ok(Arc::new(result))
    }
}

/// Computes the index in a source array given a broadcasted output index.
fn broadcast_index(out_indices: &[usize], out_shape: &Shape, src_shape: &Shape) -> usize {
    let rank_diff = out_shape.rank() - src_shape.rank();
    let mut src_indices = Vec::with_capacity(src_shape.rank());

    for i in 0..src_shape.rank() {
        let out_i = i + rank_diff;
        let src_dim = src_shape.dim(i);
        if src_dim == 1 {
            src_indices.push(0);
        } else {
            src_indices.push(out_indices[out_i]);
        }
    }

    src_shape.flat_index(&src_indices)
}

fn read_as_f64(buf: &CpuBuffer, idx: usize, dtype: DType) -> f64 {
    match dtype {
        DType::Bool => {
            let v: bool = buf.get(idx);
            if v { 1.0 } else { 0.0 }
        }
        DType::Int32 => {
            let v: i32 = buf.get(idx);
            v as f64
        }
        DType::Int64 => {
            let v: i64 = buf.get(idx);
            v as f64
        }
        DType::Float32 => {
            let v: f32 = buf.get(idx);
            v as f64
        }
        DType::Float64 => buf.get(idx),
    }
}

fn write_from_f64(buf: &mut CpuBuffer, idx: usize, v: f64, dtype: DType) {
    match dtype {
        DType::Bool => buf.set(idx, v != 0.0),
        DType::Int32 => buf.set(idx, v as i32),
        DType::Int64 => buf.set(idx, v as i64),
        DType::Float32 => buf.set(idx, v as f32),
        DType::Float64 => buf.set(idx, v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::ScalarValue;

    #[test]
    fn test_cpu_buffer() {
        let mut buf = CpuBuffer::new(4, DType::Float32);
        buf.set(0, 1.0f32);
        buf.set(1, 2.0f32);
        assert_eq!(buf.get::<f32>(0), 1.0);
        assert_eq!(buf.get::<f32>(1), 2.0);
    }

    #[test]
    fn test_const_eval() {
        let device = CpuDevice::new();
        let mut buffers = BufferMap::new();

        let uop = UOp::constant(ScalarValue::Float32(3.14), Shape::new([2, 2]));
        let result = device.realize(&uop, &mut buffers).unwrap();

        let buf = result.as_any().downcast_ref::<CpuBuffer>().unwrap();
        assert_eq!(buf.get::<f32>(0), 3.14f32);
        assert_eq!(buf.get::<f32>(3), 3.14f32);
    }

    #[test]
    fn test_add_eval() {
        let device = CpuDevice::new();
        let mut buffers = BufferMap::new();

        let a = UOp::constant(ScalarValue::Float32(1.0), Shape::new([2, 2]));
        let b = UOp::constant(ScalarValue::Float32(2.0), Shape::new([2, 2]));
        let c = a.add(&b);

        let result = device.realize(&c, &mut buffers).unwrap();
        let buf = result.as_any().downcast_ref::<CpuBuffer>().unwrap();
        assert_eq!(buf.get::<f32>(0), 3.0f32);
    }

    #[test]
    fn test_sum_eval() {
        let device = CpuDevice::new();
        let mut buffers = BufferMap::new();

        let a = UOp::constant(ScalarValue::Float32(1.0), Shape::new([2, 3]));
        let s = a.sum(None, false);

        let result = device.realize(&s, &mut buffers).unwrap();
        let buf = result.as_any().downcast_ref::<CpuBuffer>().unwrap();
        assert_eq!(buf.get::<f32>(0), 6.0f32);
    }
}
