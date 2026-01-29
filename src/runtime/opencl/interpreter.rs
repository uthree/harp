//! OpenCL interpreter for evaluating UOp graphs.

use std::collections::HashMap;
use std::sync::Arc;

use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer as ClBuffer, CL_MEM_READ_ONLY};
use opencl3::types::{CL_BLOCKING, cl_uint};

use crate::device::{Buffer, BufferMap, DeviceError, Result};
use crate::dtype::{DType, ScalarValue};
use crate::ops::Ops;
use crate::schedule::{FusedKernel, FusedSource, FusionType, ScheduleItem, Scheduler};
use crate::shape::Shape;
use crate::uop::{UOp, UOpArg};

use super::buffer::OpenCLBuffer;
use super::codegen::FusedKernelCodeGen;
use super::device::OpenCLDevice;
use super::ops::{
    compare::{gen_cmp_kernel, gen_cmp_simple_kernel, gen_where_kernel, gen_where_simple_kernel},
    elementwise::{
        gen_binary_kernel, gen_binary_simple_kernel, gen_const_kernel, gen_unary_kernel,
    },
    movement::{gen_cast_kernel, gen_expand_kernel, gen_permute_kernel},
    reduce::{gen_full_reduce_kernel, gen_reduce_kernel, gen_reduce_last_axis_kernel},
};

/// OpenCL interpreter for evaluating UOp graphs.
pub struct OpenCLInterpreter<'a> {
    device: &'a OpenCLDevice,
    buffers: &'a mut BufferMap,
    cache: HashMap<usize, Arc<dyn Buffer>>,
}

impl<'a> OpenCLInterpreter<'a> {
    /// Creates a new OpenCL interpreter.
    pub fn new(device: &'a OpenCLDevice, buffers: &'a mut BufferMap) -> Self {
        OpenCLInterpreter {
            device,
            buffers,
            cache: HashMap::new(),
        }
    }

    /// Evaluates a UOp and returns the result buffer.
    pub fn eval(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
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

    /// Evaluates a UOp with kernel fusion optimization.
    /// This analyzes the graph and fuses compatible operations into single kernels.
    pub fn eval_with_fusion(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        // Check cache first
        let ptr = uop.ptr_id();
        if let Some(cached) = self.cache.get(&ptr) {
            return Ok(cached.clone());
        }

        // Create scheduler and build schedule
        let mut scheduler = Scheduler::new(uop);
        let items = scheduler.schedule(uop);

        // Execute schedule items
        for item in &items {
            self.eval_schedule_item(item, &scheduler)?;
        }

        // The final result should be in cache now
        self.cache
            .get(&ptr)
            .cloned()
            .ok_or_else(|| DeviceError::ExecutionFailed("Result not found after scheduling".into()))
    }

    /// Evaluates a single schedule item.
    fn eval_schedule_item(
        &mut self,
        item: &ScheduleItem,
        scheduler: &Scheduler,
    ) -> Result<Arc<dyn Buffer>> {
        let output_id = item.output.ptr_id();

        // Check if already evaluated
        if let Some(cached) = self.cache.get(&output_id) {
            return Ok(cached.clone());
        }

        let result = match item.fusion_type {
            FusionType::Single => {
                // No fusion, evaluate normally
                self.eval_inner(&item.output)?
            }
            FusionType::Elementwise => {
                // Execute fused elementwise kernel
                let kernel = scheduler.build_fused_kernel(item);
                self.eval_fused_elementwise(&kernel, &item.fused_ops)?
            }
            FusionType::Reduce => {
                // For now, fall back to non-fused execution for reduce
                // TODO: Implement fused reduce kernels
                self.eval_inner(&item.output)?
            }
        };

        // Cache the result
        self.cache.insert(output_id, result.clone());
        Ok(result)
    }

    /// Evaluates a fused elementwise kernel.
    fn eval_fused_elementwise(
        &mut self,
        kernel: &FusedKernel,
        fused_ops: &[UOp],
    ) -> Result<Arc<dyn Buffer>> {
        // Collect input buffers
        let mut input_buffers: Vec<Arc<dyn Buffer>> = Vec::new();

        // Find leaf inputs from fused ops
        for fused_op in &kernel.ops_chain {
            for source in &fused_op.sources {
                if let FusedSource::Input(idx) = source {
                    // Find the corresponding UOp and evaluate it
                    if *idx >= input_buffers.len() {
                        // We need to find the input UOp
                        let input_uop = self.find_input_uop(fused_ops, *idx)?;
                        let buf = self.eval(&input_uop)?;
                        while input_buffers.len() <= *idx {
                            input_buffers.push(buf.clone());
                        }
                        input_buffers[*idx] = buf;
                    }
                }
            }
        }

        // Generate and compile kernel
        let codegen = FusedKernelCodeGen::new(kernel);
        let (source, kernel_name) = codegen.generate();

        let numel = kernel.output_numel();
        let dtype = kernel.output_dtype;

        // Create output buffer
        let output = OpenCLBuffer::new(
            self.device.context(),
            self.device.queue().clone(),
            numel,
            dtype,
        )?;

        // Compile kernel
        let mut kernel_cache = self.device.kernel_cache().write().unwrap();
        let cl_kernel = kernel_cache.get_or_compile(
            self.device.context(),
            self.device.cl_device(),
            &source,
            &kernel_name,
        )?;

        // Execute kernel
        let n = numel as cl_uint;
        let global_work_size = [compute_work_sizes(numel).0];
        let local_work_size = [compute_work_sizes(numel).1];

        unsafe {
            let mut exec = ExecuteKernel::new(cl_kernel);

            // Set input buffer arguments
            for buf in &input_buffers {
                let cl_buf = buf
                    .as_any()
                    .downcast_ref::<OpenCLBuffer>()
                    .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;
                exec.set_arg(cl_buf.cl_buffer());
            }

            // Set output buffer and n
            exec.set_arg(output.cl_buffer())
                .set_arg(&n)
                .set_global_work_sizes(&global_work_size)
                .set_local_work_sizes(&local_work_size)
                .enqueue_nd_range(self.device.queue().as_ref())
                .map_err(|e| {
                    DeviceError::ExecutionFailed(format!("Fused kernel execution failed: {:?}", e))
                })?;
        }

        Ok(Arc::new(output))
    }

    /// Finds the input UOp at the given index by traversing the fused ops.
    fn find_input_uop(&self, fused_ops: &[UOp], input_idx: usize) -> Result<UOp> {
        // Collect all leaf inputs (Load and Const operations)
        let mut inputs: Vec<UOp> = Vec::new();
        let mut visited = std::collections::HashSet::new();

        for op in fused_ops {
            self.collect_leaf_inputs(op, &mut inputs, &mut visited);
        }

        // Filter to only Load operations (real inputs)
        let load_inputs: Vec<_> = inputs
            .into_iter()
            .filter(|u| matches!(u.op(), Ops::Load))
            .collect();

        load_inputs.get(input_idx).cloned().ok_or_else(|| {
            DeviceError::ExecutionFailed(format!("Input {} not found in fused ops", input_idx))
        })
    }

    fn collect_leaf_inputs(
        &self,
        uop: &UOp,
        inputs: &mut Vec<UOp>,
        visited: &mut std::collections::HashSet<usize>,
    ) {
        let id = uop.ptr_id();
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        match uop.op() {
            Ops::Load => {
                inputs.push(uop.clone());
            }
            Ops::Const => {
                // Constants are inlined, don't add to inputs
            }
            _ => {
                for src in uop.src() {
                    self.collect_leaf_inputs(src, inputs, visited);
                }
            }
        }
    }

    fn eval_inner(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        match uop.op() {
            Ops::Const => self.eval_const(uop),
            Ops::Load => self.eval_load(uop),
            Ops::Neg => self.eval_unary(uop, "neg"),
            Ops::Exp => self.eval_unary(uop, "exp"),
            Ops::Log => self.eval_unary(uop, "log"),
            Ops::Sqrt => self.eval_unary(uop, "sqrt"),
            Ops::Sin => self.eval_unary(uop, "sin"),
            Ops::Cos => self.eval_unary(uop, "cos"),
            Ops::Recip => self.eval_unary(uop, "recip"),
            Ops::Add => self.eval_binary(uop, "add"),
            Ops::Sub => self.eval_binary(uop, "sub"),
            Ops::Mul => self.eval_binary(uop, "mul"),
            Ops::Div => self.eval_binary(uop, "div"),
            Ops::Max => self.eval_binary(uop, "max"),
            Ops::CmpLt => self.eval_cmp(uop, "cmplt"),
            Ops::CmpEq => self.eval_cmp(uop, "cmpeq"),
            Ops::Where => self.eval_where(uop),
            Ops::Sum => self.eval_reduce(uop, "sum"),
            Ops::ReduceMax => self.eval_reduce(uop, "max"),
            Ops::Reshape => self.eval(&uop.src()[0]),
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

        // Create output buffer
        let mut output = OpenCLBuffer::new(
            self.device.context(),
            self.device.queue().clone(),
            numel,
            dtype,
        )?;

        // Generate and execute const kernel
        let (source, kernel_name) = gen_const_kernel(dtype);

        let mut kernel_cache = self.device.kernel_cache().write().unwrap();
        let kernel = kernel_cache.get_or_compile(
            self.device.context(),
            self.device.cl_device(),
            &source,
            &kernel_name,
        )?;

        self.execute_const_kernel(kernel, &mut output, numel, dtype, value)?;

        Ok(Arc::new(output))
    }

    fn execute_const_kernel(
        &self,
        kernel: &Kernel,
        output: &mut OpenCLBuffer,
        numel: usize,
        dtype: DType,
        value: ScalarValue,
    ) -> Result<()> {
        let n = numel as cl_uint;
        let global_work_size = [compute_work_sizes(numel).0];
        let local_work_size = [compute_work_sizes(numel).1];

        // Set kernel arguments based on dtype
        match dtype {
            DType::Float32 => {
                let v = value.to_f64() as f32;
                unsafe {
                    ExecuteKernel::new(kernel)
                        .set_arg(output.cl_buffer())
                        .set_arg(&v)
                        .set_arg(&n)
                        .set_global_work_sizes(&global_work_size)
                        .set_local_work_sizes(&local_work_size)
                        .enqueue_nd_range(self.device.queue().as_ref())
                        .map_err(|e| {
                            DeviceError::ExecutionFailed(format!(
                                "Kernel execution failed: {:?}",
                                e
                            ))
                        })?;
                }
            }
            DType::Float64 => {
                let v = value.to_f64();
                unsafe {
                    ExecuteKernel::new(kernel)
                        .set_arg(output.cl_buffer())
                        .set_arg(&v)
                        .set_arg(&n)
                        .set_global_work_sizes(&global_work_size)
                        .set_local_work_sizes(&local_work_size)
                        .enqueue_nd_range(self.device.queue().as_ref())
                        .map_err(|e| {
                            DeviceError::ExecutionFailed(format!(
                                "Kernel execution failed: {:?}",
                                e
                            ))
                        })?;
                }
            }
            DType::Int32 => {
                let v = value.to_f64() as i32;
                unsafe {
                    ExecuteKernel::new(kernel)
                        .set_arg(output.cl_buffer())
                        .set_arg(&v)
                        .set_arg(&n)
                        .set_global_work_sizes(&global_work_size)
                        .set_local_work_sizes(&local_work_size)
                        .enqueue_nd_range(self.device.queue().as_ref())
                        .map_err(|e| {
                            DeviceError::ExecutionFailed(format!(
                                "Kernel execution failed: {:?}",
                                e
                            ))
                        })?;
                }
            }
            DType::Int64 => {
                let v = value.to_f64() as i64;
                unsafe {
                    ExecuteKernel::new(kernel)
                        .set_arg(output.cl_buffer())
                        .set_arg(&v)
                        .set_arg(&n)
                        .set_global_work_sizes(&global_work_size)
                        .set_local_work_sizes(&local_work_size)
                        .enqueue_nd_range(self.device.queue().as_ref())
                        .map_err(|e| {
                            DeviceError::ExecutionFailed(format!(
                                "Kernel execution failed: {:?}",
                                e
                            ))
                        })?;
                }
            }
            DType::Bool => {
                let v = if value.to_f64() != 0.0 { 1u8 } else { 0u8 };
                unsafe {
                    ExecuteKernel::new(kernel)
                        .set_arg(output.cl_buffer())
                        .set_arg(&v)
                        .set_arg(&n)
                        .set_global_work_sizes(&global_work_size)
                        .set_local_work_sizes(&local_work_size)
                        .enqueue_nd_range(self.device.queue().as_ref())
                        .map_err(|e| {
                            DeviceError::ExecutionFailed(format!(
                                "Kernel execution failed: {:?}",
                                e
                            ))
                        })?;
                }
            }
        }

        Ok(())
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

    fn eval_unary(&mut self, uop: &UOp, op: &str) -> Result<Arc<dyn Buffer>> {
        let src = self.eval(&uop.src()[0])?;
        let numel = uop.numel();
        let dtype = uop.dtype();

        let output = OpenCLBuffer::new(
            self.device.context(),
            self.device.queue().clone(),
            numel,
            dtype,
        )?;

        let (source, kernel_name) = gen_unary_kernel(op, dtype);

        let mut kernel_cache = self.device.kernel_cache().write().unwrap();
        let kernel = kernel_cache.get_or_compile(
            self.device.context(),
            self.device.cl_device(),
            &source,
            &kernel_name,
        )?;

        let src_buf = src
            .as_any()
            .downcast_ref::<OpenCLBuffer>()
            .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;

        let n = numel as cl_uint;
        let global_work_size = [compute_work_sizes(numel).0];
        let local_work_size = [compute_work_sizes(numel).1];

        unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(src_buf.cl_buffer())
                .set_arg(output.cl_buffer())
                .set_arg(&n)
                .set_global_work_sizes(&global_work_size)
                .set_local_work_sizes(&local_work_size)
                .enqueue_nd_range(self.device.queue().as_ref())
                .map_err(|e| {
                    DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                })?;
        }

        Ok(Arc::new(output))
    }

    fn eval_binary(&mut self, uop: &UOp, op: &str) -> Result<Arc<dyn Buffer>> {
        let lhs = self.eval(&uop.src()[0])?;
        let rhs = self.eval(&uop.src()[1])?;

        let out_shape = uop.shape();
        let numel = out_shape.numel();
        let dtype = uop.dtype();

        let lhs_shape = uop.src()[0].shape();
        let rhs_shape = uop.src()[1].shape();

        let output = OpenCLBuffer::new(
            self.device.context(),
            self.device.queue().clone(),
            numel,
            dtype,
        )?;

        let lhs_buf = lhs
            .as_any()
            .downcast_ref::<OpenCLBuffer>()
            .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;
        let rhs_buf = rhs
            .as_any()
            .downcast_ref::<OpenCLBuffer>()
            .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;

        // Check if we can use simple kernel (same shape, no broadcast)
        let can_use_simple = lhs_shape.dims() == rhs_shape.dims();

        if can_use_simple {
            let (source, kernel_name) = gen_binary_simple_kernel(op, dtype);

            let mut kernel_cache = self.device.kernel_cache().write().unwrap();
            let kernel = kernel_cache.get_or_compile(
                self.device.context(),
                self.device.cl_device(),
                &source,
                &kernel_name,
            )?;

            let n = numel as cl_uint;
            let global_work_size = [compute_work_sizes(numel).0];
            let local_work_size = [compute_work_sizes(numel).1];

            unsafe {
                ExecuteKernel::new(kernel)
                    .set_arg(lhs_buf.cl_buffer())
                    .set_arg(rhs_buf.cl_buffer())
                    .set_arg(output.cl_buffer())
                    .set_arg(&n)
                    .set_global_work_sizes(&global_work_size)
                    .set_local_work_sizes(&local_work_size)
                    .enqueue_nd_range(self.device.queue().as_ref())
                    .map_err(|e| {
                        DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                    })?;
            }
        } else {
            // Use broadcast kernel
            let rank = out_shape.rank();
            let (source, kernel_name) = gen_binary_kernel(op, dtype, rank);

            let mut kernel_cache = self.device.kernel_cache().write().unwrap();
            let kernel = kernel_cache.get_or_compile(
                self.device.context(),
                self.device.cl_device(),
                &source,
                &kernel_name,
            )?;

            // Pad shapes to same rank
            let (out_shape_vec, lhs_shape_vec, rhs_shape_vec) =
                broadcast_shapes(out_shape, lhs_shape, rhs_shape);

            // Create shape buffers
            let out_shape_buf = self.create_uint_buffer(&out_shape_vec)?;
            let lhs_shape_buf = self.create_uint_buffer(&lhs_shape_vec)?;
            let rhs_shape_buf = self.create_uint_buffer(&rhs_shape_vec)?;

            let n = numel as cl_uint;
            let rank_val = rank as cl_uint;
            let global_work_size = [compute_work_sizes(numel).0];
            let local_work_size = [compute_work_sizes(numel).1];

            unsafe {
                ExecuteKernel::new(kernel)
                    .set_arg(lhs_buf.cl_buffer())
                    .set_arg(rhs_buf.cl_buffer())
                    .set_arg(output.cl_buffer())
                    .set_arg(&n)
                    .set_arg(&out_shape_buf)
                    .set_arg(&lhs_shape_buf)
                    .set_arg(&rhs_shape_buf)
                    .set_arg(&rank_val)
                    .set_global_work_sizes(&global_work_size)
                    .set_local_work_sizes(&local_work_size)
                    .enqueue_nd_range(self.device.queue().as_ref())
                    .map_err(|e| {
                        DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                    })?;
            }
        }

        Ok(Arc::new(output))
    }

    fn eval_cmp(&mut self, uop: &UOp, op: &str) -> Result<Arc<dyn Buffer>> {
        let lhs = self.eval(&uop.src()[0])?;
        let rhs = self.eval(&uop.src()[1])?;

        let out_shape = uop.shape();
        let numel = out_shape.numel();
        let input_dtype = uop.src()[0].dtype();

        let lhs_shape = uop.src()[0].shape();
        let rhs_shape = uop.src()[1].shape();

        let output = OpenCLBuffer::new(
            self.device.context(),
            self.device.queue().clone(),
            numel,
            DType::Bool,
        )?;

        let lhs_buf = lhs
            .as_any()
            .downcast_ref::<OpenCLBuffer>()
            .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;
        let rhs_buf = rhs
            .as_any()
            .downcast_ref::<OpenCLBuffer>()
            .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;

        let can_use_simple = lhs_shape.dims() == rhs_shape.dims();

        if can_use_simple {
            let (source, kernel_name) = gen_cmp_simple_kernel(op, input_dtype);

            let mut kernel_cache = self.device.kernel_cache().write().unwrap();
            let kernel = kernel_cache.get_or_compile(
                self.device.context(),
                self.device.cl_device(),
                &source,
                &kernel_name,
            )?;

            let n = numel as cl_uint;
            let global_work_size = [compute_work_sizes(numel).0];
            let local_work_size = [compute_work_sizes(numel).1];

            unsafe {
                ExecuteKernel::new(kernel)
                    .set_arg(lhs_buf.cl_buffer())
                    .set_arg(rhs_buf.cl_buffer())
                    .set_arg(output.cl_buffer())
                    .set_arg(&n)
                    .set_global_work_sizes(&global_work_size)
                    .set_local_work_sizes(&local_work_size)
                    .enqueue_nd_range(self.device.queue().as_ref())
                    .map_err(|e| {
                        DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                    })?;
            }
        } else {
            let rank = out_shape.rank();
            let (source, kernel_name) = gen_cmp_kernel(op, input_dtype, rank);

            let mut kernel_cache = self.device.kernel_cache().write().unwrap();
            let kernel = kernel_cache.get_or_compile(
                self.device.context(),
                self.device.cl_device(),
                &source,
                &kernel_name,
            )?;

            let (out_shape_vec, lhs_shape_vec, rhs_shape_vec) =
                broadcast_shapes(out_shape, lhs_shape, rhs_shape);

            let out_shape_buf = self.create_uint_buffer(&out_shape_vec)?;
            let lhs_shape_buf = self.create_uint_buffer(&lhs_shape_vec)?;
            let rhs_shape_buf = self.create_uint_buffer(&rhs_shape_vec)?;

            let n = numel as cl_uint;
            let rank_val = rank as cl_uint;
            let global_work_size = [compute_work_sizes(numel).0];
            let local_work_size = [compute_work_sizes(numel).1];

            unsafe {
                ExecuteKernel::new(kernel)
                    .set_arg(lhs_buf.cl_buffer())
                    .set_arg(rhs_buf.cl_buffer())
                    .set_arg(output.cl_buffer())
                    .set_arg(&n)
                    .set_arg(&out_shape_buf)
                    .set_arg(&lhs_shape_buf)
                    .set_arg(&rhs_shape_buf)
                    .set_arg(&rank_val)
                    .set_global_work_sizes(&global_work_size)
                    .set_local_work_sizes(&local_work_size)
                    .enqueue_nd_range(self.device.queue().as_ref())
                    .map_err(|e| {
                        DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                    })?;
            }
        }

        Ok(Arc::new(output))
    }

    fn eval_where(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        let cond = self.eval(&uop.src()[0])?;
        let x = self.eval(&uop.src()[1])?;
        let y = self.eval(&uop.src()[2])?;

        let out_shape = uop.shape();
        let numel = out_shape.numel();
        let dtype = uop.dtype();

        let cond_shape = uop.src()[0].shape();
        let x_shape = uop.src()[1].shape();
        let y_shape = uop.src()[2].shape();

        let output = OpenCLBuffer::new(
            self.device.context(),
            self.device.queue().clone(),
            numel,
            dtype,
        )?;

        let cond_buf = cond
            .as_any()
            .downcast_ref::<OpenCLBuffer>()
            .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;
        let x_buf = x
            .as_any()
            .downcast_ref::<OpenCLBuffer>()
            .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;
        let y_buf = y
            .as_any()
            .downcast_ref::<OpenCLBuffer>()
            .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;

        let can_use_simple =
            cond_shape.dims() == x_shape.dims() && x_shape.dims() == y_shape.dims();

        if can_use_simple {
            let (source, kernel_name) = gen_where_simple_kernel(dtype);

            let mut kernel_cache = self.device.kernel_cache().write().unwrap();
            let kernel = kernel_cache.get_or_compile(
                self.device.context(),
                self.device.cl_device(),
                &source,
                &kernel_name,
            )?;

            let n = numel as cl_uint;
            let global_work_size = [compute_work_sizes(numel).0];
            let local_work_size = [compute_work_sizes(numel).1];

            unsafe {
                ExecuteKernel::new(kernel)
                    .set_arg(cond_buf.cl_buffer())
                    .set_arg(x_buf.cl_buffer())
                    .set_arg(y_buf.cl_buffer())
                    .set_arg(output.cl_buffer())
                    .set_arg(&n)
                    .set_global_work_sizes(&global_work_size)
                    .set_local_work_sizes(&local_work_size)
                    .enqueue_nd_range(self.device.queue().as_ref())
                    .map_err(|e| {
                        DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                    })?;
            }
        } else {
            let rank = out_shape.rank();
            let (source, kernel_name) = gen_where_kernel(dtype, rank);

            let mut kernel_cache = self.device.kernel_cache().write().unwrap();
            let kernel = kernel_cache.get_or_compile(
                self.device.context(),
                self.device.cl_device(),
                &source,
                &kernel_name,
            )?;

            // Pad all shapes to same rank
            let out_shape_vec: Vec<u32> = out_shape.dims().iter().map(|&x| x as u32).collect();
            let cond_shape_vec = pad_shape_to_rank(cond_shape, rank);
            let x_shape_vec = pad_shape_to_rank(x_shape, rank);
            let y_shape_vec = pad_shape_to_rank(y_shape, rank);

            let out_shape_buf = self.create_uint_buffer(&out_shape_vec)?;
            let cond_shape_buf = self.create_uint_buffer(&cond_shape_vec)?;
            let x_shape_buf = self.create_uint_buffer(&x_shape_vec)?;
            let y_shape_buf = self.create_uint_buffer(&y_shape_vec)?;

            let n = numel as cl_uint;
            let rank_val = rank as cl_uint;
            let global_work_size = [compute_work_sizes(numel).0];
            let local_work_size = [compute_work_sizes(numel).1];

            unsafe {
                ExecuteKernel::new(kernel)
                    .set_arg(cond_buf.cl_buffer())
                    .set_arg(x_buf.cl_buffer())
                    .set_arg(y_buf.cl_buffer())
                    .set_arg(output.cl_buffer())
                    .set_arg(&n)
                    .set_arg(&out_shape_buf)
                    .set_arg(&cond_shape_buf)
                    .set_arg(&x_shape_buf)
                    .set_arg(&y_shape_buf)
                    .set_arg(&rank_val)
                    .set_global_work_sizes(&global_work_size)
                    .set_local_work_sizes(&local_work_size)
                    .enqueue_nd_range(self.device.queue().as_ref())
                    .map_err(|e| {
                        DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                    })?;
            }
        }

        Ok(Arc::new(output))
    }

    fn eval_reduce(&mut self, uop: &UOp, op: &str) -> Result<Arc<dyn Buffer>> {
        let src = self.eval(&uop.src()[0])?;
        let src_shape = uop.src()[0].shape();

        let axes = match uop.arg() {
            Some(UOpArg::Axes(a)) => a.clone(),
            _ => (0..src_shape.rank()).collect(),
        };

        let out_shape = uop.shape();
        let out_numel = out_shape.numel();
        let dtype = uop.dtype();

        let mut output = OpenCLBuffer::new(
            self.device.context(),
            self.device.queue().clone(),
            out_numel,
            dtype,
        )?;

        let src_buf = src
            .as_any()
            .downcast_ref::<OpenCLBuffer>()
            .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;

        // Check for special cases
        let is_full_reduce = axes.len() == src_shape.rank();
        let is_last_axis_only = axes.len() == 1 && axes[0] == src_shape.rank() - 1;

        if is_full_reduce && out_numel == 1 {
            // Full reduction to scalar
            self.eval_full_reduce(src_buf, &mut output, src_shape.numel(), dtype, op)?;
        } else if is_last_axis_only {
            // Reduce along last axis - common and efficient
            let reduce_size = src_shape.dim(src_shape.rank() - 1);
            self.eval_reduce_last_axis(src_buf, &mut output, out_numel, reduce_size, dtype, op)?;
        } else {
            // General case
            self.eval_reduce_general(src_buf, &mut output, src_shape, out_shape, &axes, dtype, op)?;
        }

        Ok(Arc::new(output))
    }

    fn eval_full_reduce(
        &mut self,
        src: &OpenCLBuffer,
        output: &mut OpenCLBuffer,
        numel: usize,
        dtype: DType,
        op: &str,
    ) -> Result<()> {
        let (source, kernel_name) = gen_full_reduce_kernel(op, dtype);

        let mut kernel_cache = self.device.kernel_cache().write().unwrap();
        let kernel = kernel_cache.get_or_compile(
            self.device.context(),
            self.device.cl_device(),
            &source,
            &kernel_name,
        )?;

        // Use a single work-group for simplicity
        // Local size MUST be a power of 2 for the parallel reduction to work correctly
        let local_size = next_power_of_two(numel.clamp(1, 64));
        let n = numel as cl_uint;

        unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(src.cl_buffer())
                .set_arg(output.cl_buffer())
                .set_arg_local_buffer(local_size * dtype.size_bytes())
                .set_arg(&n)
                .set_global_work_sizes(&[local_size])
                .set_local_work_sizes(&[local_size])
                .enqueue_nd_range(self.device.queue().as_ref())
                .map_err(|e| {
                    DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                })?;
        }

        Ok(())
    }

    fn eval_reduce_last_axis(
        &mut self,
        src: &OpenCLBuffer,
        output: &mut OpenCLBuffer,
        out_numel: usize,
        reduce_size: usize,
        dtype: DType,
        op: &str,
    ) -> Result<()> {
        let (source, kernel_name) = gen_reduce_last_axis_kernel(op, dtype);

        let mut kernel_cache = self.device.kernel_cache().write().unwrap();
        let kernel = kernel_cache.get_or_compile(
            self.device.context(),
            self.device.cl_device(),
            &source,
            &kernel_name,
        )?;

        let out_n = out_numel as cl_uint;
        let reduce_n = reduce_size as cl_uint;
        let global_work_size = [compute_work_sizes(out_numel).0];
        let local_work_size = [compute_work_sizes(out_numel).1];

        unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(src.cl_buffer())
                .set_arg(output.cl_buffer())
                .set_arg(&out_n)
                .set_arg(&reduce_n)
                .set_global_work_sizes(&global_work_size)
                .set_local_work_sizes(&local_work_size)
                .enqueue_nd_range(self.device.queue().as_ref())
                .map_err(|e| {
                    DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                })?;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_reduce_general(
        &mut self,
        src: &OpenCLBuffer,
        output: &mut OpenCLBuffer,
        src_shape: &Shape,
        out_shape: &Shape,
        axes: &[usize],
        dtype: DType,
        op: &str,
    ) -> Result<()> {
        let (source, kernel_name) = gen_reduce_kernel(op, dtype, axes, src_shape.rank());

        let mut kernel_cache = self.device.kernel_cache().write().unwrap();
        let kernel = kernel_cache.get_or_compile(
            self.device.context(),
            self.device.cl_device(),
            &source,
            &kernel_name,
        )?;

        let src_numel = src_shape.numel() as cl_uint;
        let out_numel = out_shape.numel() as cl_uint;
        let src_rank = src_shape.rank() as cl_uint;
        let out_rank = out_shape.rank() as cl_uint;

        let src_shape_vec: Vec<u32> = src_shape.dims().iter().map(|&x| x as u32).collect();
        let out_shape_vec: Vec<u32> = out_shape.dims().iter().map(|&x| x as u32).collect();

        // Create axis mask
        let mut axis_mask: Vec<u32> = vec![0; src_shape.rank()];
        for &axis in axes {
            if axis < src_shape.rank() {
                axis_mask[axis] = 1;
            }
        }

        let src_shape_buf = self.create_uint_buffer(&src_shape_vec)?;
        let out_shape_buf = self.create_uint_buffer(&out_shape_vec)?;
        let axis_mask_buf = self.create_uint_buffer(&axis_mask)?;

        let global_work_size = [compute_work_sizes(out_shape.numel()).0];
        let local_work_size = [compute_work_sizes(out_shape.numel()).1];

        unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(src.cl_buffer())
                .set_arg(output.cl_buffer())
                .set_arg(&src_numel)
                .set_arg(&out_numel)
                .set_arg(&src_shape_buf)
                .set_arg(&out_shape_buf)
                .set_arg(&axis_mask_buf)
                .set_arg(&src_rank)
                .set_arg(&out_rank)
                .set_global_work_sizes(&global_work_size)
                .set_local_work_sizes(&local_work_size)
                .enqueue_nd_range(self.device.queue().as_ref())
                .map_err(|e| {
                    DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                })?;
        }

        Ok(())
    }

    fn eval_expand(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        let src = self.eval(&uop.src()[0])?;
        let src_shape = uop.src()[0].shape();
        let out_shape = uop.shape();
        let numel = out_shape.numel();
        let dtype = uop.dtype();

        let output = OpenCLBuffer::new(
            self.device.context(),
            self.device.queue().clone(),
            numel,
            dtype,
        )?;

        let src_buf = src
            .as_any()
            .downcast_ref::<OpenCLBuffer>()
            .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;

        let rank = out_shape.rank();
        let (source, kernel_name) = gen_expand_kernel(dtype, rank);

        let mut kernel_cache = self.device.kernel_cache().write().unwrap();
        let kernel = kernel_cache.get_or_compile(
            self.device.context(),
            self.device.cl_device(),
            &source,
            &kernel_name,
        )?;

        let out_shape_vec: Vec<u32> = out_shape.dims().iter().map(|&x| x as u32).collect();
        let src_shape_vec = pad_shape_to_rank(src_shape, rank);

        let out_shape_buf = self.create_uint_buffer(&out_shape_vec)?;
        let src_shape_buf = self.create_uint_buffer(&src_shape_vec)?;

        let n = numel as cl_uint;
        let rank_val = rank as cl_uint;
        let global_work_size = [compute_work_sizes(numel).0];
        let local_work_size = [compute_work_sizes(numel).1];

        unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(src_buf.cl_buffer())
                .set_arg(output.cl_buffer())
                .set_arg(&n)
                .set_arg(&out_shape_buf)
                .set_arg(&src_shape_buf)
                .set_arg(&rank_val)
                .set_global_work_sizes(&global_work_size)
                .set_local_work_sizes(&local_work_size)
                .enqueue_nd_range(self.device.queue().as_ref())
                .map_err(|e| {
                    DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                })?;
        }

        Ok(Arc::new(output))
    }

    fn eval_permute(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        let src = self.eval(&uop.src()[0])?;
        let src_shape = uop.src()[0].shape();
        let out_shape = uop.shape();
        let numel = out_shape.numel();
        let dtype = uop.dtype();

        let axes = match uop.arg() {
            Some(UOpArg::Axes(a)) => a.clone(),
            _ => return Err(DeviceError::ExecutionFailed("Permute missing axes".into())),
        };

        let output = OpenCLBuffer::new(
            self.device.context(),
            self.device.queue().clone(),
            numel,
            dtype,
        )?;

        let src_buf = src
            .as_any()
            .downcast_ref::<OpenCLBuffer>()
            .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;

        let rank = out_shape.rank();
        let (source, kernel_name) = gen_permute_kernel(dtype, rank);

        let mut kernel_cache = self.device.kernel_cache().write().unwrap();
        let kernel = kernel_cache.get_or_compile(
            self.device.context(),
            self.device.cl_device(),
            &source,
            &kernel_name,
        )?;

        let out_shape_vec: Vec<u32> = out_shape.dims().iter().map(|&x| x as u32).collect();
        let src_shape_vec: Vec<u32> = src_shape.dims().iter().map(|&x| x as u32).collect();
        let axes_vec: Vec<u32> = axes.iter().map(|&x| x as u32).collect();

        let out_shape_buf = self.create_uint_buffer(&out_shape_vec)?;
        let src_shape_buf = self.create_uint_buffer(&src_shape_vec)?;
        let axes_buf = self.create_uint_buffer(&axes_vec)?;

        let n = numel as cl_uint;
        let rank_val = rank as cl_uint;
        let global_work_size = [compute_work_sizes(numel).0];
        let local_work_size = [compute_work_sizes(numel).1];

        unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(src_buf.cl_buffer())
                .set_arg(output.cl_buffer())
                .set_arg(&n)
                .set_arg(&out_shape_buf)
                .set_arg(&src_shape_buf)
                .set_arg(&axes_buf)
                .set_arg(&rank_val)
                .set_global_work_sizes(&global_work_size)
                .set_local_work_sizes(&local_work_size)
                .enqueue_nd_range(self.device.queue().as_ref())
                .map_err(|e| {
                    DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                })?;
        }

        Ok(Arc::new(output))
    }

    fn eval_cast(&mut self, uop: &UOp) -> Result<Arc<dyn Buffer>> {
        let src = self.eval(&uop.src()[0])?;
        let src_dtype = uop.src()[0].dtype();
        let dst_dtype = uop.dtype();
        let numel = uop.numel();

        let output = OpenCLBuffer::new(
            self.device.context(),
            self.device.queue().clone(),
            numel,
            dst_dtype,
        )?;

        let src_buf = src
            .as_any()
            .downcast_ref::<OpenCLBuffer>()
            .ok_or_else(|| DeviceError::BufferError("Expected OpenCL buffer".into()))?;

        let (source, kernel_name) = gen_cast_kernel(src_dtype, dst_dtype);

        let mut kernel_cache = self.device.kernel_cache().write().unwrap();
        let kernel = kernel_cache.get_or_compile(
            self.device.context(),
            self.device.cl_device(),
            &source,
            &kernel_name,
        )?;

        let n = numel as cl_uint;
        let global_work_size = [compute_work_sizes(numel).0];
        let local_work_size = [compute_work_sizes(numel).1];

        unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(src_buf.cl_buffer())
                .set_arg(output.cl_buffer())
                .set_arg(&n)
                .set_global_work_sizes(&global_work_size)
                .set_local_work_sizes(&local_work_size)
                .enqueue_nd_range(self.device.queue().as_ref())
                .map_err(|e| {
                    DeviceError::ExecutionFailed(format!("Kernel execution failed: {:?}", e))
                })?;
        }

        Ok(Arc::new(output))
    }

    /// Creates a temporary buffer containing u32 values.
    fn create_uint_buffer(&self, data: &[u32]) -> Result<ClBuffer<u32>> {
        let mut buffer = unsafe {
            ClBuffer::create(
                self.device.context(),
                CL_MEM_READ_ONLY,
                data.len(),
                std::ptr::null_mut(),
            )
            .map_err(|e| DeviceError::BufferError(format!("Failed to create buffer: {:?}", e)))?
        };

        unsafe {
            self.device
                .queue()
                .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, data, &[])
                .map_err(|e| {
                    DeviceError::BufferError(format!("Failed to write buffer: {:?}", e))
                })?;
        }

        Ok(buffer)
    }
}

/// Rounds up to the nearest multiple of workgroup size.
fn round_up_to_workgroup(n: usize, workgroup_size: usize) -> usize {
    n.div_ceil(workgroup_size) * workgroup_size
}

/// Returns the next power of 2 greater than or equal to n.
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v + 1
}

/// Determines appropriate work sizes for a given element count.
/// Returns (global_work_size, local_work_size)
fn compute_work_sizes(numel: usize) -> (usize, usize) {
    // For very small workloads, use a single work item per element
    if numel <= 64 {
        return (numel, numel);
    }

    // Use workgroup size that divides evenly when possible
    let preferred_local = 64; // Use smaller size for better compatibility
    let local_size = preferred_local.min(numel);
    let global_size = round_up_to_workgroup(numel, local_size);
    (global_size, local_size)
}

/// Pads a shape to a target rank by prepending 1s.
fn pad_shape_to_rank(shape: &Shape, target_rank: usize) -> Vec<u32> {
    let mut result = vec![1u32; target_rank];
    let offset = target_rank - shape.rank();
    for (i, &dim) in shape.dims().iter().enumerate() {
        result[offset + i] = dim as u32;
    }
    result
}

/// Broadcasts shapes for binary operations.
fn broadcast_shapes(
    out_shape: &Shape,
    lhs_shape: &Shape,
    rhs_shape: &Shape,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let rank = out_shape.rank();
    let out_vec: Vec<u32> = out_shape.dims().iter().map(|&x| x as u32).collect();
    let lhs_vec = pad_shape_to_rank(lhs_shape, rank);
    let rhs_vec = pad_shape_to_rank(rhs_shape, rank);
    (out_vec, lhs_vec, rhs_vec)
}
