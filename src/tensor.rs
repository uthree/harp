use crate::dtype::DType;
use crate::shapetracker::ShapeTracker;
use crate::uop::{Op, UOp};
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum TensorOp {
    Load(Option<Vec<f32>>),
    Add(Tensor, Tensor),
}

#[derive(Debug, Clone)]
struct Tensor_ {
    tracker: ShapeTracker,
    dtype: DType,
    op: TensorOp,
}

#[derive(Debug, Clone)]
pub struct Tensor(Rc<Tensor_>);

impl Tensor {
    pub fn new(shape: Vec<usize>, dtype: DType, op: TensorOp) -> Self {
        let tracker = ShapeTracker::new(shape);
        Self(Rc::new(Tensor_ {
            tracker,
            dtype,
            op,
        }))
    }

    pub fn from_data(shape: Vec<usize>, dtype: DType, data: Vec<f32>) -> Self {
        Self::new(shape, dtype, TensorOp::Load(Some(data)))
    }

    pub fn shape(&self) -> &Vec<usize> {
        self.0.tracker.shape()
    }

    pub fn to_uop_graph(&self) -> UOp {
        let mut cache: HashMap<*const Tensor_, UOp> = HashMap::new();
        self.to_uop_graph_recursive(&mut cache)
    }

    fn to_uop_graph_recursive(&self, cache: &mut HashMap<*const Tensor_, UOp>) -> UOp {
        let ptr = Rc::as_ptr(&self.0);
        if let Some(uop) = cache.get(&ptr) {
            return uop.clone();
        }

        let uop = match &self.0.op {
            TensorOp::Load(_) => {
                let buffer_uop = UOp::var(
                    &format!("data_{ptr:p}"),
                    DType::Pointer(Box::new(self.0.dtype.clone()), self.shape().iter().product()),
                );
                let index_uop = self.0.tracker.expr_indices(None);
                UOp::new(Op::Load, self.0.dtype.clone(), vec![buffer_uop, index_uop])
            }
            TensorOp::Add(lhs, rhs) => {
                let lhs_uop = lhs.to_uop_graph_recursive(cache);
                let rhs_uop = rhs.to_uop_graph_recursive(cache);
                lhs_uop + rhs_uop
            }
        };

        cache.insert(ptr, uop.clone());
        uop
    }
}

impl std::ops::Add for Tensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // TODO: Shape checking
        let new_shape = self.shape().clone();
        let new_dtype = self.0.dtype.clone(); // TODO: Dtype promotion
        Self::new(new_shape, new_dtype, TensorOp::Add(self, rhs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn test_tensor_creation() {
        let shape = vec![2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_data(shape.clone(), DType::F32, data.clone());

        assert_eq!(*t.shape(), shape);
        assert_eq!(t.0.tracker.views.len(), 1);
        assert_eq!(t.0.tracker.views[0].shape, shape);

        if let super::TensorOp::Load(Some(loaded_data)) = &t.0.op {
            assert_eq!(loaded_data, &data);
        } else {
            panic!("TensorOp should be Load");
        }
    }

    #[test]
    fn test_tensor_add_graph() {
        let shape = vec![2, 2];
        let t1 = Tensor::from_data(shape.clone(), DType::F32, vec![1.0, 2.0, 3.0, 4.0]);
        let t2 = Tensor::from_data(shape.clone(), DType::F32, vec![5.0, 6.0, 7.0, 8.0]);
        let t3 = t1.clone() + t2.clone();

        let uop = t3.to_uop_graph();

        // Check the top-level operation
        assert_eq!(uop.0.op, Op::Add);
        assert_eq!(uop.0.src.len(), 2);

        // Check the left-hand side (t1)
        let lhs = &uop.0.src[0];
        assert_eq!(lhs.0.op, Op::Load);
        assert_eq!(lhs.0.src.len(), 2); // buffer, index

        // Check the right-hand side (t2)
        let rhs = &uop.0.src[1];
        assert_eq!(rhs.0.op, Op::Load);
        assert_eq!(rhs.0.src.len(), 2); // buffer, index
    }
}