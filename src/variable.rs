use crate::backend::Backend;
use std::rc::Rc;

pub trait VariableClone {
    fn clone_box(&self) -> Box<dyn Variable>;
}

impl<T> VariableClone for T
where
    T: 'static + Variable + Clone,
{
    fn clone_box(&self) -> Box<dyn Variable> {
        Box::new(self.clone())
    }
}

/// A trait representing a variable, which is essentially a buffer of data on a specific device (backend).
pub trait Variable: std::fmt::Debug + VariableClone {
    fn id(&self) -> usize;
    fn size(&self) -> usize;
    fn backend(&self) -> Rc<dyn Backend>;
}

impl Clone for Box<dyn Variable> {
    fn clone(&self) -> Box<dyn Variable> {
        self.clone_box()
    }
}
