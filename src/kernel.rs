use crate::variable::Variable;

pub trait Kernel {
    fn exec(&self, args: &[&dyn Variable]);
}
