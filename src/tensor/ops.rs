use super::{Dimension, Tensor, TensorType};
use std::ops::{Add, Div, Mul, Neg, Sub};

// Arithmetic operators for Tensor<T, D>

impl<T: TensorType, D: Dimension> Add for Tensor<T, D> {
    type Output = Self;

    fn add(self, _rhs: Self) -> Self::Output {
        todo!("Tensor addition will be implemented when graph building is ready")
    }
}

impl<T: TensorType, D: Dimension> Add for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn add(self, _rhs: Self) -> Self::Output {
        todo!("Tensor addition will be implemented when graph building is ready")
    }
}

impl<T: TensorType, D: Dimension> Sub for Tensor<T, D> {
    type Output = Self;

    fn sub(self, _rhs: Self) -> Self::Output {
        todo!("Tensor subtraction will be implemented when graph building is ready")
    }
}

impl<T: TensorType, D: Dimension> Sub for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn sub(self, _rhs: Self) -> Self::Output {
        todo!("Tensor subtraction will be implemented when graph building is ready")
    }
}

impl<T: TensorType, D: Dimension> Mul for Tensor<T, D> {
    type Output = Self;

    fn mul(self, _rhs: Self) -> Self::Output {
        todo!("Tensor multiplication will be implemented when graph building is ready")
    }
}

impl<T: TensorType, D: Dimension> Mul for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn mul(self, _rhs: Self) -> Self::Output {
        todo!("Tensor multiplication will be implemented when graph building is ready")
    }
}

impl<T: TensorType, D: Dimension> Div for Tensor<T, D> {
    type Output = Self;

    fn div(self, _rhs: Self) -> Self::Output {
        todo!("Tensor division will be implemented when graph building is ready")
    }
}

impl<T: TensorType, D: Dimension> Div for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn div(self, _rhs: Self) -> Self::Output {
        todo!("Tensor division will be implemented when graph building is ready")
    }
}

impl<T: TensorType, D: Dimension> Neg for Tensor<T, D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        todo!("Tensor negation will be implemented when graph building is ready")
    }
}

impl<T: TensorType, D: Dimension> Neg for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn neg(self) -> Self::Output {
        todo!("Tensor negation will be implemented when graph building is ready")
    }
}
