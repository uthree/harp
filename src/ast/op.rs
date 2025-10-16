// Operator overloading for AstNode

use crate::ast::{helper::*, AstNode};
use std::ops::{Add, Div, Mul, Neg, Sub};

impl<T: Into<AstNode>> Add<T> for AstNode {
    type Output = AstNode;

    fn add(self, rhs: T) -> Self::Output {
        let rhs_node = rhs.into();
        add(self, rhs_node)
    }
}

impl<T: Into<AstNode>> Mul<T> for AstNode {
    type Output = AstNode;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs_node = rhs.into();
        mul(self, rhs_node)
    }
}

impl<T: Into<AstNode>> Div<T> for AstNode {
    type Output = AstNode;

    fn div(self, rhs: T) -> Self::Output {
        let rhs_node = rhs.into();
        // Division is implemented as multiplication by reciprocal
        mul(self, recip(rhs_node))
    }
}

impl Neg for AstNode {
    type Output = AstNode;

    fn neg(self) -> Self::Output {
        neg(self)
    }
}

impl<T: Into<AstNode>> Sub<T> for AstNode {
    type Output = AstNode;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs_node = rhs.into();
        // Subtraction is implemented as addition of negation
        add(self, neg(rhs_node))
    }
}
