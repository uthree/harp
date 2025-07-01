use std::isize;

use crate::shape::symbolic::Expr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Axis {
    map: Expr,
    max: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeTracker {
    axes: Vec<Axis>,
}

impl ShapeTracker {
    // generate linear mapping
    pub fn from_shape(dims: Vec<usize>) -> Self {
        // calculate maps and strides
        let mut stride = 1;
        let mut maps = vec![];
        let mut maxs = vec![];
        for d in dims.iter().rev() {
            maps.push(Expr::Index * Expr::Int(stride));
            maxs.push(Expr::Int(*d as isize));
            stride = stride * (*d as isize);
        }
        let maps = maps.iter().rev().collect::<Vec<_>>();

        // format to Axis
        let mut axes = vec![];
        for (map, max) in maps.iter().zip(maxs.iter()) {
            axes.push(Axis {
                map: map.to_owned().clone().simplify(),
                max: max.clone().simplify(),
            })
        }
        ShapeTracker { axes }
    }

    pub fn ndim(&self) -> usize {
        self.axes.len()
    }
}
