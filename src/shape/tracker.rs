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
    // generate full mapping
    pub fn full(dims: Vec<usize>) -> Self {
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

    // insert dimension
    pub fn unsq(&mut self, dim: usize) {
        let axis = Axis {
            map: 0.into(),
            max: 1.into(),
        };
        self.axes.insert(dim, axis);
    }

    pub fn permute(&mut self, idxs: Vec<usize>) {
        let mut new_axes = vec![];
        for i in idxs.iter() {
            new_axes.push(self.axes[*i].clone())
        }
        self.axes = new_axes;
    }
}
