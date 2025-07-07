use crate::shape::symbolic::Expr;
use std::fmt;

#[derive(Debug, PartialEq, Clone)]
pub struct ShapeTracker {
    pub map: Vec<Expr>,
    pub max: Vec<Expr>,
}

impl ShapeTracker {
    // generate full mapping
    pub fn full(dims: Vec<Expr>) -> Self {
        // calculate maps and strides
        let mut alu: Expr = 1.into();
        let mut maps = vec![];
        let mut maxs = vec![];
        for d in dims.iter().rev() {
            maps.push((Expr::Index * alu.clone()).simplify());
            maxs.push(d.clone().simplify());
            alu = alu * d.clone();
        }
        let maps = maps.iter().rev().map(|m| m.to_owned()).collect::<Vec<_>>();
        let maxs = maxs.iter().rev().map(|m| m.to_owned()).collect::<Vec<_>>();
        ShapeTracker {
            max: maxs,
            map: maps,
        }
    }
}

impl fmt::Display for ShapeTracker {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Shape[({}), ({})]",
            self.max.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", "),
            self.map.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", "),
        )
    }
}