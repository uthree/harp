use crate::prelude::*;
use crate::shape::symbolic::Expr;

#[derive(Debug, PartialEq)]
pub struct ShapeTracker {
    pub graph: GraphRef,
    pub map: Vec<Expr>,
    pub max: Vec<Expr>,
}

impl ShapeTracker {
    // generate full mapping
    pub fn full(graph: GraphRef, dims: Vec<Expr>) -> Self {
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
            graph: graph,
            max: maxs,
            map: maps,
        }
    }
}
