use crate::prelude::*;
use crate::shape::symbolic::Expr;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ShapeTracker {
    graph: Graph,
    map: Vec<Expr>,
    max: Vec<Expr>,
}

impl ShapeTracker {
    // generate full mapping
    pub fn full(graph: Graph, dims: Vec<Expr>) -> Self {
        // calculate maps and strides
        let mut alu: Expr = 1.into();
        let mut maps = vec![];
        let mut maxs = vec![];
        for d in dims.iter().rev() {
            maps.push(Expr::Index * alu.clone());
            maxs.push(d.clone());
            alu = alu * d.clone();
        }
        let maps = maps.iter().rev().map(|m| m.to_owned()).collect::<Vec<_>>();
        ShapeTracker {
            graph: graph,
            max: maxs,
            map: maps,
        }
    }
}
