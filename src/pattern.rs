use crate::ast::Ast;
use rustc_hash::FxHashMap;
use std::rc::Rc;

pub struct AstPat {
    pattern: Ast,
    rewriter: dyn Fn(Vec<Ast>) -> Ast,
}
/*
impl AstPat {
    fn scan(&self, tgt: &Ast, pat: &Ast, mut store: &FxHashMap<usize, Ast>) -> bool {}
    fn capture(&self, target: &Ast) -> Vec<Ast> {
        let mut captures = FxHashMap::default();
    }
    pub fn apply(&self, target: &Ast) -> Ast {}
}
*/

pub struct AstPatternMatcher {
    name: String,
    patterns: Vec<Rc<AstPat>>,
}
