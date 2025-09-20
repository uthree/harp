use crate::ast::{AstNode, DType, Function, Program, Scope, VariableDecl};
use crate::graph::shape::Expr;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

#[derive(Default)]
pub struct Lowerer {}
