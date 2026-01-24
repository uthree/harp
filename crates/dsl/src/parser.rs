//! DSL Parser implementation using pest

use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;

use crate::ast::*;
use crate::errors::{DslError, DslResult};

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct DslParser;

/// Parse a DSL source string into a DslProgram
pub fn parse_program(source: &str) -> DslResult<DslProgram> {
    let pairs =
        DslParser::parse(Rule::program, source).map_err(|e| DslError::ParseError(e.to_string()))?;

    let mut graphs = Vec::new();

    for pair in pairs {
        match pair.as_rule() {
            Rule::program => {
                for inner in pair.into_inner() {
                    if let Rule::graph_def = inner.as_rule() {
                        graphs.push(parse_graph_def(inner)?);
                    }
                }
            }
            _ => {}
        }
    }

    Ok(DslProgram { graphs })
}

fn parse_graph_def(pair: Pair<Rule>) -> DslResult<GraphDef> {
    let mut inner = pair.into_inner();

    let name = inner.next().unwrap().as_str().to_string();

    // Parse params (may be empty)
    let mut params = Vec::new();
    let mut next_pair = inner.next().unwrap();

    if next_pair.as_rule() == Rule::param_list {
        params = parse_param_list(next_pair)?;
        next_pair = inner.next().unwrap();
    }

    // Parse return type
    let return_type = parse_type_spec(next_pair)?;

    // Parse body
    let body_pair = inner.next().unwrap();
    let (body, return_expr) = parse_body(body_pair)?;

    Ok(GraphDef {
        name,
        params,
        return_type,
        body,
        return_expr,
    })
}

fn parse_param_list(pair: Pair<Rule>) -> DslResult<Vec<ParamDecl>> {
    let mut params = Vec::new();
    for param_pair in pair.into_inner() {
        params.push(parse_param(param_pair)?);
    }
    Ok(params)
}

fn parse_param(pair: Pair<Rule>) -> DslResult<ParamDecl> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let type_spec = parse_type_spec(inner.next().unwrap())?;
    Ok(ParamDecl { name, type_spec })
}

fn parse_type_spec(pair: Pair<Rule>) -> DslResult<TypeSpec> {
    let mut inner = pair.into_inner();
    let dtype = parse_dtype(inner.next().unwrap())?;
    let shape = parse_shape(inner.next().unwrap())?;
    Ok(TypeSpec { dtype, shape })
}

fn parse_dtype(pair: Pair<Rule>) -> DslResult<DType> {
    match pair.as_str() {
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "f32" => Ok(DType::F32),
        "f64" => Ok(DType::F64),
        "i32" => Ok(DType::I32),
        "i64" => Ok(DType::I64),
        "u32" => Ok(DType::U32),
        "u64" => Ok(DType::U64),
        "bool" => Ok(DType::Bool),
        other => Err(DslError::UnknownDType(other.to_string())),
    }
}

fn parse_shape(pair: Pair<Rule>) -> DslResult<Vec<ShapeDim>> {
    let mut dims = Vec::new();
    for dim_pair in pair.into_inner() {
        dims.push(parse_dim(dim_pair)?);
    }
    Ok(dims)
}

fn parse_dim(pair: Pair<Rule>) -> DslResult<ShapeDim> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::integer => {
            let n: i64 = inner.as_str().parse().unwrap();
            Ok(ShapeDim::Static(n))
        }
        Rule::ident => Ok(ShapeDim::Dynamic(inner.as_str().to_string())),
        _ => unreachable!(),
    }
}

fn parse_body(pair: Pair<Rule>) -> DslResult<(Vec<Statement>, DslExpr)> {
    let mut statements = Vec::new();
    let mut return_expr = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::statement => {
                statements.push(parse_statement(inner)?);
            }
            Rule::return_stmt => {
                return_expr = Some(parse_return_stmt(inner)?);
            }
            _ => {}
        }
    }

    Ok((statements, return_expr.expect("return statement required")))
}

fn parse_statement(pair: Pair<Rule>) -> DslResult<Statement> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::let_stmt => parse_let_stmt(inner),
        _ => unreachable!(),
    }
}

fn parse_let_stmt(pair: Pair<Rule>) -> DslResult<Statement> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let value = parse_expr(inner.next().unwrap())?;
    Ok(Statement::Let { name, value })
}

fn parse_return_stmt(pair: Pair<Rule>) -> DslResult<DslExpr> {
    let inner = pair.into_inner().next().unwrap();
    parse_expr(inner)
}

fn parse_expr(pair: Pair<Rule>) -> DslResult<DslExpr> {
    parse_comparison(pair.into_inner().next().unwrap())
}

fn parse_comparison(pair: Pair<Rule>) -> DslResult<DslExpr> {
    let mut inner = pair.into_inner();
    let mut lhs = parse_additive(inner.next().unwrap())?;

    while let Some(op_pair) = inner.next() {
        let op = parse_comp_op(op_pair)?;
        let rhs = parse_additive(inner.next().unwrap())?;
        lhs = DslExpr::BinaryOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
    }

    Ok(lhs)
}

fn parse_comp_op(pair: Pair<Rule>) -> DslResult<BinOp> {
    match pair.as_str() {
        "<" => Ok(BinOp::Lt),
        ">" => Ok(BinOp::Gt),
        "<=" => Ok(BinOp::Le),
        ">=" => Ok(BinOp::Ge),
        "==" => Ok(BinOp::Eq),
        "!=" => Ok(BinOp::Ne),
        other => Err(DslError::UnknownCompOp(other.to_string())),
    }
}

fn parse_additive(pair: Pair<Rule>) -> DslResult<DslExpr> {
    let mut inner = pair.into_inner();
    let mut lhs = parse_multiplicative(inner.next().unwrap())?;

    while let Some(op_pair) = inner.next() {
        let op = match op_pair.as_str() {
            "+" => BinOp::Add,
            "-" => BinOp::Sub,
            other => return Err(DslError::UnknownBinOp(other.to_string())),
        };
        let rhs = parse_multiplicative(inner.next().unwrap())?;
        lhs = DslExpr::BinaryOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
    }

    Ok(lhs)
}

fn parse_multiplicative(pair: Pair<Rule>) -> DslResult<DslExpr> {
    let mut inner = pair.into_inner();
    let mut lhs = parse_unary(inner.next().unwrap())?;

    while let Some(op_pair) = inner.next() {
        let op = match op_pair.as_str() {
            "*" => BinOp::Mul,
            "/" => BinOp::Div,
            other => return Err(DslError::UnknownBinOp(other.to_string())),
        };
        let rhs = parse_unary(inner.next().unwrap())?;
        lhs = DslExpr::BinaryOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
    }

    Ok(lhs)
}

fn parse_unary(pair: Pair<Rule>) -> DslResult<DslExpr> {
    let mut inner = pair.into_inner().peekable();

    // Check for negation operator
    if inner.peek().map(|p| p.as_rule()) == Some(Rule::neg_op) {
        inner.next(); // consume neg_op
        let operand = parse_unary(inner.next().unwrap())?;
        return Ok(DslExpr::UnaryOp {
            op: UnaryOp::Neg,
            operand: Box::new(operand),
        });
    }

    parse_primary(inner.next().unwrap())
}

fn parse_primary(pair: Pair<Rule>) -> DslResult<DslExpr> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::func_call => parse_func_call(inner),
        Rule::literal => parse_literal(inner),
        Rule::ident => Ok(DslExpr::Var(inner.as_str().to_string())),
        Rule::expr => parse_expr(inner),
        _ => unreachable!("unexpected rule: {:?}", inner.as_rule()),
    }
}

fn parse_literal(pair: Pair<Rule>) -> DslResult<DslExpr> {
    let inner = pair.into_inner().next().unwrap();
    let value: f64 = inner.as_str().parse().unwrap();
    Ok(DslExpr::Literal(value))
}

fn parse_func_call(pair: Pair<Rule>) -> DslResult<DslExpr> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();

    let mut args = Vec::new();
    for arg_pair in inner {
        args.push(parse_arg(arg_pair)?);
    }

    Ok(DslExpr::FuncCall { name, args })
}

fn parse_arg(pair: Pair<Rule>) -> DslResult<FuncArg> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::named_arg => parse_named_arg(inner),
        Rule::shape_arg => parse_shape_arg(inner),
        Rule::expr_arg => {
            let expr = parse_expr(inner.into_inner().next().unwrap())?;
            Ok(FuncArg::Positional(expr))
        }
        _ => unreachable!("unexpected arg rule: {:?}", inner.as_rule()),
    }
}

fn parse_named_arg(pair: Pair<Rule>) -> DslResult<FuncArg> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let value_pair = inner.next().unwrap();

    match value_pair.as_rule() {
        Rule::shape => {
            let shape = parse_shape(value_pair)?;
            Ok(FuncArg::NamedShape { name, shape })
        }
        Rule::expr => {
            let value = parse_expr(value_pair)?;
            Ok(FuncArg::Named { name, value })
        }
        _ => unreachable!(),
    }
}

fn parse_shape_arg(pair: Pair<Rule>) -> DslResult<FuncArg> {
    let mut dims = Vec::new();
    for dim_pair in pair.into_inner() {
        // dim_pair is a `dim` rule, which contains either integer or ident
        dims.push(parse_dim(dim_pair)?);
    }
    Ok(FuncArg::Shape(dims))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_add() {
        let source = r#"
program {
    graph add(a: f32[16, 256, 256], b: f32[16, 256, 256]) -> f32[16, 256, 256] {
        return a + b;
    }
}
"#;
        let result = parse_program(source);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.graphs.len(), 1);
        assert_eq!(program.graphs[0].name, "add");
        assert_eq!(program.graphs[0].params.len(), 2);
    }

    #[test]
    fn test_parse_softmax() {
        let source = r#"
program {
    graph softmax(x: f32[32, 64]) -> f32[32, 64] {
        let x_max = max(x, axis=1);
        let x_shifted = x - expand(x_max, [32, 64]);
        let exp_x = exp(x_shifted);
        let sum_exp = sum(exp_x, axis=1);
        return exp_x / expand(sum_exp, [32, 64]);
    }
}
"#;
        let result = parse_program(source);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.graphs.len(), 1);
        assert_eq!(program.graphs[0].name, "softmax");
        assert_eq!(program.graphs[0].body.len(), 4);
    }

    #[test]
    fn test_parse_dynamic_shape() {
        let source = r#"
program {
    graph relu(x: f32[batch, n]) -> f32[batch, n] {
        return where(x > 0.0, x, 0.0);
    }
}
"#;
        let result = parse_program(source);
        assert!(result.is_ok());
        let program = result.unwrap();
        let params = &program.graphs[0].params;
        assert!(
            matches!(&params[0].type_spec.shape[0], ShapeDim::Dynamic(name) if name == "batch")
        );
        assert!(matches!(&params[0].type_spec.shape[1], ShapeDim::Dynamic(name) if name == "n"));
    }

    #[test]
    fn test_parse_multiple_graphs() {
        let source = r#"
program {
    graph relu(x: f32[32, 64]) -> f32[32, 64] {
        return where(x > 0.0, x, 0.0);
    }

    graph sigmoid(x: f32[32, 64]) -> f32[32, 64] {
        return recip(1.0 + exp(-x));
    }
}
"#;
        let result = parse_program(source);
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.graphs.len(), 2);
        assert_eq!(program.graphs[0].name, "relu");
        assert_eq!(program.graphs[1].name, "sigmoid");
    }

    #[test]
    fn test_parse_comparison_ops() {
        let source = r#"
program {
    graph test(x: f32[10]) -> bool[10] {
        return x <= 5.0;
    }
}
"#;
        let result = parse_program(source);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_unary_neg() {
        let source = r#"
program {
    graph test(x: f32[10]) -> f32[10] {
        return -x;
    }
}
"#;
        let result = parse_program(source);
        assert!(result.is_ok());
        let program = result.unwrap();
        match &program.graphs[0].return_expr {
            DslExpr::UnaryOp { op, .. } => assert_eq!(*op, UnaryOp::Neg),
            _ => panic!("expected unary op"),
        }
    }
}
