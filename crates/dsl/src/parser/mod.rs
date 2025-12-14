//! Parser for Harp DSL

pub mod ast;

use ast::*;
use pest::Parser;
use pest_derive::Parser;

use crate::error::DslError;

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct DslParser;

/// 予約語リスト
/// これらの識別子は変数名やグラフ名として使用できない
///
/// 注意: 型名（f32, i32等）、組み込み関数名（matmul, fused等）、メソッド名（sum等）は
/// 文脈で区別できるため、予約語に含めていません。
pub const RESERVED_KEYWORDS: &[&str] = &[
    // 構文キーワード
    "graph",
    "let",
    "return",
    // 真偽値リテラル（将来的にブールリテラルとして使用予定）
    "true",
    "false",
];

/// 識別子が予約語かどうかをチェック
fn check_reserved_keyword(name: &str, pair: &pest::iterators::Pair<Rule>) -> Result<(), DslError> {
    if RESERVED_KEYWORDS.contains(&name) {
        let (line, column) = pair.line_col();
        return Err(DslError::ParseError {
            line,
            column,
            message: format!(
                "'{}' is a reserved keyword and cannot be used as an identifier",
                name
            ),
        });
    }
    Ok(())
}

/// Parse DSL source code into a module AST
pub fn parse(source: &str) -> Result<DslModule, DslError> {
    let pairs = DslParser::parse(Rule::module, source).map_err(DslError::from_pest_error)?;

    let mut graphs = Vec::new();
    for pair in pairs {
        match pair.as_rule() {
            Rule::module => {
                // The module rule contains the inner graph_def rules
                for inner_pair in pair.into_inner() {
                    match inner_pair.as_rule() {
                        Rule::graph_def => {
                            graphs.push(parse_graph_def(inner_pair)?);
                        }
                        Rule::EOI => {}
                        _ => {}
                    }
                }
            }
            Rule::graph_def => {
                graphs.push(parse_graph_def(pair)?);
            }
            Rule::EOI => {}
            _ => {}
        }
    }

    Ok(DslModule { graphs })
}

fn parse_graph_def(pair: pest::iterators::Pair<Rule>) -> Result<DslGraph, DslError> {
    let inner = pair.into_inner();

    // Parse generic params if present
    let mut shape_vars = Vec::new();
    let mut name = String::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut body = Vec::new();

    for p in inner {
        match p.as_rule() {
            Rule::generic_params => {
                for ident in p.into_inner() {
                    if ident.as_rule() == Rule::ident {
                        let var_name = ident.as_str();
                        check_reserved_keyword(var_name, &ident)?;
                        shape_vars.push(var_name.to_string());
                    }
                }
            }
            Rule::ident => {
                let graph_name = p.as_str();
                check_reserved_keyword(graph_name, &p)?;
                name = graph_name.to_string();
            }
            Rule::param_list => {
                // Input parameters (before ->)
                inputs = parse_param_list(p)?;
            }
            Rule::return_type => {
                // Return type: either (param_list) or tensor_type
                outputs = parse_return_type(p)?;
            }
            Rule::block => {
                body = parse_block(p)?;
            }
            _ => {}
        }
    }

    Ok(DslGraph {
        name,
        shape_vars,
        inputs,
        outputs,
        body,
    })
}

fn parse_return_type(pair: pest::iterators::Pair<Rule>) -> Result<Vec<DslParam>, DslError> {
    let inner = pair.into_inner().next();

    match inner {
        Some(p) => match p.as_rule() {
            Rule::param_list => parse_param_list(p),
            Rule::tensor_type => {
                // Single unnamed output: f32[N, M] -> output: f32[N, M]
                let (dtype, shape) = parse_tensor_type(p)?;
                Ok(vec![DslParam {
                    name: "output".to_string(),
                    dtype,
                    shape,
                }])
            }
            _ => Ok(Vec::new()),
        },
        None => Ok(Vec::new()), // Empty return type: -> ()
    }
}

fn parse_param_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<DslParam>, DslError> {
    let mut params = Vec::new();
    for p in pair.into_inner() {
        if p.as_rule() == Rule::param {
            params.push(parse_param(p)?);
        }
    }
    Ok(params)
}

fn parse_param(pair: pest::iterators::Pair<Rule>) -> Result<DslParam, DslError> {
    let mut inner = pair.into_inner();
    let name_pair = inner.next().unwrap();
    let name = name_pair.as_str();
    check_reserved_keyword(name, &name_pair)?;

    let tensor_type = inner.next().unwrap();
    let (dtype, shape) = parse_tensor_type(tensor_type)?;

    Ok(DslParam {
        name: name.to_string(),
        dtype,
        shape,
    })
}

fn parse_tensor_type(
    pair: pest::iterators::Pair<Rule>,
) -> Result<(DslDType, Vec<ShapeExpr>), DslError> {
    let mut inner = pair.into_inner();
    let dtype_pair = inner.next().unwrap();
    let dtype = match dtype_pair.as_str() {
        "f32" => DslDType::F32,
        "i32" => DslDType::I32,
        "bool" => DslDType::Bool,
        "c64" => DslDType::C64,
        _ => {
            return Err(DslError::ParseError {
                line: 0,
                column: 0,
                message: format!("Unknown dtype: {}", dtype_pair.as_str()),
            });
        }
    };

    let shape_list = inner.next().unwrap();
    let mut shape = Vec::new();
    for shape_expr in shape_list.into_inner() {
        shape.push(parse_shape_expr(shape_expr)?);
    }

    Ok((dtype, shape))
}

fn parse_shape_expr(pair: pest::iterators::Pair<Rule>) -> Result<ShapeExpr, DslError> {
    let mut inner = pair.into_inner().peekable();
    let mut result = parse_shape_term(inner.next().unwrap())?;

    while let Some(op_pair) = inner.next() {
        let op = match op_pair.as_str() {
            "+" => ShapeBinOp::Add,
            "-" => ShapeBinOp::Sub,
            _ => continue,
        };
        let rhs = parse_shape_term(inner.next().unwrap())?;
        result = ShapeExpr::BinOp {
            op,
            lhs: Box::new(result),
            rhs: Box::new(rhs),
        };
    }

    Ok(result)
}

fn parse_shape_term(pair: pest::iterators::Pair<Rule>) -> Result<ShapeExpr, DslError> {
    let mut inner = pair.into_inner().peekable();
    let mut result = parse_shape_factor(inner.next().unwrap())?;

    while let Some(op_pair) = inner.next() {
        let op = match op_pair.as_str() {
            "*" => ShapeBinOp::Mul,
            "/" => ShapeBinOp::Div,
            "%" => ShapeBinOp::Rem,
            _ => continue,
        };
        let rhs = parse_shape_factor(inner.next().unwrap())?;
        result = ShapeExpr::BinOp {
            op,
            lhs: Box::new(result),
            rhs: Box::new(rhs),
        };
    }

    Ok(result)
}

fn parse_shape_factor(pair: pest::iterators::Pair<Rule>) -> Result<ShapeExpr, DslError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::integer => {
            let val: isize = inner.as_str().parse().map_err(|_| DslError::ParseError {
                line: 0,
                column: 0,
                message: format!("Invalid integer: {}", inner.as_str()),
            })?;
            Ok(ShapeExpr::Const(val))
        }
        Rule::ident => {
            let name = inner.as_str();
            check_reserved_keyword(name, &inner)?;
            Ok(ShapeExpr::Var(name.to_string()))
        }
        Rule::shape_expr => parse_shape_expr(inner),
        _ => Err(DslError::ParseError {
            line: 0,
            column: 0,
            message: format!("Unexpected shape factor: {:?}", inner.as_rule()),
        }),
    }
}

fn parse_block(pair: pest::iterators::Pair<Rule>) -> Result<Vec<DslStatement>, DslError> {
    let mut statements = Vec::new();
    for p in pair.into_inner() {
        if p.as_rule() == Rule::statement {
            statements.push(parse_statement(p)?);
        }
    }
    Ok(statements)
}

fn parse_statement(pair: pest::iterators::Pair<Rule>) -> Result<DslStatement, DslError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::let_statement => {
            let mut parts = inner.into_inner();
            let name_pair = parts.next().unwrap();
            let name = name_pair.as_str();
            check_reserved_keyword(name, &name_pair)?;
            let value = parse_expr(parts.next().unwrap())?;
            Ok(DslStatement::Let {
                name: name.to_string(),
                value,
            })
        }
        Rule::assign_statement => {
            let mut parts = inner.into_inner();
            let name_pair = parts.next().unwrap();
            let name = name_pair.as_str();
            check_reserved_keyword(name, &name_pair)?;
            let value = parse_expr(parts.next().unwrap())?;
            Ok(DslStatement::Assign {
                name: name.to_string(),
                value,
            })
        }
        Rule::return_statement => {
            let mut parts = inner.into_inner();
            let name = parts.next().unwrap().as_str().to_string();
            // return文の識別子はチェック不要（変数参照なので）
            Ok(DslStatement::Return { name })
        }
        _ => Err(DslError::ParseError {
            line: 0,
            column: 0,
            message: format!("Unexpected statement: {:?}", inner.as_rule()),
        }),
    }
}

fn parse_expr(pair: pest::iterators::Pair<Rule>) -> Result<DslExpr, DslError> {
    // Handle expression hierarchy
    let inner = pair.into_inner().next().unwrap();
    parse_comparison(inner)
}

fn parse_comparison(pair: pest::iterators::Pair<Rule>) -> Result<DslExpr, DslError> {
    let mut inner = pair.into_inner().peekable();
    let mut result = parse_additive(inner.next().unwrap())?;

    while let Some(op_pair) = inner.next() {
        if op_pair.as_rule() != Rule::comp_op {
            continue;
        }
        let op = match op_pair.as_str() {
            "==" => DslBinOp::Eq,
            "!=" => DslBinOp::Ne,
            "<" => DslBinOp::Lt,
            "<=" => DslBinOp::Le,
            ">" => DslBinOp::Gt,
            ">=" => DslBinOp::Ge,
            _ => continue,
        };
        let rhs = parse_additive(inner.next().unwrap())?;
        result = DslExpr::BinOp {
            op,
            lhs: Box::new(result),
            rhs: Box::new(rhs),
        };
    }

    Ok(result)
}

fn parse_additive(pair: pest::iterators::Pair<Rule>) -> Result<DslExpr, DslError> {
    let mut inner = pair.into_inner().peekable();
    let mut result = parse_multiplicative(inner.next().unwrap())?;

    while let Some(op_pair) = inner.next() {
        if op_pair.as_rule() != Rule::add_op {
            continue;
        }
        let op = match op_pair.as_str() {
            "+" => DslBinOp::Add,
            "-" => DslBinOp::Sub,
            _ => continue,
        };
        let rhs = parse_multiplicative(inner.next().unwrap())?;
        result = DslExpr::BinOp {
            op,
            lhs: Box::new(result),
            rhs: Box::new(rhs),
        };
    }

    Ok(result)
}

fn parse_multiplicative(pair: pest::iterators::Pair<Rule>) -> Result<DslExpr, DslError> {
    let mut inner = pair.into_inner().peekable();
    let mut result = parse_unary(inner.next().unwrap())?;

    while let Some(op_pair) = inner.next() {
        if op_pair.as_rule() != Rule::mul_op {
            continue;
        }
        let op = match op_pair.as_str() {
            "*" => DslBinOp::Mul,
            "/" => DslBinOp::Div,
            "%" => DslBinOp::Rem,
            _ => continue,
        };
        let rhs = parse_unary(inner.next().unwrap())?;
        result = DslExpr::BinOp {
            op,
            lhs: Box::new(result),
            rhs: Box::new(rhs),
        };
    }

    Ok(result)
}

fn parse_unary(pair: pest::iterators::Pair<Rule>) -> Result<DslExpr, DslError> {
    let mut inner = pair.into_inner().peekable();
    let first = inner.next().unwrap();

    match first.as_rule() {
        Rule::unary_op => {
            let op = match first.as_str() {
                "-" => DslUnaryOp::Neg,
                "!" => DslUnaryOp::Not,
                _ => {
                    return Err(DslError::ParseError {
                        line: 0,
                        column: 0,
                        message: format!("Unknown unary operator: {}", first.as_str()),
                    });
                }
            };
            let operand = parse_unary(inner.next().unwrap())?;
            Ok(DslExpr::UnaryOp {
                op,
                operand: Box::new(operand),
            })
        }
        Rule::postfix => parse_postfix(first),
        _ => Err(DslError::ParseError {
            line: 0,
            column: 0,
            message: format!("Unexpected unary: {:?}", first.as_rule()),
        }),
    }
}

fn parse_postfix(pair: pest::iterators::Pair<Rule>) -> Result<DslExpr, DslError> {
    let mut inner = pair.into_inner();
    let primary = inner.next().unwrap();
    let mut result = parse_primary(primary)?;

    for postfix_op in inner {
        if postfix_op.as_rule() == Rule::postfix_op {
            let op_inner = postfix_op.into_inner().next().unwrap();
            match op_inner.as_rule() {
                Rule::method_call => {
                    result = parse_method_call(result, op_inner)?;
                }
                Rule::index_access => {
                    let args = parse_arg_list_inner(op_inner.into_inner().next().unwrap())?;
                    let indices = args
                        .into_iter()
                        .filter_map(|a| match a {
                            DslArg::Positional(e) => Some(e),
                            _ => None,
                        })
                        .collect();
                    result = DslExpr::Index {
                        base: Box::new(result),
                        indices,
                    };
                }
                _ => {}
            }
        }
    }

    Ok(result)
}

fn parse_method_call(
    receiver: DslExpr,
    pair: pest::iterators::Pair<Rule>,
) -> Result<DslExpr, DslError> {
    let mut inner = pair.into_inner();
    let method = inner.next().unwrap().as_str().to_string();
    let args = if let Some(arg_list) = inner.next() {
        parse_arg_list_inner(arg_list)?
    } else {
        Vec::new()
    };

    Ok(DslExpr::MethodCall {
        receiver: Box::new(receiver),
        method,
        args,
    })
}

fn parse_primary(pair: pest::iterators::Pair<Rule>) -> Result<DslExpr, DslError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::expr => parse_expr(inner),
        Rule::fused_expr => parse_fused_expr(inner),
        Rule::fused_reduce_expr => parse_fused_reduce_expr(inner),
        Rule::fused_cumulative_expr => parse_fused_cumulative_expr(inner),
        Rule::function_call => {
            let mut parts = inner.into_inner();
            let name = parts.next().unwrap().as_str().to_string();
            let args = if let Some(arg_list) = parts.next() {
                parse_arg_list_inner(arg_list)?
            } else {
                Vec::new()
            };
            Ok(DslExpr::FunctionCall { name, args })
        }
        Rule::float_literal => {
            let val: f64 = inner.as_str().parse().map_err(|_| DslError::ParseError {
                line: 0,
                column: 0,
                message: format!("Invalid float: {}", inner.as_str()),
            })?;
            Ok(DslExpr::FloatLit(val))
        }
        Rule::integer => {
            let val: i64 = inner.as_str().parse().map_err(|_| DslError::ParseError {
                line: 0,
                column: 0,
                message: format!("Invalid integer: {}", inner.as_str()),
            })?;
            Ok(DslExpr::IntLit(val))
        }
        Rule::ident => Ok(DslExpr::Var(inner.as_str().to_string())),
        _ => Err(DslError::ParseError {
            line: 0,
            column: 0,
            message: format!("Unexpected primary: {:?}", inner.as_rule()),
        }),
    }
}

fn parse_arg_list_inner(pair: pest::iterators::Pair<Rule>) -> Result<Vec<DslArg>, DslError> {
    let mut args = Vec::new();
    for arg in pair.into_inner() {
        if arg.as_rule() == Rule::arg {
            args.push(parse_arg(arg)?);
        }
    }
    Ok(args)
}

fn parse_arg(pair: pest::iterators::Pair<Rule>) -> Result<DslArg, DslError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::named_arg => {
            let mut parts = inner.into_inner();
            let name = parts.next().unwrap().as_str().to_string();
            let value_pair = parts.next().unwrap();
            let value = match value_pair.as_rule() {
                Rule::expr => DslArgValue::Expr(parse_expr(value_pair)?),
                Rule::array_literal => DslArgValue::Array(parse_array_literal(value_pair)?),
                Rule::ident => DslArgValue::Ident(value_pair.as_str().to_string()),
                _ => {
                    return Err(DslError::ParseError {
                        line: 0,
                        column: 0,
                        message: format!("Unexpected named arg value: {:?}", value_pair.as_rule()),
                    });
                }
            };
            Ok(DslArg::Named { name, value })
        }
        Rule::expr => Ok(DslArg::Positional(parse_expr(inner)?)),
        Rule::array_literal => Ok(DslArg::Array(parse_array_literal(inner)?)),
        _ => Err(DslError::ParseError {
            line: 0,
            column: 0,
            message: format!("Unexpected arg: {:?}", inner.as_rule()),
        }),
    }
}

fn parse_array_literal(pair: pest::iterators::Pair<Rule>) -> Result<Vec<DslExpr>, DslError> {
    let mut elements = Vec::new();
    for elem in pair.into_inner() {
        if elem.as_rule() == Rule::array_elem {
            let inner = elem.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::expr => elements.push(parse_expr(inner)?),
                Rule::tuple_literal => {
                    // For padding tuples, we create an array with two elements
                    let mut parts = inner.into_inner();
                    let first = parse_expr(parts.next().unwrap())?;
                    let second = parse_expr(parts.next().unwrap())?;
                    elements.push(DslExpr::ArrayLit(vec![first, second]));
                }
                _ => {}
            }
        }
    }
    Ok(elements)
}

fn parse_fused_expr(pair: pest::iterators::Pair<Rule>) -> Result<DslExpr, DslError> {
    let mut inner = pair.into_inner();
    let ident_list = inner.next().unwrap();
    let mut inputs = Vec::new();
    for p in ident_list.into_inner() {
        let name = p.as_str();
        check_reserved_keyword(name, &p)?;
        inputs.push(name.to_string());
    }
    let fused_block = inner.next().unwrap();
    let expr = parse_expr(fused_block.into_inner().next().unwrap())?;

    Ok(DslExpr::FusedElementwise {
        inputs,
        expr: Box::new(expr),
    })
}

fn parse_fused_reduce_expr(pair: pest::iterators::Pair<Rule>) -> Result<DslExpr, DslError> {
    let mut inner = pair.into_inner();
    let ident_list = inner.next().unwrap();
    let mut inputs = Vec::new();
    for p in ident_list.into_inner() {
        let name = p.as_str();
        check_reserved_keyword(name, &p)?;
        inputs.push(name.to_string());
    }

    let fused_params = inner.next().unwrap();
    let (axis, op) = parse_fused_reduce_params(fused_params)?;

    let fused_block = inner.next().unwrap();
    let expr = parse_expr(fused_block.into_inner().next().unwrap())?;

    Ok(DslExpr::FusedReduce {
        inputs,
        axis,
        op,
        expr: Box::new(expr),
    })
}

fn parse_fused_cumulative_expr(pair: pest::iterators::Pair<Rule>) -> Result<DslExpr, DslError> {
    let mut inner = pair.into_inner();
    let ident_list = inner.next().unwrap();
    let mut inputs = Vec::new();
    for p in ident_list.into_inner() {
        let name = p.as_str();
        check_reserved_keyword(name, &p)?;
        inputs.push(name.to_string());
    }

    let fused_params = inner.next().unwrap();
    let (axis, op) = parse_fused_cumulative_params(fused_params)?;

    let fused_block = inner.next().unwrap();
    let expr = parse_expr(fused_block.into_inner().next().unwrap())?;

    Ok(DslExpr::FusedCumulative {
        inputs,
        axis,
        op,
        expr: Box::new(expr),
    })
}

fn parse_fused_reduce_params(
    pair: pest::iterators::Pair<Rule>,
) -> Result<(usize, ReduceOpKind), DslError> {
    let mut axis = 0;
    let mut op = ReduceOpKind::Sum;

    for named_arg in pair.into_inner() {
        if named_arg.as_rule() == Rule::named_arg {
            let mut parts = named_arg.into_inner();
            let name = parts.next().unwrap().as_str();
            let value = parts.next().unwrap();

            match name {
                "axis" => {
                    axis = value.as_str().parse().map_err(|_| DslError::ParseError {
                        line: 0,
                        column: 0,
                        message: "Invalid axis".to_string(),
                    })?;
                }
                "op" => {
                    op = match value.as_str() {
                        "sum" => ReduceOpKind::Sum,
                        "prod" => ReduceOpKind::Prod,
                        "max" => ReduceOpKind::Max,
                        _ => {
                            return Err(DslError::ParseError {
                                line: 0,
                                column: 0,
                                message: format!("Unknown reduce op: {}", value.as_str()),
                            });
                        }
                    };
                }
                _ => {}
            }
        }
    }

    Ok((axis, op))
}

fn parse_fused_cumulative_params(
    pair: pest::iterators::Pair<Rule>,
) -> Result<(usize, CumulativeOpKind), DslError> {
    let mut axis = 0;
    let mut op = CumulativeOpKind::Sum;

    for named_arg in pair.into_inner() {
        if named_arg.as_rule() == Rule::named_arg {
            let mut parts = named_arg.into_inner();
            let name = parts.next().unwrap().as_str();
            let value = parts.next().unwrap();

            match name {
                "axis" => {
                    axis = value.as_str().parse().map_err(|_| DslError::ParseError {
                        line: 0,
                        column: 0,
                        message: "Invalid axis".to_string(),
                    })?;
                }
                "op" => {
                    op = match value.as_str() {
                        "sum" => CumulativeOpKind::Sum,
                        "prod" => CumulativeOpKind::Prod,
                        _ => {
                            return Err(DslError::ParseError {
                                line: 0,
                                column: 0,
                                message: format!("Unknown cumulative op: {}", value.as_str()),
                            });
                        }
                    };
                }
                _ => {}
            }
        }
    }

    Ok((axis, op))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_graph() {
        let source = r#"
            graph add(a: f32[N, M], b: f32[N, M]) -> (c: f32[N, M]) {
                c = a + b
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        assert_eq!(module.graphs.len(), 1);
        assert_eq!(module.graphs[0].name, "add");
        assert_eq!(module.graphs[0].inputs.len(), 2);
        assert_eq!(module.graphs[0].outputs.len(), 1);
    }

    #[test]
    fn test_parse_generic_params() {
        let source = r#"
            graph<L, M, N> matmul(a: f32[L, M], b: f32[M, N]) -> (c: f32[L, N]) {
                c = a
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        assert_eq!(module.graphs[0].shape_vars, vec!["L", "M", "N"]);
    }

    #[test]
    fn test_parse_method_chain() {
        let source = r#"
            graph test(x: f32[N]) -> (y: f32[N]) {
                let a = x.unsqueeze(0)
                y = a.sum(1)
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        assert_eq!(module.graphs[0].body.len(), 2);
    }

    #[test]
    fn test_parse_shape_expr() {
        let source = r#"
            graph test(x: f32[N * 2, M + 1]) -> (y: f32[N, M]) {
                y = x
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        let input = &module.graphs[0].inputs[0];
        assert_eq!(input.shape.len(), 2);
    }

    #[test]
    fn test_parse_simplified_return() {
        // Test simplified return syntax: -> f32[N, M] instead of -> (output: f32[N, M])
        let source = r#"
            graph add(a: f32[N, M], b: f32[N, M]) -> f32[N, M] {
                output = a + b
            }
        "#;

        let module = parse(source).expect("Failed to parse");
        assert_eq!(module.graphs.len(), 1);
        assert_eq!(module.graphs[0].outputs.len(), 1);
        assert_eq!(module.graphs[0].outputs[0].name, "output");
        assert_eq!(module.graphs[0].outputs[0].dtype, DslDType::F32);
    }

    #[test]
    fn test_reserved_keyword_in_graph_name() {
        let source = r#"
            graph graph(a: f32[N]) -> (b: f32[N]) {
                b = a
            }
        "#;
        let result = parse(source);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("reserved keyword"));
    }

    #[test]
    fn test_reserved_keyword_in_let_variable() {
        let source = r#"
            graph test(a: f32[N]) -> (b: f32[N]) {
                let return = a
                b = return
            }
        "#;
        let result = parse(source);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("reserved keyword"));
    }

    #[test]
    fn test_reserved_keyword_in_param_name() {
        let source = r#"
            graph test(let: f32[N]) -> (b: f32[N]) {
                b = let
            }
        "#;
        let result = parse(source);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("reserved keyword"));
    }

    #[test]
    fn test_typename_allowed_as_variable() {
        // 型名は文脈で区別できるため、変数名として使用可能
        let source = r#"
            graph test(a: f32[N]) -> (b: f32[N]) {
                let f32 = a
                b = f32
            }
        "#;
        let result = parse(source);
        assert!(result.is_ok(), "Type names should be allowed as variable names");
    }

    #[test]
    fn test_builtin_function_name_allowed_as_variable() {
        // 組み込み関数名は変数名として使用可能
        let source = r#"
            graph test(a: f32[N]) -> (b: f32[N]) {
                let sum = a
                b = sum
            }
        "#;
        let result = parse(source);
        assert!(
            result.is_ok(),
            "Built-in function names should be allowed as variable names"
        );
    }
}
