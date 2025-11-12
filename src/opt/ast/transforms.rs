//! AST変換関数
//!
//! ループのタイル化やループ展開などの複雑なAST変換を提供します。

use crate::ast::{AstNode, Literal, Scope};

/// ループ回数が固定かつ小さいfor文をインライン展開する
///
/// # 引数
/// * `loop_node` - Rangeノード
/// * `max_iterations` - 展開する最大反復回数（これを超えると展開しない）
///
/// # 変換例
/// ```text
/// // 元のループ (0..4)
/// for i in 0..4 step 1 {
///   body(i)
/// }
///
/// // 展開後
/// {
///   body(0)
///   body(1)
///   body(2)
///   body(3)
/// }
/// ```
pub fn inline_small_loop(loop_node: &AstNode, max_iterations: usize) -> Option<AstNode> {
    match loop_node {
        AstNode::Range {
            var,
            start,
            step,
            stop,
            body,
        } => {
            // start, step, stopが全て定数の場合のみ展開可能
            let start_val = match start.as_ref() {
                AstNode::Const(Literal::Int(v)) => *v as usize,
                _ => return None,
            };

            let step_val = match step.as_ref() {
                AstNode::Const(Literal::Int(v)) if *v > 0 => *v as usize,
                _ => return None, // ステップが定数でないか、0以下の場合は展開不可
            };

            let stop_val = match stop.as_ref() {
                AstNode::Const(Literal::Int(v)) => *v as usize,
                _ => return None,
            };

            // ループ回数を計算
            if start_val >= stop_val {
                // ループが実行されない
                return Some(AstNode::Block {
                    statements: vec![],
                    scope: Box::new(Scope::new()),
                });
            }

            let iterations = (stop_val - start_val + step_val - 1) / step_val;

            // 反復回数が多すぎる場合は展開しない
            if iterations > max_iterations {
                return None;
            }

            // 各反復でループ変数を置き換えたbodyを生成
            let mut statements = Vec::new();
            let mut current = start_val;

            while current < stop_val {
                let var_value = Box::new(AstNode::Const(Literal::Int(current as isize)));
                let replaced_body = replace_var_in_ast(body, var, &var_value);
                statements.push(replaced_body);
                current += step_val;
            }

            Some(AstNode::Block {
                statements,
                scope: Box::new(Scope::new()),
            })
        }
        _ => None,
    }
}

/// ループをタイル化する
///
/// # 引数
/// * `loop_node` - Rangeノード
/// * `tile_size` - タイルサイズ
///
/// # 変換例
/// ```text
/// // 元のループ
/// for i in 0..N step 1 {
///   body(i)
/// }
///
/// // タイル化後
/// for i_outer in start..(stop/tile_size)*tile_size step tile_size {
///   for i_inner in 0..tile_size step 1 {
///     i = i_outer + i_inner
///     body(i)
///   }
/// }
/// // 端数処理
/// for i in (stop/tile_size)*tile_size..stop step 1 {
///   body(i)
/// }
/// ```
pub fn tile_loop(loop_node: &AstNode, tile_size: usize) -> Option<AstNode> {
    if tile_size <= 1 {
        return None; // タイル化しない
    }

    match loop_node {
        AstNode::Range {
            var,
            start,
            step,
            stop,
            body,
        } => {
            // 既にタイル化されたループ（_outer, _inner, _remainderを含む変数名）は再度タイル化しない
            if var.contains("_outer") || var.contains("_inner") || var.contains("_remainder") {
                log::trace!("Skipping tiling for already tiled loop: {}", var);
                return None;
            }

            // ステップが1の場合のみタイル化可能（簡易実装）
            let is_step_one = matches!(step.as_ref(), AstNode::Const(Literal::Int(1)));
            if !is_step_one {
                log::trace!("Skipping tiling for loop with step != 1: {}", var);
                return None;
            }

            // stopが定数の場合のみタイル化可能（簡易実装）
            // これによりタイル化後のRangeノードのstopフィールドが必ず定数になる
            let stop_val = match stop.as_ref() {
                AstNode::Const(Literal::Int(v)) if *v >= 0 => *v as usize,
                _ => {
                    log::trace!("Skipping tiling for loop with non-constant stop: {}", var);
                    return None;
                }
            };

            log::debug!(
                "Tiling loop: {} with tile_size: {}, stop_val: {}",
                var,
                tile_size,
                stop_val
            );
            let make_zero = || AstNode::Const(Literal::Int(0));
            let make_one = || AstNode::Const(Literal::Int(1));
            let make_tile_size = || AstNode::Const(Literal::Int(tile_size as isize));

            let outer_var = format!("{}_outer", var);
            let inner_var = format!("{}_inner", var);
            let remainder_var = format!("{}_remainder", var);

            // 内側ループの本体: i = i_outer + i_inner; body(i)
            let i_expr = Box::new(AstNode::Add(
                Box::new(AstNode::Var(outer_var.clone())),
                Box::new(AstNode::Var(inner_var.clone())),
            ));

            // i = i_outer + i_inner の代入
            let assign_i = AstNode::Assign {
                var: var.clone(),
                value: i_expr.clone(),
            };

            let inner_body_statements = vec![assign_i, body.as_ref().clone()];

            let inner_body = AstNode::Block {
                statements: inner_body_statements,
                scope: Box::new(Scope::new()),
            };

            // 内側ループ: for i_inner in 0..tile_size step 1
            let inner_loop = AstNode::Range {
                var: inner_var.clone(),
                start: Box::new(make_zero()),
                step: Box::new(make_one()),
                stop: Box::new(make_tile_size()),
                body: Box::new(inner_body),
            };

            // メインループの終了値: (stop_val / tile_size) * tile_size
            // stopが定数であることが保証されているので、main_stopも常に定数
            let aligned_stop = (stop_val / tile_size) * tile_size;
            let main_stop = Box::new(AstNode::Const(Literal::Int(aligned_stop as isize)));

            // 外側ループ: for i_outer in start..main_stop step tile_size
            let outer_loop = AstNode::Range {
                var: outer_var.clone(),
                start: start.clone(),
                step: Box::new(make_tile_size()),
                stop: main_stop.clone(),
                body: Box::new(inner_loop),
            };

            // 端数処理ループ: for i_remainder in main_stop..stop step 1
            // ループ本体の中で元の変数名を使うため、代入を追加
            let remainder_assign = AstNode::Assign {
                var: var.clone(),
                value: Box::new(AstNode::Var(remainder_var.clone())),
            };
            let remainder_body = AstNode::Block {
                statements: vec![remainder_assign, body.as_ref().clone()],
                scope: Box::new(Scope::new()),
            };
            let remainder_loop = AstNode::Range {
                var: remainder_var,
                start: main_stop,
                step: step.clone(),
                stop: stop.clone(),
                body: Box::new(remainder_body),
            };

            // メインループと端数処理ループを含むBlock
            Some(AstNode::Block {
                statements: vec![outer_loop, remainder_loop],
                scope: Box::new(Scope::new()),
            })
        }
        _ => None, // Rangeノードでない場合は変換不可
    }
}

/// AST内の変数を置き換える
fn replace_var_in_ast(ast: &AstNode, var_name: &str, replacement: &AstNode) -> AstNode {
    match ast {
        AstNode::Var(name) if name == var_name => replacement.clone(),
        AstNode::Add(a, b) => AstNode::Add(
            Box::new(replace_var_in_ast(a, var_name, replacement)),
            Box::new(replace_var_in_ast(b, var_name, replacement)),
        ),
        AstNode::Mul(a, b) => AstNode::Mul(
            Box::new(replace_var_in_ast(a, var_name, replacement)),
            Box::new(replace_var_in_ast(b, var_name, replacement)),
        ),
        AstNode::Max(a, b) => AstNode::Max(
            Box::new(replace_var_in_ast(a, var_name, replacement)),
            Box::new(replace_var_in_ast(b, var_name, replacement)),
        ),
        AstNode::Rem(a, b) => AstNode::Rem(
            Box::new(replace_var_in_ast(a, var_name, replacement)),
            Box::new(replace_var_in_ast(b, var_name, replacement)),
        ),
        AstNode::Idiv(a, b) => AstNode::Idiv(
            Box::new(replace_var_in_ast(a, var_name, replacement)),
            Box::new(replace_var_in_ast(b, var_name, replacement)),
        ),
        AstNode::Recip(a) => AstNode::Recip(Box::new(replace_var_in_ast(a, var_name, replacement))),
        AstNode::Sqrt(a) => AstNode::Sqrt(Box::new(replace_var_in_ast(a, var_name, replacement))),
        AstNode::Log2(a) => AstNode::Log2(Box::new(replace_var_in_ast(a, var_name, replacement))),
        AstNode::Exp2(a) => AstNode::Exp2(Box::new(replace_var_in_ast(a, var_name, replacement))),
        AstNode::Sin(a) => AstNode::Sin(Box::new(replace_var_in_ast(a, var_name, replacement))),
        AstNode::Cast(a, dtype) => AstNode::Cast(
            Box::new(replace_var_in_ast(a, var_name, replacement)),
            dtype.clone(),
        ),
        AstNode::Load {
            ptr,
            offset,
            count,
            dtype,
        } => AstNode::Load {
            ptr: Box::new(replace_var_in_ast(ptr, var_name, replacement)),
            offset: Box::new(replace_var_in_ast(offset, var_name, replacement)),
            count: *count,
            dtype: dtype.clone(),
        },
        AstNode::Store { ptr, offset, value } => AstNode::Store {
            ptr: Box::new(replace_var_in_ast(ptr, var_name, replacement)),
            offset: Box::new(replace_var_in_ast(offset, var_name, replacement)),
            value: Box::new(replace_var_in_ast(value, var_name, replacement)),
        },
        AstNode::Block { statements, scope } => AstNode::Block {
            statements: statements
                .iter()
                .map(|s| replace_var_in_ast(s, var_name, replacement))
                .collect(),
            scope: scope.clone(),
        },
        AstNode::Range {
            var,
            start,
            step,
            stop,
            body,
        } => {
            // ループ変数がシャドウイングされている場合は置き換えない
            if var == var_name {
                ast.clone()
            } else {
                AstNode::Range {
                    var: var.clone(),
                    start: Box::new(replace_var_in_ast(start, var_name, replacement)),
                    step: Box::new(replace_var_in_ast(step, var_name, replacement)),
                    stop: Box::new(replace_var_in_ast(stop, var_name, replacement)),
                    body: Box::new(replace_var_in_ast(body, var_name, replacement)),
                }
            }
        }
        AstNode::Assign { var, value } => AstNode::Assign {
            var: var.clone(),
            value: Box::new(replace_var_in_ast(value, var_name, replacement)),
        },
        // その他のノードは再帰的に処理
        _ => ast.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inline_small_loop() {
        // for i in 0..4 step 1 { Store(ptr, i, i) }
        let body = Box::new(AstNode::Store {
            ptr: Box::new(AstNode::Var("ptr".to_string())),
            offset: Box::new(AstNode::Var("i".to_string())),
            value: Box::new(AstNode::Var("i".to_string())),
        });

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(4))),
            body,
        };

        let inlined = inline_small_loop(&loop_node, 10);
        assert!(inlined.is_some());

        // 展開結果がBlockノードで4つのstatementを持つか確認
        if let Some(AstNode::Block { statements, .. }) = inlined {
            assert_eq!(statements.len(), 4);

            // 各statementでiが0, 1, 2, 3に置き換えられているか確認
            for (idx, stmt) in statements.iter().enumerate() {
                if let AstNode::Store { offset, value, .. } = stmt {
                    assert_eq!(**offset, AstNode::Const(Literal::Int(idx as isize)));
                    assert_eq!(**value, AstNode::Const(Literal::Int(idx as isize)));
                } else {
                    panic!("Expected Store node");
                }
            }
        } else {
            panic!("Expected Block node");
        }
    }

    #[test]
    fn test_inline_small_loop_too_large() {
        let body = Box::new(AstNode::Var("x".to_string()));

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(100))),
            body,
        };

        // max_iterations=10なので、100回のループは展開されない
        let inlined = inline_small_loop(&loop_node, 10);
        assert!(inlined.is_none());
    }

    #[test]
    fn test_inline_small_loop_with_step() {
        // for i in 0..10 step 2 { body(i) }
        let body = Box::new(AstNode::Var("i".to_string()));

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(2))),
            stop: Box::new(AstNode::Const(Literal::Int(10))),
            body,
        };

        let inlined = inline_small_loop(&loop_node, 10);
        assert!(inlined.is_some());

        // 5回の反復（0, 2, 4, 6, 8）
        if let Some(AstNode::Block { statements, .. }) = inlined {
            assert_eq!(statements.len(), 5);
        } else {
            panic!("Expected Block node");
        }
    }

    #[test]
    fn test_tile_loop_simple() {
        // for i in 0..16 step 1 { Store(ptr, i, i) }
        let body = Box::new(AstNode::Store {
            ptr: Box::new(AstNode::Var("ptr".to_string())),
            offset: Box::new(AstNode::Var("i".to_string())),
            value: Box::new(AstNode::Var("i".to_string())),
        });

        let original_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(16))),
            body,
        };

        let tiled = tile_loop(&original_loop, 4);
        assert!(tiled.is_some());

        // タイル化結果がBlockノードになっているか確認
        if let Some(AstNode::Block { statements, .. }) = tiled {
            assert_eq!(statements.len(), 2); // メインループ + 端数ループ

            // 最初のstatementがメインループ（外側ループ）
            if let AstNode::Range {
                var,
                step,
                body: outer_body,
                ..
            } = &statements[0]
            {
                assert_eq!(var, "i_outer");
                assert_eq!(**step, AstNode::Const(Literal::Int(4))); // tile_size

                // 内側ループが存在するか確認
                assert!(matches!(outer_body.as_ref(), AstNode::Range { .. }));
            } else {
                panic!("Expected Range node for main loop");
            }

            // 2番目のstatementが端数処理ループ
            if let AstNode::Range { var, .. } = &statements[1] {
                assert_eq!(var, "i_remainder");
            } else {
                panic!("Expected Range node for remainder loop");
            }
        } else {
            panic!("Expected Block node");
        }
    }

    #[test]
    fn test_tile_loop_with_remainder() {
        // for i in 0..10 step 1 { body } (10 = 4*2 + 2, 端数あり)
        let body = Box::new(AstNode::Var("x".to_string()));

        let original_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(10))),
            body,
        };

        let tiled = tile_loop(&original_loop, 4);
        assert!(tiled.is_some());

        // メインループ + 端数ループが含まれることを確認
        if let Some(AstNode::Block { statements, .. }) = tiled {
            assert_eq!(statements.len(), 2);
        } else {
            panic!("Expected Block node");
        }
    }

    #[test]
    fn test_tile_loop_size_one() {
        let body = Box::new(AstNode::Var("x".to_string()));

        let original_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(16))),
            body,
        };

        let tiled = tile_loop(&original_loop, 1);
        assert!(tiled.is_none()); // tile_size=1ではタイル化しない
    }

    #[test]
    fn test_replace_var_in_ast() {
        let ast = AstNode::Add(
            Box::new(AstNode::Var("i".to_string())),
            Box::new(AstNode::Const(Literal::Int(5))),
        );

        let replacement = AstNode::Mul(
            Box::new(AstNode::Var("j".to_string())),
            Box::new(AstNode::Const(Literal::Int(2))),
        );

        let result = replace_var_in_ast(&ast, "i", &replacement);

        // i が (j * 2) に置き換えられているか確認
        if let AstNode::Add(left, right) = result {
            assert_eq!(*left, replacement);
            assert_eq!(*right, AstNode::Const(Literal::Int(5)));
        } else {
            panic!("Expected Add node");
        }
    }
}
