//! AST変換関数
//!
//! ループのタイル化やループ展開などの複雑なAST変換を提供します。

use crate::ast::{
    AstNode, DType, Literal, Mutability, Scope,
    helper::{assign, const_int, empty_block, range, var as helper_var},
};
use std::collections::HashSet;

/// AST内のすべての変数名を収集する
pub fn collect_var_names(ast: &AstNode) -> HashSet<String> {
    let mut names = HashSet::new();
    collect_var_names_recursive(ast, &mut names);
    names
}

fn collect_var_names_recursive(ast: &AstNode, names: &mut HashSet<String>) {
    match ast {
        AstNode::Var(name) => {
            names.insert(name.clone());
        }
        AstNode::Range {
            var,
            start,
            step,
            stop,
            body,
        } => {
            names.insert(var.clone());
            collect_var_names_recursive(start, names);
            collect_var_names_recursive(step, names);
            collect_var_names_recursive(stop, names);
            collect_var_names_recursive(body, names);
        }
        AstNode::Assign { var, value } => {
            names.insert(var.clone());
            collect_var_names_recursive(value, names);
        }
        AstNode::Block { statements, .. } => {
            for stmt in statements {
                collect_var_names_recursive(stmt, names);
            }
        }
        AstNode::Function { params, body, .. } => {
            for param in params {
                names.insert(param.name.clone());
            }
            collect_var_names_recursive(body, names);
        }
        AstNode::Kernel { params, body, .. } => {
            for param in params {
                names.insert(param.name.clone());
            }
            collect_var_names_recursive(body, names);
        }
        AstNode::Program { functions, .. } => {
            for func in functions {
                collect_var_names_recursive(func, names);
            }
        }
        AstNode::Add(a, b)
        | AstNode::Mul(a, b)
        | AstNode::Max(a, b)
        | AstNode::Rem(a, b)
        | AstNode::Idiv(a, b)
        | AstNode::BitwiseAnd(a, b)
        | AstNode::BitwiseOr(a, b)
        | AstNode::BitwiseXor(a, b)
        | AstNode::LeftShift(a, b)
        | AstNode::RightShift(a, b) => {
            collect_var_names_recursive(a, names);
            collect_var_names_recursive(b, names);
        }
        AstNode::Recip(a)
        | AstNode::Sqrt(a)
        | AstNode::Log2(a)
        | AstNode::Exp2(a)
        | AstNode::Sin(a)
        | AstNode::BitwiseNot(a)
        | AstNode::Cast(a, _) => {
            collect_var_names_recursive(a, names);
        }
        AstNode::Store { ptr, offset, value } => {
            collect_var_names_recursive(ptr, names);
            collect_var_names_recursive(offset, names);
            collect_var_names_recursive(value, names);
        }
        AstNode::Load { ptr, offset, .. } => {
            collect_var_names_recursive(ptr, names);
            collect_var_names_recursive(offset, names);
        }
        AstNode::Call { args, .. } => {
            for arg in args {
                collect_var_names_recursive(arg, names);
            }
        }
        AstNode::Return { value } => {
            collect_var_names_recursive(value, names);
        }
        AstNode::Allocate { size, .. } => {
            collect_var_names_recursive(size, names);
        }
        AstNode::Deallocate { ptr } => {
            collect_var_names_recursive(ptr, names);
        }
        AstNode::CallKernel { args, .. } => {
            for arg in args {
                collect_var_names_recursive(arg, names);
            }
        }
        // リーフノード
        AstNode::Const(_) | AstNode::Wildcard(_) | AstNode::Rand | AstNode::Barrier => {}
    }
}

/// 使われていない連番のridx変数名を3つ見つける
///
/// 既存の変数名と衝突しない `ridxN` 形式の名前を3つ返します。
fn find_next_ridx_names(used_names: &HashSet<String>) -> (String, String, String) {
    let mut idx = 0;
    let mut result = Vec::new();

    while result.len() < 3 {
        let name = format!("ridx{}", idx);
        if !used_names.contains(&name) {
            result.push(name);
        }
        idx += 1;
    }

    (result[0].clone(), result[1].clone(), result[2].clone())
}

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
            // タイル化されたループの内側・外側ループは展開しない
            // タイル化後のループは本体の最初の文が代入文になっている
            // 端数処理ループも代入文を含むが、端数は小さいので展開を許可する判断は
            // 反復回数チェックで行う
            if is_tiled_loop_body(body) {
                log::trace!("Skipping inlining for tiled loop component: {}", var);
                return None;
            }

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
                return Some(empty_block());
            }

            let iterations = (stop_val - start_val).div_ceil(step_val);

            // 反復回数が多すぎる場合は展開しない
            if iterations > max_iterations {
                return None;
            }

            // 各反復でループ変数を置き換えたbodyを生成
            let mut statements = Vec::new();
            let mut current = start_val;

            while current < stop_val {
                let var_value = const_int(current as isize);
                let replaced_body = replace_var_in_ast(body, var, &var_value);

                // Blockの場合、その中身を直接追加（フラット化）
                if let AstNode::Block {
                    statements: inner_stmts,
                    ..
                } = replaced_body
                {
                    statements.extend(inner_stmts);
                } else {
                    statements.push(replaced_body);
                }

                current += step_val;
            }

            // 元のループ本体のScopeをコピー（変数宣言を保持）
            let scope = if let AstNode::Block { scope, .. } = body.as_ref() {
                scope.clone()
            } else {
                Box::new(Scope::new())
            };

            Some(AstNode::Block { statements, scope })
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
/// for ridx0 in 0..N step 1 {
///   body(ridx0)
/// }
///
/// // タイル化後（ridx1, ridx2, ridx3は既存変数と衝突しない連番）
/// for ridx1 in start..(stop/tile_size)*tile_size step tile_size {
///   for ridx2 in 0..tile_size step 1 {
///     ridx0 = ridx1 + ridx2
///     body(ridx0)
///   }
/// }
/// // 端数処理
/// for ridx3 in (stop/tile_size)*tile_size..stop step 1 {
///   ridx0 = ridx3
///   body(ridx0)
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
            // 既にタイル化されたループは再度タイル化しない
            // タイル化後のループ本体は、最初の文が元変数への代入文になっている
            if is_tiled_loop_body(body) {
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

            // 既存変数名を収集し、衝突しない連番を取得
            let used_names = collect_var_names(loop_node);
            let (outer_var, inner_var, remainder_var) = find_next_ridx_names(&used_names);

            // 内側ループの本体: original_var = outer_var + inner_var; body(original_var)
            let i_expr = helper_var(outer_var.clone()) + helper_var(inner_var.clone());

            // original_var = outer_var + inner_var の代入
            let assign_i = assign(var.clone(), i_expr);

            let inner_body_statements = vec![assign_i, body.as_ref().clone()];

            // 元のループ変数をスコープに宣言（ローカル変数として生成されるように）
            let mut inner_scope = Scope::new();
            inner_scope
                .declare(var.clone(), DType::Int, Mutability::Mutable)
                .expect("Failed to declare loop variable in inner scope");

            let inner_body = AstNode::Block {
                statements: inner_body_statements,
                scope: Box::new(inner_scope),
            };

            // 内側ループ: for inner_var in 0..tile_size step 1
            let inner_loop = range(
                inner_var.clone(),
                const_int(0),
                const_int(1),
                const_int(tile_size as isize),
                inner_body,
            );

            // メインループの終了値: (stop_val / tile_size) * tile_size
            // stopが定数であることが保証されているので、main_stopも常に定数
            let aligned_stop = (stop_val / tile_size) * tile_size;
            let main_stop = const_int(aligned_stop as isize);

            // 外側ループ: for outer_var in start..main_stop step tile_size
            let outer_loop = range(
                outer_var.clone(),
                start.as_ref().clone(),
                const_int(tile_size as isize),
                main_stop.clone(),
                inner_loop,
            );

            // 端数が0の場合（割り切れる場合）は端数ループを生成しない
            let remainder_count = stop_val - aligned_stop;
            if remainder_count == 0 {
                log::trace!(
                    "No remainder loop needed for {} (stop_val {} is divisible by tile_size {})",
                    var,
                    stop_val,
                    tile_size
                );
                // メインループのみを返す
                Some(outer_loop)
            } else {
                // 端数処理ループ: for remainder_var in main_stop..stop step 1
                // ループ本体の中で元の変数名を使うため、代入を追加
                let remainder_assign = assign(var.clone(), helper_var(remainder_var.clone()));

                // 端数ループでも元のループ変数をスコープに宣言
                let mut remainder_scope = Scope::new();
                remainder_scope
                    .declare(var.clone(), DType::Int, Mutability::Mutable)
                    .expect("Failed to declare loop variable in remainder scope");

                let remainder_body = AstNode::Block {
                    statements: vec![remainder_assign, body.as_ref().clone()],
                    scope: Box::new(remainder_scope),
                };
                let remainder_loop = range(
                    remainder_var,
                    main_stop,
                    step.as_ref().clone(),
                    stop.as_ref().clone(),
                    remainder_body,
                );

                // メインループと端数処理ループを含むBlock
                Some(AstNode::Block {
                    statements: vec![outer_loop, remainder_loop],
                    scope: Box::new(Scope::new()),
                })
            }
        }
        _ => None, // Rangeノードでない場合は変換不可
    }
}

/// タイル化されたループの本体かどうかを判定する
///
/// タイル化後のループ本体は、最初の文が変数への代入文（`var = expr`）になっている。
/// この構造を持つループは既にタイル化済みとみなし、再タイル化を防ぐ。
fn is_tiled_loop_body(body: &AstNode) -> bool {
    match body {
        AstNode::Block { statements, .. } => {
            // 本体の最初の文がAssignノードの場合、タイル化済み
            if let Some(first_stmt) = statements.first() {
                matches!(first_stmt, AstNode::Assign { .. })
            } else {
                false
            }
        }
        _ => false,
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
        AstNode::BitwiseAnd(a, b) => AstNode::BitwiseAnd(
            Box::new(replace_var_in_ast(a, var_name, replacement)),
            Box::new(replace_var_in_ast(b, var_name, replacement)),
        ),
        AstNode::BitwiseOr(a, b) => AstNode::BitwiseOr(
            Box::new(replace_var_in_ast(a, var_name, replacement)),
            Box::new(replace_var_in_ast(b, var_name, replacement)),
        ),
        AstNode::BitwiseXor(a, b) => AstNode::BitwiseXor(
            Box::new(replace_var_in_ast(a, var_name, replacement)),
            Box::new(replace_var_in_ast(b, var_name, replacement)),
        ),
        AstNode::BitwiseNot(a) => {
            AstNode::BitwiseNot(Box::new(replace_var_in_ast(a, var_name, replacement)))
        }
        AstNode::LeftShift(a, b) => AstNode::LeftShift(
            Box::new(replace_var_in_ast(a, var_name, replacement)),
            Box::new(replace_var_in_ast(b, var_name, replacement)),
        ),
        AstNode::RightShift(a, b) => AstNode::RightShift(
            Box::new(replace_var_in_ast(a, var_name, replacement)),
            Box::new(replace_var_in_ast(b, var_name, replacement)),
        ),
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
        // 16は4で割り切れるので、端数ループは生成されない
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

        // 16は4で割り切れるので、メインループ（外側ループ）のみが返される
        if let Some(AstNode::Range {
            var,
            step,
            body: outer_body,
            ..
        }) = tiled
        {
            // 変数名はridxN形式（既存変数と衝突しない連番）
            assert!(var.starts_with("ridx"), "Expected ridxN, got {}", var);
            assert_eq!(*step, AstNode::Const(Literal::Int(4))); // tile_size

            // 内側ループが存在するか確認
            assert!(matches!(outer_body.as_ref(), AstNode::Range { .. }));
        } else {
            panic!("Expected Range node (no remainder loop for divisible case)");
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

    #[test]
    fn test_tile_loop_declares_original_variable() {
        // 元のループ変数がタイル化後のスコープに宣言されていることを確認
        // for ridx2 in 0..1024 step 1 { body }
        // 1024は128で割り切れるので端数ループは生成されない
        let body = Box::new(AstNode::Var("acc".to_string()));

        let original_loop = AstNode::Range {
            var: "ridx2".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(1024))),
            body,
        };

        let tiled = tile_loop(&original_loop, 128);
        assert!(tiled.is_some());

        // 1024は128で割り切れるので、メインループ（外側ループ）のみが返される
        if let Some(AstNode::Range {
            body: outer_body, ..
        }) = tiled
        {
            // 内側ループを取得
            if let AstNode::Range {
                body: inner_body, ..
            } = outer_body.as_ref()
            {
                // 内側ループの本体（Block）を取得
                if let AstNode::Block { scope, .. } = inner_body.as_ref() {
                    // スコープに ridx2 が宣言されていることを確認
                    let ridx2_decl = scope.get("ridx2");
                    assert!(
                        ridx2_decl.is_some(),
                        "Original loop variable 'ridx2' should be declared in inner body scope"
                    );
                    assert_eq!(ridx2_decl.unwrap().dtype, DType::Int);
                } else {
                    panic!("Expected Block node for inner body");
                }
            } else {
                panic!("Expected Range node for inner loop");
            }
        } else {
            panic!("Expected Range node (no remainder for divisible case)");
        }
    }

    #[test]
    fn test_tile_loop_declares_original_variable_with_remainder() {
        // 端数がある場合に端数ループでも元変数が宣言されることを確認
        // for ridx2 in 0..1000 step 1 { body } (1000 = 128*7 + 104, 端数あり)
        let body = Box::new(AstNode::Var("acc".to_string()));

        let original_loop = AstNode::Range {
            var: "ridx2".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(1000))),
            body,
        };

        let tiled = tile_loop(&original_loop, 128);
        assert!(tiled.is_some());

        // 1000は128で割り切れないので、Blockノード（メインループ + 端数ループ）が返される
        if let Some(AstNode::Block { statements, .. }) = tiled {
            assert_eq!(statements.len(), 2);

            // 端数ループを確認
            if let AstNode::Range {
                body: remainder_body,
                ..
            } = &statements[1]
            {
                if let AstNode::Block { scope, .. } = remainder_body.as_ref() {
                    let ridx2_decl = scope.get("ridx2");
                    assert!(
                        ridx2_decl.is_some(),
                        "Original loop variable 'ridx2' should be declared in remainder body scope"
                    );
                } else {
                    panic!("Expected Block node for remainder body");
                }
            } else {
                panic!("Expected Range node for remainder loop");
            }
        } else {
            panic!("Expected Block node for non-divisible case");
        }
    }
}
