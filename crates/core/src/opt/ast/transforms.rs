//! AST変換関数
//!
//! ループのタイル化やループ展開などの複雑なAST変換を提供します。

use crate::ast::{
    AstNode, DType, Literal, Mutability, Scope,
    helper::{assign, const_int, empty_block, idiv, range, var as helper_var},
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
        | AstNode::RightShift(a, b)
        | AstNode::Lt(a, b)
        | AstNode::Le(a, b)
        | AstNode::Gt(a, b)
        | AstNode::Ge(a, b)
        | AstNode::Eq(a, b)
        | AstNode::Ne(a, b) => {
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
        AstNode::Fma { a, b, c } => {
            collect_var_names_recursive(a, names);
            collect_var_names_recursive(b, names);
            collect_var_names_recursive(c, names);
        }
        AstNode::AtomicAdd {
            ptr, offset, value, ..
        }
        | AstNode::AtomicMax {
            ptr, offset, value, ..
        } => {
            collect_var_names_recursive(ptr, names);
            collect_var_names_recursive(offset, names);
            collect_var_names_recursive(value, names);
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
        AstNode::If {
            condition,
            then_body,
            else_body,
        } => {
            collect_var_names_recursive(condition, names);
            collect_var_names_recursive(then_body, names);
            if let Some(else_b) = else_body {
                collect_var_names_recursive(else_b, names);
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
            // Note: タイル化されたループの内側ループも展開対象にする
            // これにより、タイル化 + 内側ループ展開でアンロールと同等の効果が得られる

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
///
/// # 備考
/// - stopが変数の場合でもタイル化可能
/// - aligned_stop = (stop / tile_size) * tile_size は計算式ノードとして生成される
/// - 端数処理ループは常に生成される（定数畳み込み最適化で後から削除される可能性あり）
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
            // ステップが1の場合のみタイル化可能（簡易実装）
            let is_step_one = matches!(step.as_ref(), AstNode::Const(Literal::Int(1)));
            if !is_step_one {
                log::trace!("Skipping tiling for loop with step != 1: {}", var);
                return None;
            }

            log::debug!("Tiling loop: {} with tile_size: {}", var, tile_size);

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

            // メインループの終了値: (stop / tile_size) * tile_size
            // 計算式ノードとして生成（最適化で定数畳み込みされる可能性あり）
            let tile_size_node = const_int(tile_size as isize);
            let aligned_stop = idiv(stop.as_ref().clone(), tile_size_node.clone()) * tile_size_node;

            // 外側ループ: for outer_var in start..aligned_stop step tile_size
            let outer_loop = range(
                outer_var.clone(),
                start.as_ref().clone(),
                const_int(tile_size as isize),
                aligned_stop.clone(),
                inner_loop,
            );

            // 端数処理ループ: for remainder_var in aligned_stop..stop step 1
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
                aligned_stop,
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
        _ => None, // Rangeノードでない場合は変換不可
    }
}

/// ループをガード付きでタイル化する
///
/// 端数処理を別ループではなく、内側ループ内のif文で行う方式です。
/// GPUでは境界チェックのオーバーヘッドが小さいため、この方式が有効な場合があります。
///
/// # 引数
/// * `loop_node` - Rangeノード
/// * `tile_size` - タイルサイズ
///
/// # 変換例
/// ```text
/// // 元のループ
/// for ridx0 in start..stop step 1 {
///   body(ridx0)
/// }
///
/// // タイル化後（ガード方式）
/// for ridx1 in 0..ceil_div(stop - start, tile_size) step 1 {
///   for ridx2 in 0..tile_size step 1 {
///     ridx0 = start + ridx1 * tile_size + ridx2
///     if (ridx0 < stop) {
///       body(ridx0)
///     }
///   }
/// }
/// ```
pub fn tile_loop_with_guard(loop_node: &AstNode, tile_size: usize) -> Option<AstNode> {
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
            // ステップが1の場合のみタイル化可能（簡易実装）
            let is_step_one = matches!(step.as_ref(), AstNode::Const(Literal::Int(1)));
            if !is_step_one {
                log::trace!(
                    "Skipping tiling with guard for loop with step != 1: {}",
                    var
                );
                return None;
            }

            // startが0であることを要求（デフォルト）
            // start != 0 の場合は tile_loop_with_guard_any_start を使用
            let is_start_zero = matches!(start.as_ref(), AstNode::Const(Literal::Int(0)));
            if !is_start_zero {
                log::trace!(
                    "Skipping tiling with guard for loop with start != 0: {}",
                    var
                );
                return None;
            }

            // stopの値を取得（定数の場合のみ最適化に使用）
            let stop_val = match stop.as_ref() {
                AstNode::Const(Literal::Int(v)) => Some(*v),
                _ => None,
            };

            log::debug!(
                "Tiling loop with guard: {} with tile_size: {}, stop: {:?}",
                var,
                tile_size,
                stop_val
            );

            // 既存変数名を収集し、衝突しない連番を取得
            let used_names = collect_var_names(loop_node);
            let (outer_var, inner_var, _) = find_next_ridx_names(&used_names);

            // 内側ループの本体: original_var = outer_var * tile_size + inner_var
            let i_expr = helper_var(outer_var.clone()) * const_int(tile_size as isize)
                + helper_var(inner_var.clone());

            // original_var = outer_var * tile_size + inner_var の代入
            let assign_i = assign(var.clone(), i_expr);

            // 境界チェック: if (original_var < stop) { body }
            let guard_condition = AstNode::Lt(
                Box::new(helper_var(var.clone())),
                Box::new(stop.as_ref().clone()),
            );

            let guarded_body = AstNode::If {
                condition: Box::new(guard_condition),
                then_body: Box::new(body.as_ref().clone()),
                else_body: None,
            };

            let inner_body_statements = vec![assign_i, guarded_body];

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

            // 外側ループの終了値: ceil_div(stop, tile_size)
            let outer_stop = if let Some(n) = stop_val {
                // 定数の場合: (n + tile_size - 1) / tile_size
                const_int((n as usize).div_ceil(tile_size) as isize)
            } else {
                // 変数の場合: (stop + tile_size - 1) / tile_size
                ceil_div_ast(stop.as_ref().clone(), const_int(tile_size as isize))
            };

            // 外側ループ: for outer_var in 0..ceil_div(stop, tile_size) step 1
            let outer_loop = range(
                outer_var.clone(),
                const_int(0),
                const_int(1),
                outer_stop,
                inner_loop,
            );

            Some(outer_loop)
        }
        _ => None, // Rangeノードでない場合は変換不可
    }
}

/// ceil_div(a, b)を計算するAstNodeを生成
/// (a + b - 1) / b と等価
fn ceil_div_ast(a: AstNode, b: AstNode) -> AstNode {
    // (a + b - 1) / b
    AstNode::Idiv(
        Box::new(AstNode::Add(
            Box::new(a),
            Box::new(AstNode::Add(Box::new(b.clone()), Box::new(const_int(-1)))),
        )),
        Box::new(b),
    )
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
        // 常にメインループ + 端数ループのBlockが返される
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

        // メインループ + 端数ループを含むBlockが返される
        if let Some(AstNode::Block { statements, .. }) = tiled {
            assert_eq!(statements.len(), 2);

            // メインループ（外側ループ）を確認
            if let AstNode::Range {
                var,
                step,
                body: outer_body,
                ..
            } = &statements[0]
            {
                // 変数名はridxN形式（既存変数と衝突しない連番）
                assert!(var.starts_with("ridx"), "Expected ridxN, got {}", var);
                assert_eq!(**step, AstNode::Const(Literal::Int(4))); // tile_size

                // 内側ループが存在するか確認
                assert!(matches!(outer_body.as_ref(), AstNode::Range { .. }));
            } else {
                panic!("Expected Range node for main loop");
            }

            // 端数ループが存在することを確認
            assert!(matches!(statements[1], AstNode::Range { .. }));
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

    #[test]
    fn test_tile_loop_declares_original_variable() {
        // 元のループ変数がタイル化後のスコープに宣言されていることを確認
        // for ridx2 in 0..1024 step 1 { body }
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

        // メインループ + 端数ループを含むBlockが返される
        if let Some(AstNode::Block { statements, .. }) = tiled {
            assert_eq!(statements.len(), 2);

            // メインループ（外側ループ）を確認
            if let AstNode::Range {
                body: outer_body, ..
            } = &statements[0]
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
                panic!("Expected Range node for main loop");
            }
        } else {
            panic!("Expected Block node");
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

    #[test]
    fn test_tile_loop_variable_stop() {
        // for i in 0..N step 1 { body } (Nは変数)
        // stopが変数でもタイル化可能なことを確認
        let body = Box::new(AstNode::Var("x".to_string()));

        let original_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Var("N".to_string())), // 変数
            body,
        };

        let tiled = tile_loop(&original_loop, 4);
        assert!(
            tiled.is_some(),
            "Should be able to tile loop with variable stop"
        );

        // メインループ + 端数ループを含むBlockが返される
        if let Some(AstNode::Block { statements, .. }) = tiled {
            assert_eq!(statements.len(), 2);

            // メインループを確認
            if let AstNode::Range { stop, step, .. } = &statements[0] {
                // stepはタイルサイズ
                assert_eq!(**step, AstNode::Const(Literal::Int(4)));

                // stopは (N / 4) * 4 の計算式ノード
                // Mul(Idiv(Var("N"), Const(4)), Const(4))
                if let AstNode::Mul(lhs, rhs) = stop.as_ref() {
                    assert!(
                        matches!(lhs.as_ref(), AstNode::Idiv(_, _)),
                        "Expected Idiv node in stop expression"
                    );
                    assert_eq!(**rhs, AstNode::Const(Literal::Int(4)));
                } else {
                    panic!("Expected Mul node for aligned_stop, got {:?}", stop);
                }
            } else {
                panic!("Expected Range node for main loop");
            }

            // 端数ループを確認
            if let AstNode::Range {
                start: rem_start,
                stop: rem_stop,
                ..
            } = &statements[1]
            {
                // startはaligned_stop (計算式ノード)
                assert!(
                    matches!(rem_start.as_ref(), AstNode::Mul(_, _)),
                    "Expected Mul node for remainder start"
                );
                // stopは元のN
                assert_eq!(**rem_stop, AstNode::Var("N".to_string()));
            } else {
                panic!("Expected Range node for remainder loop");
            }
        } else {
            panic!("Expected Block node");
        }
    }

    #[test]
    fn test_tile_loop_with_guard_simple() {
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

        let tiled = tile_loop_with_guard(&original_loop, 4);
        assert!(tiled.is_some());

        // ガード方式では単一の外側ループが返される（端数ループはない）
        if let Some(AstNode::Range {
            var,
            stop,
            body: outer_body,
            ..
        }) = tiled
        {
            // 変数名はridxN形式
            assert!(var.starts_with("ridx"), "Expected ridxN, got {}", var);
            // 外側ループの終了値: ceil_div(16, 4) = 4
            assert_eq!(*stop, AstNode::Const(Literal::Int(4)));

            // 内側ループが存在するか確認
            assert!(matches!(outer_body.as_ref(), AstNode::Range { .. }));

            // 内側ループの本体にIf文が含まれるか確認
            if let AstNode::Range {
                body: inner_body, ..
            } = outer_body.as_ref()
            {
                if let AstNode::Block { statements, .. } = inner_body.as_ref() {
                    // 2つ目の文がIf文（ガード）であること
                    assert!(statements.len() >= 2);
                    assert!(
                        matches!(&statements[1], AstNode::If { .. }),
                        "Expected If guard in inner body"
                    );
                } else {
                    panic!("Expected Block node for inner body");
                }
            }
        } else {
            panic!("Expected Range node");
        }
    }

    #[test]
    fn test_tile_loop_with_guard_remainder() {
        // for i in 0..10 step 1 { body } (10 = 4*2 + 2, 端数あり)
        let body = Box::new(AstNode::Var("x".to_string()));

        let original_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(10))),
            body,
        };

        let tiled = tile_loop_with_guard(&original_loop, 4);
        assert!(tiled.is_some());

        // ガード方式では、端数があっても単一の外側ループ
        // 外側ループの終了値: ceil_div(10, 4) = 3
        if let Some(AstNode::Range { stop, .. }) = tiled {
            assert_eq!(*stop, AstNode::Const(Literal::Int(3)));
        } else {
            panic!("Expected Range node");
        }
    }

    #[test]
    fn test_tile_loop_with_guard_variable_stop() {
        // for i in 0..N step 1 { body } (Nは変数)
        let body = Box::new(AstNode::Var("x".to_string()));

        let original_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Var("N".to_string())),
            body,
        };

        let tiled = tile_loop_with_guard(&original_loop, 4);
        assert!(tiled.is_some());

        // 変数stopでもタイル化できる（ガード方式の利点）
        if let Some(AstNode::Range { stop, .. }) = tiled {
            // 外側ループの終了値: ceil_div(N, 4) = (N + 3) / 4
            assert!(matches!(*stop, AstNode::Idiv(_, _)));
        } else {
            panic!("Expected Range node");
        }
    }

    #[test]
    fn test_tile_loop_with_guard_size_one() {
        let body = Box::new(AstNode::Var("x".to_string()));

        let original_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(16))),
            body,
        };

        let tiled = tile_loop_with_guard(&original_loop, 1);
        assert!(tiled.is_none()); // tile_size=1ではタイル化しない
    }
}
