//! AST書き換えルール集
//!
//! このモジュールは、ASTノードに対する標準的な代数的変形ルールを提供します。
//! これらのルールは、式の簡約や正規化、最適化に使用できます。

pub mod algebraic;
pub mod bit_ops;
pub mod const_folding;
mod macros;

use crate::ast::pat::AstRewriteRule;
use std::rc::Rc;

// 各モジュールから主要な関数を再エクスポート
pub use algebraic::{
    // 結合則
    add_associate_left_to_right,
    add_associate_right_to_left,
    // 交換則
    add_commutative,
    // 同項規則
    add_same_to_mul_two,
    // 単位元
    add_zero_left,
    add_zero_right,
    // 分配則
    distributive_left,
    distributive_right,
    // 逆演算
    exp2_log2,
    factor_left,
    factor_right,
    log2_exp2,
    max_commutative,
    // 冪等則
    max_idempotent,
    mul_associate_left_to_right,
    mul_associate_right_to_left,
    mul_commutative,
    mul_one_left,
    mul_one_right,
    // 零元
    mul_zero_left,
    mul_zero_right,
    recip_recip,
    sqrt_squared,
    // Block簡約
    unwrap_single_statement_block,
};

pub use bit_ops::{mul_power_of_two_to_shift_left, mul_power_of_two_to_shift_right};

pub use const_folding::{
    const_fold_add, const_fold_exp2, const_fold_idiv, const_fold_log2, const_fold_max,
    const_fold_mul, const_fold_recip, const_fold_rem, const_fold_sin, const_fold_sqrt,
    constant_folding_rules,
};

/// 簡約ルール集（式を簡単にする）
pub fn simplification_rules() -> Vec<Rc<AstRewriteRule>> {
    vec![
        // 単位元
        add_zero_right(),
        add_zero_left(),
        mul_one_right(),
        mul_one_left(),
        // 零元
        mul_zero_right(),
        mul_zero_left(),
        // 冪等則
        max_idempotent(),
        // 逆演算
        recip_recip(),
        sqrt_squared(),
        log2_exp2(),
        exp2_log2(),
        // 同項規則
        add_same_to_mul_two(),
        // Block簡約
        unwrap_single_statement_block(),
        // ビット演算最適化（2の累乗の乗算をシフトに変換）
        mul_power_of_two_to_shift_right(),
        mul_power_of_two_to_shift_left(),
    ]
}

/// 正規化ルール集（式を標準形に変換する）
pub fn normalization_rules() -> Vec<Rc<AstRewriteRule>> {
    vec![
        // 結合則（右結合に統一）
        add_associate_left_to_right(),
        mul_associate_left_to_right(),
    ]
}

/// すべての代数的ルール集（定数畳み込み含む）
pub fn all_algebraic_rules() -> Vec<Rc<AstRewriteRule>> {
    let mut rules = Vec::new();
    rules.extend(constant_folding_rules());
    rules.extend(simplification_rules());
    rules.extend(normalization_rules());
    // 交換則は探索用なのでデフォルトには含めない（無限ループの可能性）
    rules
}

/// 探索用の完全なルール集（交換則・分配則を含む）
///
/// ビームサーチなどの探索ベース最適化で使用することを想定しています。
/// RuleBaseOptimizerで直接使うと無限ループする可能性があるため注意してください。
pub fn all_rules_with_search() -> Vec<Rc<AstRewriteRule>> {
    let mut rules = Vec::new();
    rules.extend(constant_folding_rules());
    rules.extend(simplification_rules());
    rules.extend(normalization_rules());
    // 交換則
    rules.push(add_commutative());
    rules.push(mul_commutative());
    rules.push(max_commutative());
    // 分配則を追加
    rules.push(distributive_left());
    rules.push(distributive_right());
    rules.push(factor_left());
    rules.push(factor_right());
    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::{const_f32, const_int, exp2, idiv, log2, max, recip, rem, sqrt, var};
    use crate::ast::{AstNode, Scope};
    use crate::opt::ast::Optimizer;
    use crate::opt::ast::RuleBaseOptimizer;

    #[test]
    fn test_add_zero() {
        let rule = add_zero_right();
        let input = const_int(42) + const_int(0);
        let result = rule.apply(&input);
        assert_eq!(result, const_int(42));
    }

    #[test]
    fn test_mul_one() {
        let rule = mul_one_right();
        let input = const_int(42) * const_int(1);
        let result = rule.apply(&input);
        assert_eq!(result, const_int(42));
    }

    #[test]
    fn test_mul_zero() {
        let rule = mul_zero_right();
        let input = const_int(42) * const_int(0);
        let result = rule.apply(&input);
        assert_eq!(result, const_int(0));
    }

    #[test]
    fn test_max_idempotent() {
        let rule = max_idempotent();
        let var_a = var("a");
        let input = max(var_a.clone(), var_a.clone());
        let result = rule.apply(&input);
        assert_eq!(result, var_a);
    }

    #[test]
    fn test_recip_recip() {
        let rule = recip_recip();
        let var_a = var("a");
        let input = recip(recip(var_a.clone()));
        let result = rule.apply(&input);
        assert_eq!(result, var_a);
    }

    #[test]
    fn test_sqrt_squared() {
        let rule = sqrt_squared();
        let var_a = var("a");
        let input = sqrt(var_a.clone()) * sqrt(var_a.clone());
        let result = rule.apply(&input);
        assert_eq!(result, var_a);
    }

    #[test]
    fn test_distributive_left() {
        let rule = distributive_left();
        // a * (b + c)
        let input = var("a") * (var("b") + var("c"));
        let result = rule.apply(&input);
        // a * b + a * c
        let expected = var("a") * var("b") + var("a") * var("c");
        assert_eq!(result, expected);
    }

    #[test]
    fn test_factor_left() {
        let rule = factor_left();
        // a * b + a * c
        let input = var("a") * var("b") + var("a") * var("c");
        let result = rule.apply(&input);
        // a * (b + c)
        let expected = var("a") * (var("b") + var("c"));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simplification_rules() {
        let optimizer = RuleBaseOptimizer::new(simplification_rules());

        // (42 + 0) * 1
        let input = (const_int(42) + const_int(0)) * const_int(1);

        let result = optimizer.optimize(input);
        // 42
        assert_eq!(result, const_int(42));
    }

    #[test]
    fn test_associative_rules() {
        let rule = add_associate_left_to_right();
        // (a + b) + c
        let input = (var("a") + var("b")) + var("c");
        let result = rule.apply(&input);
        // a + (b + c)
        let expected = var("a") + (var("b") + var("c"));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_const_fold_add() {
        let rule = const_fold_add();

        // Isize: 2 + 3 = 5
        let input = const_int(2) + const_int(3);
        let result = rule.apply(&input);
        assert_eq!(result, const_int(5));

        // F32: 1.5 + 2.5 = 4.0
        let input = const_f32(1.5) + const_f32(2.5);
        let result = rule.apply(&input);
        assert_eq!(result, const_f32(4.0));
    }

    #[test]
    fn test_const_fold_mul() {
        let rule = const_fold_mul();

        // Isize: 6 * 7 = 42
        let input = const_int(6) * const_int(7);
        let result = rule.apply(&input);
        assert_eq!(result, const_int(42));

        // F32: 2.5 * 4.0 = 10.0
        let input = const_f32(2.5) * const_f32(4.0);
        let result = rule.apply(&input);
        assert_eq!(result, const_f32(10.0));
    }

    #[test]
    fn test_const_fold_sqrt() {
        let rule = const_fold_sqrt();

        // sqrt(4.0) = 2.0
        let input = sqrt(const_f32(4.0));
        let result = rule.apply(&input);
        assert_eq!(result, const_f32(2.0));

        // sqrt(9.0) = 3.0
        let input = sqrt(const_f32(9.0));
        let result = rule.apply(&input);
        assert_eq!(result, const_f32(3.0));
    }

    #[test]
    fn test_const_fold_recip() {
        let rule = const_fold_recip();

        // recip(2.0) = 0.5
        let input = recip(const_f32(2.0));
        let result = rule.apply(&input);
        assert_eq!(result, const_f32(0.5));

        // recip(4.0) = 0.25
        let input = recip(const_f32(4.0));
        let result = rule.apply(&input);
        assert_eq!(result, const_f32(0.25));
    }

    #[test]
    fn test_const_fold_max() {
        let rule = const_fold_max();

        // max(3, 5) = 5
        let input = max(const_int(3), const_int(5));
        let result = rule.apply(&input);
        assert_eq!(result, const_int(5));

        // max(2.5, 1.5) = 2.5
        let input = max(const_f32(2.5), const_f32(1.5));
        let result = rule.apply(&input);
        assert_eq!(result, const_f32(2.5));
    }

    #[test]
    fn test_const_fold_rem() {
        let rule = const_fold_rem();

        // 10 % 3 = 1
        let input = rem(const_int(10), const_int(3));
        let result = rule.apply(&input);
        assert_eq!(result, const_int(1));
    }

    #[test]
    fn test_const_fold_idiv() {
        let rule = const_fold_idiv();

        // 10 / 3 = 3
        let input = idiv(const_int(10), const_int(3));
        let result = rule.apply(&input);
        assert_eq!(result, const_int(3));
    }

    #[test]
    fn test_constant_folding_with_optimizer() {
        let optimizer = RuleBaseOptimizer::new(constant_folding_rules());

        // (2 + 3) * 4 = 5 * 4 = 20
        let input = (const_int(2) + const_int(3)) * const_int(4);

        let result = optimizer.optimize(input);
        assert_eq!(result, const_int(20));
    }

    #[test]
    fn test_combined_optimization() {
        // 定数畳み込みと簡約を組み合わせ
        let optimizer = RuleBaseOptimizer::new(all_algebraic_rules());

        // ((2 + 3) * 1) + 0 = 5 * 1 + 0 = 5 + 0 = 5
        let input = ((const_int(2) + const_int(3)) * const_int(1)) + const_int(0);

        let result = optimizer.optimize(input);
        assert_eq!(result, const_int(5));
    }

    #[test]
    fn test_mul_power_of_two_to_shift_right() {
        // x * 4 → x << 2
        let rule = mul_power_of_two_to_shift_right();
        let input = var("x") * const_int(4);
        let result = rule.apply(&input);

        match result {
            AstNode::LeftShift(left, right) => {
                assert_eq!(*left, var("x"));
                assert_eq!(*right, const_int(2));
            }
            _ => panic!("Expected LeftShift node"),
        }
    }

    #[test]
    fn test_mul_power_of_two_to_shift_left() {
        // 8 * x → x << 3
        let rule = mul_power_of_two_to_shift_left();
        let input = const_int(8) * var("x");
        let result = rule.apply(&input);

        match result {
            AstNode::LeftShift(left, right) => {
                assert_eq!(*left, var("x"));
                assert_eq!(*right, const_int(3));
            }
            _ => panic!("Expected LeftShift node"),
        }
    }

    #[test]
    fn test_mul_non_power_of_two() {
        // 非2の累乗の場合は変換されない（元のノードが返る）
        let rule = mul_power_of_two_to_shift_right();
        let input = var("x") * const_int(5);
        let result = rule.apply(&input);

        // 元のMulノードが返ってくるはず
        match result {
            AstNode::Mul(left, right) => {
                assert_eq!(*left, var("x"));
                assert_eq!(*right, const_int(5));
            }
            _ => panic!("Expected Mul node (unchanged)"),
        }
    }

    #[test]
    fn test_mul_various_powers_of_two() {
        // さまざまな2の累乗をテスト
        let test_cases = vec![
            (1, 0),   // 1 = 2^0
            (2, 1),   // 2 = 2^1
            (4, 2),   // 4 = 2^2
            (8, 3),   // 8 = 2^3
            (16, 4),  // 16 = 2^4
            (32, 5),  // 32 = 2^5
            (64, 6),  // 64 = 2^6
            (128, 7), // 128 = 2^7
            (256, 8), // 256 = 2^8
        ];

        let rule = mul_power_of_two_to_shift_right();
        for (power_of_two, expected_shift) in test_cases {
            let input = var("x") * const_int(power_of_two);
            let result = rule.apply(&input);

            match result {
                AstNode::LeftShift(_, right) => {
                    assert_eq!(*right, const_int(expected_shift));
                }
                _ => panic!("Expected LeftShift node for {}", power_of_two),
            }
        }
    }

    #[test]
    fn test_mul_power_of_two_with_optimizer() {
        // オプティマイザーと組み合わせてテスト
        let optimizer = RuleBaseOptimizer::new(simplification_rules());

        // x * 16 → x << 4
        let input = var("x") * const_int(16);

        let result = optimizer.optimize(input);

        match result {
            AstNode::LeftShift(left, right) => {
                assert_eq!(*left, var("x"));
                assert_eq!(*right, const_int(4));
            }
            _ => panic!("Expected LeftShift node"),
        }
    }

    #[test]
    fn test_log2_exp2() {
        let rule = log2_exp2();
        let var_a = var("a");
        let input = log2(exp2(var_a.clone()));
        let result = rule.apply(&input);
        assert_eq!(result, var_a);
    }

    #[test]
    fn test_exp2_log2() {
        let rule = exp2_log2();
        let var_a = var("a");
        let input = exp2(log2(var_a.clone()));
        let result = rule.apply(&input);
        assert_eq!(result, var_a);
    }

    #[test]
    fn test_add_same_to_mul_two() {
        let rule = add_same_to_mul_two();
        let var_x = var("x");
        // x + x
        let input = var_x.clone() + var_x.clone();
        let result = rule.apply(&input);
        // x * 2
        let expected = var_x * const_int(2);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_add_same_to_mul_two_with_const() {
        let rule = add_same_to_mul_two();
        // 5 + 5
        let const_5 = const_int(5);
        let input = const_5.clone() + const_5.clone();
        let result = rule.apply(&input);
        // 5 * 2
        let expected = const_5 * const_int(2);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_unwrap_single_statement_block() {
        use crate::ast::helper::block;
        let rule = unwrap_single_statement_block();

        // 単一のステートメントを持つBlockは展開される
        let single_stmt = var("x");
        let blk = block(vec![single_stmt.clone()], Scope::new());
        let result = rule.apply(&blk);
        assert_eq!(result, single_stmt);
    }

    #[test]
    fn test_unwrap_single_statement_block_multiple_statements() {
        use crate::ast::helper::block;
        let rule = unwrap_single_statement_block();

        // 複数のステートメントを持つBlockは展開されない
        let multi_block = block(vec![var("x"), var("y")], Scope::new());
        let result = rule.apply(&multi_block);
        // Blockのまま変わらないはず
        match result {
            AstNode::Block { statements, .. } => {
                assert_eq!(statements.len(), 2);
            }
            _ => panic!("Expected Block node"),
        }
    }

    #[test]
    fn test_unwrap_single_statement_block_with_optimizer() {
        use crate::ast::helper::{block, range};
        let optimizer = RuleBaseOptimizer::new(simplification_rules());

        // Block内にRangeがあるケース（ループ交換などで生成されるパターン）
        let inner_range = range("i", const_int(0), const_int(1), const_int(10), var("body"));

        let blk = block(vec![inner_range.clone()], Scope::new());

        let result = optimizer.optimize(blk);

        // Blockが展開されてRangeが直接返されるはず
        match result {
            AstNode::Range { var: v, .. } => {
                assert_eq!(v, "i");
            }
            _ => panic!("Expected Range node after unwrapping"),
        }
    }
}
