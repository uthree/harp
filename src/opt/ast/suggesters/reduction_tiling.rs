//! 縮約ループのタイル化を提案するSuggester
//!
//! 縮約パターン（acc = acc op value）を検出し、
//! 部分和配列を用いたタイル化された縮約に変換します。
//!
//! # 変換例
//!
//! ```text
//! // 入力
//! acc = 0
//! for i in 0..N { acc = acc + array[i] }
//!
//! // 出力（タイルサイズ=4）
//! num_tiles = ceil_div(N, 4)
//! partial = Allocate(F32, num_tiles)
//! for t in 0..num_tiles { store(partial, t, 0.0) }
//! for tile in 0..num_tiles {
//!     for j in 0..4 {
//!         idx = tile * 4 + j
//!         if (idx < N) {
//!             store(partial, tile, load(partial, tile) + array[idx])
//!         }
//!     }
//! }
//! Barrier
//! acc = 0
//! for t in 0..num_tiles { acc = acc + load(partial, t) }
//! Deallocate(partial)
//! ```

use crate::ast::helper::{assign, barrier, block, const_int, if_then, load, range, store, var};
use crate::ast::scope::Scope;
use crate::ast::{AstNode, DType, Literal};
use crate::opt::ast::transforms::collect_var_names;
use crate::opt::ast::{AstSuggestResult, AstSuggester};
use log::{debug, trace};
use std::collections::HashSet;

/// 縮約演算の種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// 加算: acc = acc + value
    Add,
    /// 乗算: acc = acc * value
    Mul,
    /// 最大値: acc = max(acc, value)
    Max,
    /// 最小値: acc = min(acc, value) = -max(-acc, -value)
    Min,
}

impl ReductionOp {
    /// 演算の単位元を取得
    pub fn identity(&self, dtype: &DType) -> AstNode {
        match (self, dtype) {
            (ReductionOp::Add, DType::F32) => AstNode::Const(Literal::F32(0.0)),
            (ReductionOp::Add, DType::F64) => AstNode::Const(Literal::F64(0.0)),
            (ReductionOp::Add, _) => AstNode::Const(Literal::I64(0)),

            (ReductionOp::Mul, DType::F32) => AstNode::Const(Literal::F32(1.0)),
            (ReductionOp::Mul, DType::F64) => AstNode::Const(Literal::F64(1.0)),
            (ReductionOp::Mul, _) => AstNode::Const(Literal::I64(1)),

            (ReductionOp::Max, DType::F32) => AstNode::Const(Literal::F32(f32::NEG_INFINITY)),
            (ReductionOp::Max, DType::F64) => AstNode::Const(Literal::F64(f64::NEG_INFINITY)),
            (ReductionOp::Max, _) => AstNode::Const(Literal::I64(i64::MIN)),

            (ReductionOp::Min, DType::F32) => AstNode::Const(Literal::F32(f32::INFINITY)),
            (ReductionOp::Min, DType::F64) => AstNode::Const(Literal::F64(f64::INFINITY)),
            (ReductionOp::Min, _) => AstNode::Const(Literal::I64(i64::MAX)),
        }
    }

    /// 二項演算を適用したノードを生成
    pub fn apply(&self, acc: AstNode, value: AstNode) -> AstNode {
        match self {
            ReductionOp::Add => AstNode::Add(Box::new(acc), Box::new(value)),
            ReductionOp::Mul => AstNode::Mul(Box::new(acc), Box::new(value)),
            ReductionOp::Max => AstNode::Max(Box::new(acc), Box::new(value)),
            // Min: min(a, b) = -max(-a, -b)
            ReductionOp::Min => {
                let neg_one = AstNode::Const(Literal::F32(-1.0));
                let neg_acc = AstNode::Mul(Box::new(neg_one.clone()), Box::new(acc));
                let neg_value = AstNode::Mul(Box::new(neg_one.clone()), Box::new(value));
                let neg_max = AstNode::Max(Box::new(neg_acc), Box::new(neg_value));
                AstNode::Mul(Box::new(neg_one), Box::new(neg_max))
            }
        }
    }
}

/// 縮約パターンの情報
#[derive(Debug, Clone)]
pub struct ReductionPattern {
    /// アキュムレータ変数名
    pub accumulator: String,
    /// 縮約演算の種類
    pub op: ReductionOp,
    /// 縮約される値の式（アキュムレータを除いた部分）
    pub value_expr: AstNode,
    /// アキュムレータのデータ型
    pub dtype: DType,
}

/// 縮約ループのタイル化を提案するSuggester
pub struct ReductionTilingSuggester {
    /// 試行するタイルサイズのリスト
    tile_sizes: Vec<usize>,
}

impl Default for ReductionTilingSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl ReductionTilingSuggester {
    /// デフォルトのタイルサイズで新しいSuggesterを作成
    pub fn new() -> Self {
        Self {
            tile_sizes: vec![2, 4, 8, 16],
        }
    }

    /// 指定したタイルサイズで新しいSuggesterを作成
    pub fn with_sizes(tile_sizes: Vec<usize>) -> Self {
        Self { tile_sizes }
    }

    /// Assign文から縮約パターンを検出
    fn detect_reduction_pattern(assign: &AstNode) -> Option<ReductionPattern> {
        let AstNode::Assign {
            var: acc_var,
            value,
        } = assign
        else {
            return None;
        };

        // パターンマッチで演算と値を抽出
        match value.as_ref() {
            // acc = acc + expr
            AstNode::Add(left, right) => {
                if let AstNode::Var(name) = left.as_ref()
                    && name == acc_var
                {
                    return Some(ReductionPattern {
                        accumulator: acc_var.clone(),
                        op: ReductionOp::Add,
                        value_expr: *right.clone(),
                        dtype: right.infer_type(),
                    });
                }
                // acc = expr + acc
                if let AstNode::Var(name) = right.as_ref()
                    && name == acc_var
                {
                    return Some(ReductionPattern {
                        accumulator: acc_var.clone(),
                        op: ReductionOp::Add,
                        value_expr: *left.clone(),
                        dtype: left.infer_type(),
                    });
                }
            }
            // acc = acc * expr
            AstNode::Mul(left, right) => {
                if let AstNode::Var(name) = left.as_ref()
                    && name == acc_var
                {
                    return Some(ReductionPattern {
                        accumulator: acc_var.clone(),
                        op: ReductionOp::Mul,
                        value_expr: *right.clone(),
                        dtype: right.infer_type(),
                    });
                }
                // acc = expr * acc
                if let AstNode::Var(name) = right.as_ref()
                    && name == acc_var
                {
                    return Some(ReductionPattern {
                        accumulator: acc_var.clone(),
                        op: ReductionOp::Mul,
                        value_expr: *left.clone(),
                        dtype: left.infer_type(),
                    });
                }
            }
            // acc = max(acc, expr)
            AstNode::Max(left, right) => {
                if let AstNode::Var(name) = left.as_ref()
                    && name == acc_var
                {
                    return Some(ReductionPattern {
                        accumulator: acc_var.clone(),
                        op: ReductionOp::Max,
                        value_expr: *right.clone(),
                        dtype: right.infer_type(),
                    });
                }
                // acc = max(expr, acc)
                if let AstNode::Var(name) = right.as_ref()
                    && name == acc_var
                {
                    return Some(ReductionPattern {
                        accumulator: acc_var.clone(),
                        op: ReductionOp::Max,
                        value_expr: *left.clone(),
                        dtype: left.infer_type(),
                    });
                }
            }
            _ => {}
        }

        None
    }

    /// ループ本体から縮約パターンを抽出
    fn extract_reduction_from_body(body: &AstNode) -> Option<ReductionPattern> {
        match body {
            // 直接Assignの場合
            AstNode::Assign { .. } => Self::detect_reduction_pattern(body),
            // Block内の単一Assignの場合
            AstNode::Block { statements, .. } => {
                if statements.len() == 1 {
                    Self::detect_reduction_pattern(&statements[0])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// 一意の変数名を生成
    fn find_unique_name(used_names: &HashSet<String>, base: &str) -> String {
        if !used_names.contains(base) {
            return base.to_string();
        }
        for i in 0.. {
            let name = format!("{}_{}", base, i);
            if !used_names.contains(&name) {
                return name;
            }
        }
        unreachable!()
    }

    /// ループ変数を新しい式で置換
    fn substitute_loop_var(expr: &AstNode, loop_var: &str, replacement: &AstNode) -> AstNode {
        match expr {
            AstNode::Var(name) if name == loop_var => replacement.clone(),
            AstNode::Add(left, right) => AstNode::Add(
                Box::new(Self::substitute_loop_var(left, loop_var, replacement)),
                Box::new(Self::substitute_loop_var(right, loop_var, replacement)),
            ),
            AstNode::Mul(left, right) => AstNode::Mul(
                Box::new(Self::substitute_loop_var(left, loop_var, replacement)),
                Box::new(Self::substitute_loop_var(right, loop_var, replacement)),
            ),
            AstNode::Max(left, right) => AstNode::Max(
                Box::new(Self::substitute_loop_var(left, loop_var, replacement)),
                Box::new(Self::substitute_loop_var(right, loop_var, replacement)),
            ),
            AstNode::Load {
                ptr,
                offset,
                count,
                dtype,
            } => AstNode::Load {
                ptr: Box::new(Self::substitute_loop_var(ptr, loop_var, replacement)),
                offset: Box::new(Self::substitute_loop_var(offset, loop_var, replacement)),
                count: *count,
                dtype: dtype.clone(),
            },
            _ => expr.clone(),
        }
    }

    /// ceil_div(a, b) = (a + b - 1) / b を表すASTノードを生成
    fn ceil_div(a: AstNode, b: AstNode) -> AstNode {
        // (a + b - 1) / b
        let b_minus_1 = AstNode::Add(Box::new(b.clone()), Box::new(const_int(-1)));
        let numerator = AstNode::Add(Box::new(a), Box::new(b_minus_1));
        AstNode::Idiv(Box::new(numerator), Box::new(b))
    }

    /// 縮約ループをタイル化
    fn tile_reduction_loop(
        &self,
        range_node: &AstNode,
        pattern: &ReductionPattern,
        tile_size: usize,
    ) -> Option<AstNode> {
        let AstNode::Range {
            var: loop_var,
            start,
            step,
            stop,
            body: _,
        } = range_node
        else {
            return None;
        };

        // ステップが1でない場合はスキップ
        if !matches!(step.as_ref(), AstNode::Const(Literal::I64(1))) {
            return None;
        }

        // startが0でない場合は複雑になるのでスキップ
        if !matches!(start.as_ref(), AstNode::Const(Literal::I64(0))) {
            return None;
        }

        let used_names = collect_var_names(range_node);

        // 新しい変数名を生成
        let partial_var = Self::find_unique_name(&used_names, "partial");
        let num_tiles_var = Self::find_unique_name(&used_names, "num_tiles");
        let tile_var = Self::find_unique_name(&used_names, "tile");
        let inner_var = Self::find_unique_name(&used_names, "j");
        let idx_var = Self::find_unique_name(&used_names, "idx");
        let init_var = Self::find_unique_name(&used_names, "init_t");
        let final_var = Self::find_unique_name(&used_names, "final_t");

        let tile_size_node = const_int(tile_size as i64);

        // num_tiles = ceil_div(N, tile_size)
        let num_tiles_calc = Self::ceil_div(stop.as_ref().clone(), tile_size_node.clone());
        let num_tiles_assign = assign(&num_tiles_var, num_tiles_calc);

        // partial = Allocate(dtype, num_tiles)
        let partial_dtype = pattern.dtype.clone();
        let allocate = AstNode::Allocate {
            dtype: Box::new(partial_dtype.clone()),
            size: Box::new(var(&num_tiles_var)),
        };
        let partial_assign = assign(&partial_var, allocate);

        // 初期化ループ: for init_t in 0..num_tiles { store(partial, init_t, identity) }
        let identity = pattern.op.identity(&partial_dtype);
        let init_store = store(var(&partial_var), var(&init_var), identity);
        let init_loop = range(
            &init_var,
            const_int(0),
            const_int(1),
            var(&num_tiles_var),
            init_store,
        );

        // タイル化された縮約ループ
        // for tile in 0..num_tiles {
        //     for j in 0..tile_size {
        //         idx = tile * tile_size + j
        //         if (idx < N) {
        //             store(partial, tile, op(load(partial, tile), value_expr[idx]))
        //         }
        //     }
        // }
        let idx_calc = AstNode::Add(
            Box::new(AstNode::Mul(
                Box::new(var(&tile_var)),
                Box::new(tile_size_node.clone()),
            )),
            Box::new(var(&inner_var)),
        );
        let idx_assign = assign(&idx_var, idx_calc);

        // value_expr内のループ変数をidxで置換
        let substituted_value =
            Self::substitute_loop_var(&pattern.value_expr, loop_var, &var(&idx_var));

        // load(partial, tile)
        let partial_load = load(var(&partial_var), var(&tile_var), partial_dtype.clone());

        // op(load(partial, tile), substituted_value)
        let reduction_value = pattern.op.apply(partial_load, substituted_value);

        // store(partial, tile, reduction_value)
        let reduction_store = store(var(&partial_var), var(&tile_var), reduction_value);

        // if (idx < N) { reduction_store }
        let guard_cond = AstNode::Lt(Box::new(var(&idx_var)), Box::new(stop.as_ref().clone()));
        let guarded_reduction = if_then(guard_cond, reduction_store);

        // 内側ループのブロック
        let inner_block = block(vec![idx_assign, guarded_reduction], Scope::new());

        // 内側ループ: for j in 0..tile_size { ... }
        let inner_loop = range(
            &inner_var,
            const_int(0),
            const_int(1),
            tile_size_node,
            inner_block,
        );

        // 外側ループ: for tile in 0..num_tiles { inner_loop }
        let outer_loop = range(
            &tile_var,
            const_int(0),
            const_int(1),
            var(&num_tiles_var),
            inner_loop,
        );

        // バリアー
        let sync_barrier = barrier();

        // 最終縮約ループ
        // acc = identity
        // for final_t in 0..num_tiles {
        //     acc = op(acc, load(partial, final_t))
        // }
        let acc_init = assign(&pattern.accumulator, pattern.op.identity(&partial_dtype));

        let final_load = load(var(&partial_var), var(&final_var), partial_dtype.clone());
        let final_reduction = pattern.op.apply(var(&pattern.accumulator), final_load);
        let final_update = assign(&pattern.accumulator, final_reduction);
        let final_loop = range(
            &final_var,
            const_int(0),
            const_int(1),
            var(&num_tiles_var),
            final_update,
        );

        // Deallocate(partial)
        let deallocate = AstNode::Deallocate {
            ptr: Box::new(var(&partial_var)),
        };

        // 全体をBlockで囲む（スコープは空、変数は動的に割り当て）
        let result = AstNode::Block {
            statements: vec![
                num_tiles_assign,
                partial_assign,
                init_loop,
                outer_loop,
                sync_barrier,
                acc_init,
                final_loop,
                deallocate,
            ],
            scope: Box::new(Scope::new()),
        };

        Some(result)
    }

    /// AST内の縮約ループを探索してタイル化候補を収集
    fn collect_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        // 現在のノードがRangeで縮約パターンの場合
        if let AstNode::Range { body, .. } = ast
            && let Some(pattern) = Self::extract_reduction_from_body(body)
        {
            trace!(
                "Found reduction pattern: {} = {} op ...",
                pattern.accumulator,
                match pattern.op {
                    ReductionOp::Add => "acc +",
                    ReductionOp::Mul => "acc *",
                    ReductionOp::Max => "max(acc,",
                    ReductionOp::Min => "min(acc,",
                }
            );

            for &tile_size in &self.tile_sizes {
                if let Some(tiled) = self.tile_reduction_loop(ast, &pattern, tile_size) {
                    candidates.push(tiled);
                }
            }
        }

        // 子ノードを再帰的に探索
        match ast {
            AstNode::Block { statements, scope } => {
                for (i, stmt) in statements.iter().enumerate() {
                    for candidate in self.collect_candidates(stmt) {
                        let mut new_stmts = statements.clone();
                        new_stmts[i] = candidate;
                        candidates.push(AstNode::Block {
                            statements: new_stmts,
                            scope: scope.clone(),
                        });
                    }
                }
            }
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => {
                for candidate in self.collect_candidates(body) {
                    candidates.push(AstNode::Range {
                        var: var.clone(),
                        start: start.clone(),
                        step: step.clone(),
                        stop: stop.clone(),
                        body: Box::new(candidate),
                    });
                }
            }
            AstNode::Function {
                name,
                params,
                return_type,
                body,
            } => {
                for candidate in self.collect_candidates(body) {
                    candidates.push(AstNode::Function {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(candidate),
                    });
                }
            }
            AstNode::Kernel {
                name,
                params,
                return_type,
                body,
                default_grid_size,
                default_thread_group_size,
            } => {
                for candidate in self.collect_candidates(body) {
                    candidates.push(AstNode::Kernel {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(candidate),
                        default_grid_size: default_grid_size.clone(),
                        default_thread_group_size: default_thread_group_size.clone(),
                    });
                }
            }
            AstNode::Program {
                functions,
                execution_waves,
            } => {
                for (i, func) in functions.iter().enumerate() {
                    for candidate in self.collect_candidates(func) {
                        let mut new_functions = functions.clone();
                        new_functions[i] = candidate;
                        candidates.push(AstNode::Program {
                            functions: new_functions,
                            execution_waves: execution_waves.clone(),
                        });
                    }
                }
            }
            _ => {}
        }

        candidates
    }
}

impl AstSuggester for ReductionTilingSuggester {
    fn name(&self) -> &str {
        "ReductionTiling"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        trace!("ReductionTilingSuggester: Searching for reduction loops");
        let candidates = self.collect_candidates(ast);
        let suggestions = super::deduplicate_candidates(candidates);
        debug!(
            "ReductionTilingSuggester: Generated {} unique suggestions",
            suggestions.len()
        );

        suggestions
            .into_iter()
            .map(|ast| AstSuggestResult::with_description(ast, self.name(), "tile reduction loop"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::{assign, const_int, load, range, store, var};

    #[test]
    fn test_detect_add_reduction() {
        // acc = acc + x
        let assign_node = assign("acc", var("acc") + var("x"));
        let pattern = ReductionTilingSuggester::detect_reduction_pattern(&assign_node);

        assert!(pattern.is_some());
        let p = pattern.unwrap();
        assert_eq!(p.accumulator, "acc");
        assert!(matches!(p.op, ReductionOp::Add));
    }

    #[test]
    fn test_detect_add_reduction_reversed() {
        // acc = x + acc
        let assign_node = assign("acc", var("x") + var("acc"));
        let pattern = ReductionTilingSuggester::detect_reduction_pattern(&assign_node);

        assert!(pattern.is_some());
        let p = pattern.unwrap();
        assert_eq!(p.accumulator, "acc");
        assert!(matches!(p.op, ReductionOp::Add));
    }

    #[test]
    fn test_detect_mul_reduction() {
        // prod = prod * x
        let assign_node = assign("prod", var("prod") * var("x"));
        let pattern = ReductionTilingSuggester::detect_reduction_pattern(&assign_node);

        assert!(pattern.is_some());
        let p = pattern.unwrap();
        assert_eq!(p.accumulator, "prod");
        assert!(matches!(p.op, ReductionOp::Mul));
    }

    #[test]
    fn test_detect_max_reduction() {
        // m = max(m, x)
        let assign_node = assign("m", AstNode::Max(Box::new(var("m")), Box::new(var("x"))));
        let pattern = ReductionTilingSuggester::detect_reduction_pattern(&assign_node);

        assert!(pattern.is_some());
        let p = pattern.unwrap();
        assert_eq!(p.accumulator, "m");
        assert!(matches!(p.op, ReductionOp::Max));
    }

    #[test]
    fn test_no_reduction_normal_assign() {
        // x = y + z （縮約ではない）
        let assign_node = assign("x", var("y") + var("z"));
        let pattern = ReductionTilingSuggester::detect_reduction_pattern(&assign_node);

        assert!(pattern.is_none());
    }

    #[test]
    fn test_identity_values() {
        assert_eq!(
            ReductionOp::Add.identity(&DType::F32),
            AstNode::Const(Literal::F32(0.0))
        );
        assert_eq!(
            ReductionOp::Mul.identity(&DType::F32),
            AstNode::Const(Literal::F32(1.0))
        );
        assert_eq!(
            ReductionOp::Max.identity(&DType::F32),
            AstNode::Const(Literal::F32(f32::NEG_INFINITY))
        );
        assert_eq!(
            ReductionOp::Min.identity(&DType::F32),
            AstNode::Const(Literal::F32(f32::INFINITY))
        );
    }

    #[test]
    fn test_simple_sum_tiling() {
        let suggester = ReductionTilingSuggester::with_sizes(vec![4]);

        // for i in 0..16 { acc = acc + load(array, i) }
        let reduction_body = assign("acc", var("acc") + load(var("array"), var("i"), DType::F32));
        let reduction_loop = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(16),
            reduction_body,
        );

        let suggestions = suggester.suggest(&reduction_loop);

        assert!(
            !suggestions.is_empty(),
            "Should generate tiling suggestions"
        );

        // 変換結果にAllocate, Barrier, Deallocateが含まれることを確認
        let result = &suggestions[0].ast;
        let result_str = format!("{:?}", result);
        assert!(result_str.contains("Allocate"), "Should contain Allocate");
        assert!(result_str.contains("Barrier"), "Should contain Barrier");
        assert!(
            result_str.contains("Deallocate"),
            "Should contain Deallocate"
        );
    }

    #[test]
    fn test_no_tiling_for_non_reduction() {
        let suggester = ReductionTilingSuggester::new();

        // 通常のループ（縮約ではない）
        let normal_body = store(
            var("output"),
            var("i"),
            load(var("input"), var("i"), DType::F32),
        );
        let normal_loop = range("i", const_int(0), const_int(1), const_int(16), normal_body);

        let suggestions = suggester.suggest(&normal_loop);

        assert!(
            suggestions.is_empty(),
            "Should not generate suggestions for non-reduction loop"
        );
    }

    #[test]
    fn test_variable_bound_reduction() {
        let suggester = ReductionTilingSuggester::with_sizes(vec![4]);

        // for i in 0..N { acc = acc + load(array, i) }
        let reduction_body = assign("acc", var("acc") + load(var("array"), var("i"), DType::F32));
        let reduction_loop = range("i", const_int(0), const_int(1), var("N"), reduction_body);

        let suggestions = suggester.suggest(&reduction_loop);

        assert!(!suggestions.is_empty(), "Should handle variable bounds");

        // ceil_div(N, 4)が生成されることを確認
        let result_str = format!("{:?}", &suggestions[0].ast);
        assert!(
            result_str.contains("Idiv"),
            "Should contain ceil_div operation"
        );
    }

    #[test]
    fn test_nested_loop_only_reduces_inner() {
        let suggester = ReductionTilingSuggester::with_sizes(vec![4]);

        // for i in 0..M {
        //   for j in 0..N {
        //     acc = acc + load(array, j)
        //   }
        // }
        let inner_body = assign("acc", var("acc") + load(var("array"), var("j"), DType::F32));
        let inner_loop = range("j", const_int(0), const_int(1), var("N"), inner_body);
        let outer_loop = range("i", const_int(0), const_int(1), var("M"), inner_loop);

        let suggestions = suggester.suggest(&outer_loop);

        // 内側の縮約ループのみタイル化される
        assert!(!suggestions.is_empty());
    }
}
