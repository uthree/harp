//! WMMA (Tensor Core) 行列積最適化 Suggester
//!
//! 3重ループの行列積パターンを検出し、WmmaMatmul ノードに変換する。
//!
//! # 検出パターン
//!
//! ```text
//! for ridx0 in 0..M:           // 外側ループ (M次元)
//!     for ridx2 in 0..N:       // 中間ループ (N次元)
//!         acc = 0
//!         for ridx1 in 0..K:   // 内側ループ (K次元、reduce軸)
//!             acc += A[ridx0, ridx1] * B[ridx1, ridx2]
//!         C[ridx0, ridx2] = acc
//! ```

use crate::ast::{AddressSpace, AstNode, DType, Literal, Scope};
use crate::opt::ast::{AstSuggestResult, AstSuggester};
use log::{debug, trace};

/// 検出された行列積パターンの情報
#[derive(Debug, Clone)]
#[allow(dead_code)] // デバッグや将来の拡張用
struct MatmulPattern {
    /// 外側ループ変数（M次元）
    m_var: String,
    /// 中間ループ変数（N次元）
    n_var: String,
    /// 内側ループ変数（K次元、reduce軸）
    k_var: String,

    /// ループ境界
    m_bound: AstNode,
    n_bound: AstNode,
    k_bound: AstNode,

    /// 行列Aのロード情報
    a_ptr: AstNode,
    a_stride: AstNode, // M方向のstride (= K)

    /// 行列Bのロード情報
    b_ptr: AstNode,
    b_stride: AstNode, // K方向のstride (= N)

    /// 出力行列Cの情報
    c_ptr: AstNode,
    c_stride: AstNode, // M方向のstride (= N)

    /// データ型
    dtype: DType,
}

/// WMMA最適化を提案するSuggester
pub struct WmmaSuggester {
    /// 16の倍数のみ変換するか（初期実装: true）
    strict_alignment: bool,
}

impl WmmaSuggester {
    /// 新しいWmmaSuggesterを作成
    pub fn new() -> Self {
        Self {
            strict_alignment: true,
        }
    }

    /// アライメント要件を緩和する設定
    #[allow(dead_code)]
    pub fn with_relaxed_alignment(mut self) -> Self {
        self.strict_alignment = false;
        self
    }

    /// ASTから行列積パターンを検出
    fn detect_matmul_pattern(&self, ast: &AstNode) -> Option<MatmulPattern> {
        // 外側 Range (M ループ)
        let AstNode::Range {
            var: m_var,
            start: m_start,
            step: m_step,
            stop: m_bound,
            body: m_body,
            ..
        } = ast
        else {
            return None;
        };

        // start=0, step=1 であることを確認
        if !is_zero(m_start) || !is_one(m_step) {
            return None;
        }

        // 中間 Range (N ループ)
        let AstNode::Range {
            var: n_var,
            start: n_start,
            step: n_step,
            stop: n_bound,
            body: n_body,
            ..
        } = m_body.as_ref()
        else {
            return None;
        };

        if !is_zero(n_start) || !is_one(n_step) {
            return None;
        }

        // Block with scope
        let AstNode::Block {
            statements, scope, ..
        } = n_body.as_ref()
        else {
            return None;
        };

        // ブロック構造: [acc初期化, Kループ, Store]
        if statements.len() != 3 {
            return None;
        }

        // acc 初期化
        let AstNode::Assign {
            var: acc_var,
            value: acc_init,
        } = &statements[0]
        else {
            return None;
        };

        // accが0で初期化されているか確認
        if !is_zero_f32(acc_init) {
            return None;
        }

        // スコープでacc変数を確認（acc変数が宣言されていること）
        if scope.check_read(acc_var).is_err() {
            trace!("acc variable '{}' not found in scope", acc_var);
            return None;
        }

        // 内側 Range (K ループ)
        let AstNode::Range {
            var: k_var,
            start: k_start,
            step: k_step,
            stop: k_bound,
            body: k_body,
            ..
        } = &statements[1]
        else {
            return None;
        };

        if !is_zero(k_start) || !is_one(k_step) {
            return None;
        }

        // K ループの本体: acc = acc + Mul(Load, Load)
        let AstNode::Assign {
            var: inner_var,
            value: inner_value,
        } = k_body.as_ref()
        else {
            return None;
        };

        if inner_var != acc_var {
            return None;
        }

        // acc + Mul(Load(A), Load(B)) パターン
        let (a_load, b_load) = self.extract_matmul_loads(inner_value, acc_var)?;

        // Loadからポインタとインデックス式を抽出
        let (a_ptr, a_index) = extract_load_info(&a_load)?;
        let (b_ptr, b_index) = extract_load_info(&b_load)?;

        // インデックス式からストライドを抽出
        // A[m_var * K + k_var] → a_stride = K
        // B[k_var * N + n_var] → b_stride = N
        let a_stride = self.extract_stride(&a_index, m_var, k_var)?;
        let b_stride = self.extract_stride(&b_index, k_var, n_var)?;

        // Store文を確認
        let AstNode::Store {
            ptr: c_ptr,
            offset: c_offset,
            value: store_value,
        } = &statements[2]
        else {
            return None;
        };

        // Store値がacc変数であることを確認
        if !matches!(store_value.as_ref(), AstNode::Var(v) if v == acc_var) {
            return None;
        }

        // C[m_var * N + n_var] → c_stride = N
        let c_stride = self.extract_stride(c_offset, m_var, n_var)?;

        // データ型を取得
        let dtype = self.get_load_dtype(&a_load)?;

        Some(MatmulPattern {
            m_var: m_var.clone(),
            n_var: n_var.clone(),
            k_var: k_var.clone(),
            m_bound: *m_bound.clone(),
            n_bound: *n_bound.clone(),
            k_bound: *k_bound.clone(),
            a_ptr: *a_ptr.clone(),
            a_stride,
            b_ptr: *b_ptr.clone(),
            b_stride,
            c_ptr: *c_ptr.clone(),
            c_stride,
            dtype,
        })
    }

    /// acc = acc + Mul(Load, Load) パターンからLoadノードを抽出
    fn extract_matmul_loads(
        &self,
        value: &AstNode,
        acc_var: &str,
    ) -> Option<(AstNode, AstNode)> {
        // Add(Var(acc), Mul(Load, Load)) パターン
        let AstNode::Add(lhs, rhs) = value else {
            return None;
        };

        // lhs が acc 変数か確認
        if !matches!(lhs.as_ref(), AstNode::Var(v) if v == acc_var) {
            return None;
        }

        // rhs が Mul(Load, Load) か確認
        let AstNode::Mul(a, b) = rhs.as_ref() else {
            return None;
        };

        // 両方がLoadであることを確認
        if !matches!(a.as_ref(), AstNode::Load { .. }) {
            return None;
        }
        if !matches!(b.as_ref(), AstNode::Load { .. }) {
            return None;
        }

        Some((*a.clone(), *b.clone()))
    }

    /// インデックス式からストライドを抽出
    ///
    /// 例: `row_var * stride + col_var` → stride
    fn extract_stride(
        &self,
        index: &AstNode,
        row_var: &str,
        col_var: &str,
    ) -> Option<AstNode> {
        // パターン: Add(Mul(Var(row), stride), Var(col))
        // または: Add(Var(col), Mul(Var(row), stride))

        let AstNode::Add(lhs, rhs) = index else {
            // 単純な場合: row_var だけ（stride=1, col_var=0）
            // この場合は対応しない（行列積では通常ありえない）
            return None;
        };

        // パターン1: Add(Mul(Var(row), stride), Var(col))
        if let Some(stride) = self.try_extract_stride_pattern(lhs, rhs, row_var, col_var) {
            return Some(stride);
        }

        // パターン2: Add(Var(col), Mul(Var(row), stride))
        if let Some(stride) = self.try_extract_stride_pattern(rhs, lhs, row_var, col_var) {
            return Some(stride);
        }

        None
    }

    /// stride抽出のヘルパー: Mul(Var(row), stride) と Var(col) のペアを検証
    fn try_extract_stride_pattern(
        &self,
        mul_part: &AstNode,
        var_part: &AstNode,
        row_var: &str,
        col_var: &str,
    ) -> Option<AstNode> {
        // var_part が col_var であることを確認
        if !matches!(var_part, AstNode::Var(v) if v == col_var) {
            return None;
        }

        // mul_part が Mul(Var(row), stride) または Mul(stride, Var(row))
        let AstNode::Mul(a, b) = mul_part else {
            return None;
        };

        if matches!(a.as_ref(), AstNode::Var(v) if v == row_var) {
            return Some(*b.clone());
        }

        if matches!(b.as_ref(), AstNode::Var(v) if v == row_var) {
            return Some(*a.clone());
        }

        None
    }

    /// LoadノードからDTypeを取得
    fn get_load_dtype(&self, load: &AstNode) -> Option<DType> {
        if let AstNode::Load { dtype, .. } = load {
            Some(dtype.clone())
        } else {
            None
        }
    }

    /// WMMA適用可能かチェック
    fn is_wmma_eligible(&self, pattern: &MatmulPattern) -> bool {
        // 1. dtype が F16
        if !matches!(pattern.dtype, DType::F16) {
            trace!("Not F16: {:?}", pattern.dtype);
            return false;
        }

        // 2. M, K, N が定数かつ16の倍数
        if self.strict_alignment {
            if !is_multiple_of_16(&pattern.m_bound)
                || !is_multiple_of_16(&pattern.k_bound)
                || !is_multiple_of_16(&pattern.n_bound)
            {
                trace!(
                    "Dimensions not multiple of 16: M={:?}, K={:?}, N={:?}",
                    pattern.m_bound,
                    pattern.k_bound,
                    pattern.n_bound
                );
                return false;
            }
        }

        true
    }

    /// MatmulPattern を WmmaMatmul ノードに変換
    fn convert_to_wmma(&self, pattern: MatmulPattern) -> AstNode {
        AstNode::WmmaMatmul {
            a_ptr: Box::new(pattern.a_ptr),
            a_offset: Box::new(AstNode::Const(Literal::I64(0))),
            a_stride: Box::new(pattern.a_stride),
            b_ptr: Box::new(pattern.b_ptr),
            b_offset: Box::new(AstNode::Const(Literal::I64(0))),
            b_stride: Box::new(pattern.b_stride),
            c_ptr: Box::new(pattern.c_ptr),
            c_offset: Box::new(AstNode::Const(Literal::I64(0))),
            c_stride: Box::new(pattern.c_stride),
            m: Box::new(pattern.m_bound),
            k: Box::new(pattern.k_bound),
            n: Box::new(pattern.n_bound),
            dtype_ab: DType::F16,
            dtype_c: DType::F32,
        }
    }

    /// Kernel/Function内のbodyを変換して新しいKernel/Functionを返す
    fn transform_kernel_body(&self, kernel: &AstNode) -> Option<AstNode> {
        match kernel {
            AstNode::Kernel {
                name,
                params,
                return_type,
                body,
                default_grid_size,
                default_thread_group_size,
            } => {
                let new_body = self.transform_body(body)?;
                Some(AstNode::Kernel {
                    name: name.clone(),
                    params: params.clone(),
                    return_type: return_type.clone(),
                    body: Box::new(new_body),
                    default_grid_size: default_grid_size.clone(),
                    default_thread_group_size: default_thread_group_size.clone(),
                })
            }
            AstNode::Function {
                name,
                params,
                return_type,
                body,
            } => {
                let new_body = self.transform_body(body)?;
                Some(AstNode::Function {
                    name: name.clone(),
                    params: params.clone(),
                    return_type: return_type.clone(),
                    body: Box::new(new_body),
                })
            }
            _ => None,
        }
    }

    /// body内のmatmulパターンを検出して変換
    fn transform_body(&self, body: &AstNode) -> Option<AstNode> {
        match body {
            AstNode::Block {
                statements,
                scope,
            } => {
                // 各statementを検査
                let mut new_statements = Vec::new();
                let mut changed = false;

                for stmt in statements {
                    if let Some(pattern) = self.detect_matmul_pattern(stmt) {
                        if self.is_wmma_eligible(&pattern) {
                            debug!(
                                "Detected WMMA-eligible matmul: M={:?}, K={:?}, N={:?}",
                                pattern.m_bound, pattern.k_bound, pattern.n_bound
                            );
                            new_statements.push(self.convert_to_wmma(pattern));
                            changed = true;
                            continue;
                        }
                    }
                    // 再帰的に子ノードを検査
                    if let Some(transformed) = self.transform_body(stmt) {
                        new_statements.push(transformed);
                        changed = true;
                    } else {
                        new_statements.push(stmt.clone());
                    }
                }

                if changed {
                    Some(AstNode::Block {
                        statements: new_statements,
                        scope: scope.clone(),
                    })
                } else {
                    None
                }
            }
            AstNode::Range { .. } => {
                // Rangeノード自体がmatmulパターンかチェック
                if let Some(pattern) = self.detect_matmul_pattern(body) {
                    if self.is_wmma_eligible(&pattern) {
                        debug!(
                            "Detected WMMA-eligible matmul at Range: M={:?}, K={:?}, N={:?}",
                            pattern.m_bound, pattern.k_bound, pattern.n_bound
                        );
                        return Some(AstNode::Block {
                            statements: vec![self.convert_to_wmma(pattern)],
                            scope: Box::new(Scope::new()),
                        });
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Program全体からWMMA候補を収集
    fn collect_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        match ast {
            AstNode::Program {
                functions,
                execution_waves,
            } => {
                let mut new_functions = Vec::new();
                let mut changed = false;

                for func in functions {
                    if let Some(transformed) = self.transform_kernel_body(func) {
                        new_functions.push(transformed);
                        changed = true;
                    } else {
                        new_functions.push(func.clone());
                    }
                }

                if changed {
                    candidates.push(AstNode::Program {
                        functions: new_functions,
                        execution_waves: execution_waves.clone(),
                    });
                }
            }
            AstNode::Kernel { .. } | AstNode::Function { .. } => {
                if let Some(transformed) = self.transform_kernel_body(ast) {
                    candidates.push(transformed);
                }
            }
            _ => {}
        }

        candidates
    }
}

impl Default for WmmaSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl AstSuggester for WmmaSuggester {
    fn name(&self) -> &str {
        "wmma"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        trace!("WmmaSuggester: Scanning for matmul patterns");
        let candidates = self.collect_candidates(ast);
        let suggestions = super::deduplicate_candidates(candidates);

        debug!(
            "WmmaSuggester: Generated {} suggestions",
            suggestions.len()
        );

        suggestions
            .into_iter()
            .map(|ast| {
                AstSuggestResult::with_description(ast, self.name(), "convert matmul to WMMA")
            })
            .collect()
    }
}

// ヘルパー関数

/// AstNodeが0か確認（整数）
fn is_zero(node: &AstNode) -> bool {
    matches!(
        node,
        AstNode::Const(Literal::I64(0))
            | AstNode::Const(Literal::I32(0))
            | AstNode::Const(Literal::U64(0))
    )
}

/// AstNodeが1か確認（整数）
fn is_one(node: &AstNode) -> bool {
    matches!(
        node,
        AstNode::Const(Literal::I64(1))
            | AstNode::Const(Literal::I32(1))
            | AstNode::Const(Literal::U64(1))
    )
}

/// AstNodeが0.0か確認（浮動小数点）
fn is_zero_f32(node: &AstNode) -> bool {
    matches!(
        node,
        AstNode::Const(Literal::F32(v)) if *v == 0.0
    ) || matches!(
        node,
        AstNode::Const(Literal::F16(v)) if *v == half::f16::from_f32(0.0)
    )
}

/// AstNodeが16の倍数の定数か確認
fn is_multiple_of_16(node: &AstNode) -> bool {
    match node {
        AstNode::Const(Literal::I64(v)) => *v > 0 && *v % 16 == 0,
        AstNode::Const(Literal::I32(v)) => *v > 0 && *v % 16 == 0,
        AstNode::Const(Literal::U64(v)) => *v > 0 && *v % 16 == 0,
        _ => false,
    }
}

/// LoadノードからポインタとオフセットExprを抽出
fn extract_load_info(load: &AstNode) -> Option<(&Box<AstNode>, &AstNode)> {
    if let AstNode::Load { ptr, offset, .. } = load {
        Some((ptr, offset.as_ref()))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;
    use crate::ast::{Mutability, ParallelInfo, VarDecl, VarKind};

    /// テスト用の3重ループmatmul ASTを構築
    fn build_matmul_ast(m: i64, k: i64, n: i64, dtype: DType) -> AstNode {
        // for ridx0 in 0..M:
        //     for ridx2 in 0..N:
        //         acc = 0.0
        //         for ridx1 in 0..K:
        //             acc = acc + A[ridx0 * K + ridx1] * B[ridx1 * N + ridx2]
        //         C[ridx0 * N + ridx2] = acc

        let m_const = const_int(m);
        let k_const = const_int(k);
        let n_const = const_int(n);

        // A[ridx0 * K + ridx1]
        let a_index = var("ridx0") * const_int(k) + var("ridx1");
        let a_load = AstNode::Load {
            ptr: Box::new(var("A")),
            offset: Box::new(a_index),
            count: 1,
            dtype: dtype.clone(),
        };

        // B[ridx1 * N + ridx2]
        let b_index = var("ridx1") * const_int(n) + var("ridx2");
        let b_load = AstNode::Load {
            ptr: Box::new(var("B")),
            offset: Box::new(b_index),
            count: 1,
            dtype: dtype.clone(),
        };

        // acc = acc + a_load * b_load
        let inner_assign = AstNode::Assign {
            var: "acc".to_string(),
            value: Box::new(var("acc") + a_load * b_load),
        };

        // K loop
        let k_loop = AstNode::Range {
            var: "ridx1".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(k_const.clone()),
            body: Box::new(inner_assign),
            parallel: ParallelInfo::default(),
        };

        // C[ridx0 * N + ridx2] = acc
        let c_index = var("ridx0") * const_int(n) + var("ridx2");
        let store = AstNode::Store {
            ptr: Box::new(var("C")),
            offset: Box::new(c_index),
            value: Box::new(var("acc")),
        };

        // Block with acc declaration
        let init_val = if dtype == DType::F16 {
            AstNode::Const(Literal::F16(half::f16::from_f32(0.0)))
        } else {
            AstNode::Const(Literal::F32(0.0))
        };

        let mut scope = Scope::new();
        scope
            .declare("acc".to_string(), DType::F32, Mutability::Mutable)
            .unwrap();

        let inner_block = AstNode::Block {
            statements: vec![
                AstNode::Assign {
                    var: "acc".to_string(),
                    value: Box::new(init_val),
                },
                k_loop,
                store,
            ],
            scope: Box::new(scope),
        };

        // N loop
        let n_loop = AstNode::Range {
            var: "ridx2".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(n_const),
            body: Box::new(inner_block),
            parallel: ParallelInfo::default(),
        };

        // M loop
        AstNode::Range {
            var: "ridx0".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(m_const),
            body: Box::new(n_loop),
            parallel: ParallelInfo::default(),
        }
    }

    /// テスト用のKernelでラップ
    fn wrap_in_kernel(body: AstNode) -> AstNode {
        let one = Box::new(const_int(1));
        AstNode::Kernel {
            name: Some("test_kernel".to_string()),
            params: vec![
                VarDecl {
                    name: "A".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F16), AddressSpace::Global),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "B".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F16), AddressSpace::Global),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "C".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32), AddressSpace::Global),
                    mutability: Mutability::Mutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: DType::Void,
            body: Box::new(AstNode::Block {
                statements: vec![body],
                scope: Box::new(Scope::new()),
            }),
            default_grid_size: [one.clone(), one.clone(), one.clone()],
            default_thread_group_size: [one.clone(), one.clone(), one],
        }
    }

    #[test]
    fn test_detect_matmul_pattern() {
        let suggester = WmmaSuggester::new();
        let matmul_ast = build_matmul_ast(32, 64, 48, DType::F16);

        let pattern = suggester.detect_matmul_pattern(&matmul_ast);
        assert!(pattern.is_some(), "Should detect matmul pattern");

        let pattern = pattern.unwrap();
        assert_eq!(pattern.m_var, "ridx0");
        assert_eq!(pattern.n_var, "ridx2");
        assert_eq!(pattern.k_var, "ridx1");
        assert!(matches!(pattern.m_bound, AstNode::Const(Literal::I64(32))));
        assert!(matches!(pattern.k_bound, AstNode::Const(Literal::I64(64))));
        assert!(matches!(pattern.n_bound, AstNode::Const(Literal::I64(48))));
    }

    #[test]
    fn test_wmma_eligibility_f16_aligned() {
        let suggester = WmmaSuggester::new();
        let matmul_ast = build_matmul_ast(32, 64, 48, DType::F16);

        let pattern = suggester.detect_matmul_pattern(&matmul_ast).unwrap();
        assert!(
            suggester.is_wmma_eligible(&pattern),
            "F16 with 16-aligned dimensions should be eligible"
        );
    }

    #[test]
    fn test_wmma_eligibility_f32_not_eligible() {
        let suggester = WmmaSuggester::new();
        let matmul_ast = build_matmul_ast(32, 64, 48, DType::F32);

        let pattern = suggester.detect_matmul_pattern(&matmul_ast).unwrap();
        assert!(
            !suggester.is_wmma_eligible(&pattern),
            "F32 should not be eligible for WMMA"
        );
    }

    #[test]
    fn test_wmma_eligibility_not_aligned() {
        let suggester = WmmaSuggester::new();
        // 15 is not a multiple of 16
        let matmul_ast = build_matmul_ast(15, 64, 48, DType::F16);

        let pattern = suggester.detect_matmul_pattern(&matmul_ast).unwrap();
        assert!(
            !suggester.is_wmma_eligible(&pattern),
            "Non-aligned dimensions should not be eligible"
        );
    }

    #[test]
    fn test_convert_to_wmma() {
        let suggester = WmmaSuggester::new();
        let matmul_ast = build_matmul_ast(32, 64, 48, DType::F16);

        let pattern = suggester.detect_matmul_pattern(&matmul_ast).unwrap();
        let wmma_node = suggester.convert_to_wmma(pattern);

        assert!(
            matches!(wmma_node, AstNode::WmmaMatmul { .. }),
            "Should convert to WmmaMatmul"
        );

        if let AstNode::WmmaMatmul {
            m,
            k,
            n,
            dtype_ab,
            dtype_c,
            ..
        } = wmma_node
        {
            assert!(matches!(*m, AstNode::Const(Literal::I64(32))));
            assert!(matches!(*k, AstNode::Const(Literal::I64(64))));
            assert!(matches!(*n, AstNode::Const(Literal::I64(48))));
            assert_eq!(dtype_ab, DType::F16);
            assert_eq!(dtype_c, DType::F32);
        }
    }

    #[test]
    fn test_suggester_with_kernel() {
        let suggester = WmmaSuggester::new();
        let matmul_ast = build_matmul_ast(32, 64, 48, DType::F16);
        let kernel = wrap_in_kernel(matmul_ast);

        let suggestions = suggester.suggest(&kernel);

        assert_eq!(suggestions.len(), 1, "Should produce one suggestion");
        assert_eq!(suggestions[0].suggester_name, "wmma");

        // 変換後のKernelにWmmaMatmulが含まれることを確認
        if let AstNode::Kernel { body, .. } = &suggestions[0].ast {
            let body_str = format!("{:?}", body);
            assert!(
                body_str.contains("WmmaMatmul"),
                "Transformed kernel should contain WmmaMatmul"
            );
        } else {
            panic!("Expected Kernel node");
        }
    }

    #[test]
    fn test_suggester_no_match_simple_loop() {
        let suggester = WmmaSuggester::new();

        // 単純な1重ループ（matmulではない）
        let simple_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(const_int(100)),
            body: Box::new(AstNode::Assign {
                var: "x".to_string(),
                value: Box::new(var("x") + const_int(1)),
            }),
            parallel: ParallelInfo::default(),
        };

        let kernel = wrap_in_kernel(simple_loop);
        let suggestions = suggester.suggest(&kernel);

        assert_eq!(
            suggestions.len(),
            0,
            "Simple loop should not match matmul pattern"
        );
    }
}
