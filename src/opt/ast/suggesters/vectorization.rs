//! ベクトル化（SIMD化）のためのSuggester実装
//!
//! ループ展開後の連続メモリアクセスパターンを検出し、
//! ベクトル命令（load_vec, store_vec）に変換します。

use crate::ast::{AstNode, DType, Literal};
use crate::opt::ast::{AstSuggestResult, AstSuggester};
use log::{debug, trace};
use std::collections::HashMap;

/// Store文グループの内部表現: (インデックス, オフセット, 型, 値式)
type StoreGroupData = (Vec<usize>, Vec<i64>, DType, AstNode);

/// 連続アクセスグループ
///
/// 同一ポインタに対する連続したオフセットでのアクセスをグループ化します。
#[derive(Debug, Clone)]
#[allow(dead_code)] // 将来の拡張用にフィールドを保持
struct ContiguousAccessGroup {
    /// 対象ポインタのAST表現
    ptr: AstNode,
    /// ベースオフセット式（定数オフセット部分を除いた式）
    base_offset: AstNode,
    /// 連続する定数オフセット [0, 1, 2, 3] など
    const_offsets: Vec<i64>,
    /// 対応する文のインデックス
    statement_indices: Vec<usize>,
    /// アクセスの型
    dtype: DType,
    /// Store文の場合の値式（代表として最初の文の値を保持）
    value_expr: Option<AstNode>,
}

/// ベクトル化（SIMD化）を提案するSuggester
///
/// Block内の連続したLoad/Store文を検出し、ベクトル命令に変換します。
/// LoopTilingSuggesterとLoopInliningSuggesterの後に適用することで、
/// 展開されたループ内の連続アクセスをSIMD化できます。
pub struct VectorizationSuggester {
    /// 利用可能なSIMD幅（優先順位順）
    available_widths: Vec<usize>,
    /// 最小グループサイズ（これ以上のアクセスがあればベクトル化）
    min_group_size: usize,
}

impl Default for VectorizationSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorizationSuggester {
    /// 新しいVectorizationSuggesterを作成
    ///
    /// デフォルトでは幅4, 2の順に試行し、最小グループサイズは2
    pub fn new() -> Self {
        Self {
            available_widths: vec![8, 4, 2],
            min_group_size: 2,
        }
    }

    /// SIMD幅を指定して作成
    pub fn with_widths(available_widths: Vec<usize>) -> Self {
        Self {
            available_widths,
            min_group_size: 2,
        }
    }

    /// 最小グループサイズを設定
    pub fn with_min_group_size(mut self, min_group_size: usize) -> Self {
        self.min_group_size = min_group_size;
        self
    }

    /// 定数オフセットが連続しているかチェック
    fn are_offsets_contiguous(offsets: &[i64]) -> bool {
        if offsets.len() < 2 {
            return false;
        }
        let mut sorted = offsets.to_vec();
        sorted.sort();
        for i in 1..sorted.len() {
            if sorted[i] - sorted[i - 1] != 1 {
                return false;
            }
        }
        true
    }

    /// オフセット式を「ベース式 + 定数」の形式に分解
    ///
    /// 成功した場合は (ベース式, 定数オフセット) を返す
    fn decompose_offset(offset: &AstNode) -> Option<(AstNode, i64)> {
        match offset {
            // 単純な定数
            AstNode::Const(Literal::I64(n)) => Some((AstNode::Const(Literal::I64(0)), *n)),

            // base + const
            AstNode::Add(a, b) => {
                if let AstNode::Const(Literal::I64(n)) = b.as_ref() {
                    Some((a.as_ref().clone(), *n))
                } else if let AstNode::Const(Literal::I64(n)) = a.as_ref() {
                    Some((b.as_ref().clone(), *n))
                } else {
                    // 両方とも定数でない場合、全体をベースとして定数0を返す
                    Some((offset.clone(), 0))
                }
            }

            // base - const (base + (-const)として扱う)
            AstNode::Mul(a, b) => {
                // -1 * x または x * -1 のパターンをチェック
                if let AstNode::Const(Literal::I64(-1)) = a.as_ref()
                    && let AstNode::Const(Literal::I64(n)) = b.as_ref()
                {
                    return Some((AstNode::Const(Literal::I64(0)), -(*n)));
                }
                if let AstNode::Const(Literal::I64(-1)) = b.as_ref()
                    && let AstNode::Const(Literal::I64(n)) = a.as_ref()
                {
                    return Some((AstNode::Const(Literal::I64(0)), -(*n)));
                }
                // その他の乗算はベースとして扱う
                Some((offset.clone(), 0))
            }

            // 変数やその他の式はそのままベースとして扱う
            _ => Some((offset.clone(), 0)),
        }
    }

    /// 2つのベース式が等価かどうかを判定
    fn are_bases_equivalent(a: &AstNode, b: &AstNode) -> bool {
        // Debug表現で比較（簡易的な実装）
        format!("{:?}", a) == format!("{:?}", b)
    }

    /// Block内のStore文から連続アクセスグループを抽出
    fn extract_store_groups(&self, statements: &[AstNode]) -> Vec<ContiguousAccessGroup> {
        // ptr -> base_offset -> StoreGroupData
        let mut groups: HashMap<String, HashMap<String, StoreGroupData>> = HashMap::new();

        for (idx, stmt) in statements.iter().enumerate() {
            if let AstNode::Store { ptr, offset, value } = stmt
                && let Some((base, const_offset)) = Self::decompose_offset(offset)
            {
                let ptr_key = format!("{:?}", ptr);
                let base_key = format!("{:?}", base);

                // 値の型を推論
                let dtype = value.infer_type();

                let entry = groups.entry(ptr_key).or_default();
                let group = entry.entry(base_key.clone()).or_insert_with(|| {
                    (
                        Vec::new(),
                        Vec::new(),
                        dtype.clone(),
                        value.as_ref().clone(),
                    )
                });

                group.0.push(idx);
                group.1.push(const_offset);
            }
        }

        // グループを連続アクセスグループに変換
        let mut result = Vec::new();
        for (_ptr_key, base_groups) in groups {
            for (_base_key, (indices, offsets, dtype, value_expr)) in base_groups {
                // ステートメントインデックスも連続している必要がある
                // 例: indices = [2, 5] はNG（間に他のステートメントがある）
                //     indices = [3, 4, 5] はOK（連続している）
                let indices_contiguous = if indices.len() > 1 {
                    let mut sorted_indices = indices.clone();
                    sorted_indices.sort();
                    sorted_indices.windows(2).all(|w| w[1] == w[0] + 1)
                } else {
                    true
                };

                if indices.len() >= self.min_group_size
                    && indices_contiguous
                    && Self::are_offsets_contiguous(&offsets)
                    // ptrとbase_offsetを再構築（キーから復元はできないので、最初の文から取得）
                    && let AstNode::Store { ptr, offset, .. } = &statements[indices[0]]
                    && let Some((base, _)) = Self::decompose_offset(offset)
                {
                    result.push(ContiguousAccessGroup {
                        ptr: ptr.as_ref().clone(),
                        base_offset: base,
                        const_offsets: offsets,
                        statement_indices: indices,
                        dtype,
                        value_expr: Some(value_expr),
                    });
                }
            }
        }

        result
    }

    /// 式を再帰的にベクトル化
    ///
    /// スカラー式をベクトル式に変換します。
    /// Load → load_vec, Const → broadcast, 演算 → そのまま（要素ごとに適用）
    ///
    /// ベクトル化できない場合（Loadのオフセットが一致しない等）はNoneを返す。
    fn try_vectorize_expr(
        &self,
        expr: &AstNode,
        width: usize,
        base_offset: &AstNode,
    ) -> Option<AstNode> {
        match expr {
            // 定数 → broadcast
            AstNode::Const(lit) => {
                let scalar_type = match lit {
                    Literal::F32(_) => DType::F32,
                    Literal::I64(_) => DType::I64,
                    Literal::I32(_) => DType::I32,
                    Literal::Bool(_) => DType::Bool,
                };
                let vec_type = scalar_type.to_vec(width);
                Some(AstNode::Cast(Box::new(expr.clone()), vec_type))
            }

            // Load → load_vec（連続アクセスの場合のみ）
            AstNode::Load {
                ptr,
                offset,
                count: 1,
                dtype,
            } => {
                // オフセットを分解してベースを取得
                if let Some((load_base, _const_offset)) = Self::decompose_offset(offset) {
                    // ベースが一致すれば連続アクセスとみなしてベクトル化
                    if Self::are_bases_equivalent(&load_base, base_offset) {
                        let vec_type = dtype.to_vec(width);
                        return Some(AstNode::Load {
                            ptr: ptr.clone(),
                            offset: Box::new(base_offset.clone()),
                            count: width,
                            dtype: vec_type,
                        });
                    }
                }
                // 連続でない場合はベクトル化失敗
                // スカラーLoadをベクトルにキャストするのは不正なコードになる
                None
            }

            // 二項演算 → 再帰的にベクトル化
            AstNode::Add(a, b) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                let b_vec = self.try_vectorize_expr(b, width, base_offset)?;
                Some(AstNode::Add(Box::new(a_vec), Box::new(b_vec)))
            }
            AstNode::Mul(a, b) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                let b_vec = self.try_vectorize_expr(b, width, base_offset)?;
                Some(AstNode::Mul(Box::new(a_vec), Box::new(b_vec)))
            }
            AstNode::Max(a, b) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                let b_vec = self.try_vectorize_expr(b, width, base_offset)?;
                Some(AstNode::Max(Box::new(a_vec), Box::new(b_vec)))
            }
            AstNode::Idiv(a, b) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                let b_vec = self.try_vectorize_expr(b, width, base_offset)?;
                Some(AstNode::Idiv(Box::new(a_vec), Box::new(b_vec)))
            }
            AstNode::Rem(a, b) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                let b_vec = self.try_vectorize_expr(b, width, base_offset)?;
                Some(AstNode::Rem(Box::new(a_vec), Box::new(b_vec)))
            }

            // 比較演算
            AstNode::Lt(a, b) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                let b_vec = self.try_vectorize_expr(b, width, base_offset)?;
                Some(AstNode::Lt(Box::new(a_vec), Box::new(b_vec)))
            }
            AstNode::Le(a, b) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                let b_vec = self.try_vectorize_expr(b, width, base_offset)?;
                Some(AstNode::Le(Box::new(a_vec), Box::new(b_vec)))
            }
            AstNode::Gt(a, b) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                let b_vec = self.try_vectorize_expr(b, width, base_offset)?;
                Some(AstNode::Gt(Box::new(a_vec), Box::new(b_vec)))
            }
            AstNode::Ge(a, b) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                let b_vec = self.try_vectorize_expr(b, width, base_offset)?;
                Some(AstNode::Ge(Box::new(a_vec), Box::new(b_vec)))
            }
            AstNode::Eq(a, b) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                let b_vec = self.try_vectorize_expr(b, width, base_offset)?;
                Some(AstNode::Eq(Box::new(a_vec), Box::new(b_vec)))
            }
            AstNode::Ne(a, b) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                let b_vec = self.try_vectorize_expr(b, width, base_offset)?;
                Some(AstNode::Ne(Box::new(a_vec), Box::new(b_vec)))
            }

            // 単項演算 → 再帰的にベクトル化
            AstNode::Recip(a) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                Some(AstNode::Recip(Box::new(a_vec)))
            }
            AstNode::Sqrt(a) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                Some(AstNode::Sqrt(Box::new(a_vec)))
            }
            AstNode::Log2(a) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                Some(AstNode::Log2(Box::new(a_vec)))
            }
            AstNode::Exp2(a) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                Some(AstNode::Exp2(Box::new(a_vec)))
            }
            AstNode::Sin(a) => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                Some(AstNode::Sin(Box::new(a_vec)))
            }

            // Fma
            AstNode::Fma { a, b, c } => {
                let a_vec = self.try_vectorize_expr(a, width, base_offset)?;
                let b_vec = self.try_vectorize_expr(b, width, base_offset)?;
                let c_vec = self.try_vectorize_expr(c, width, base_offset)?;
                Some(AstNode::Fma {
                    a: Box::new(a_vec),
                    b: Box::new(b_vec),
                    c: Box::new(c_vec),
                })
            }

            // Cast
            AstNode::Cast(inner, dtype) => {
                let inner_vec = self.try_vectorize_expr(inner, width, base_offset)?;
                let vec_type = dtype.to_vec(width);
                Some(AstNode::Cast(Box::new(inner_vec), vec_type))
            }

            // 変数はそのまま（ループ変数など、スカラーのまま）
            // ベクトル化できないのでNoneを返す
            AstNode::Var(_) => None,

            // その他はベクトル化できない
            _ => None,
        }
    }

    /// 式にLoadノードが含まれているかチェック
    ///
    /// Loadを含まない式（純粋な定数やスカラー変数のみ）はベクトル化しても
    /// パフォーマンス向上にならない上、不正なコードを生成する可能性がある。
    fn contains_load(expr: &AstNode) -> bool {
        match expr {
            AstNode::Load { .. } => true,
            // 二項演算
            AstNode::Add(a, b)
            | AstNode::Mul(a, b)
            | AstNode::Max(a, b)
            | AstNode::Idiv(a, b)
            | AstNode::Rem(a, b)
            | AstNode::Lt(a, b)
            | AstNode::Le(a, b)
            | AstNode::Gt(a, b)
            | AstNode::Ge(a, b)
            | AstNode::Eq(a, b)
            | AstNode::Ne(a, b) => Self::contains_load(a) || Self::contains_load(b),
            // 単項演算
            AstNode::Recip(a)
            | AstNode::Sqrt(a)
            | AstNode::Log2(a)
            | AstNode::Exp2(a)
            | AstNode::Sin(a) => Self::contains_load(a),
            // Fma
            AstNode::Fma { a, b, c } => {
                Self::contains_load(a) || Self::contains_load(b) || Self::contains_load(c)
            }
            // Cast
            AstNode::Cast(inner, _) => Self::contains_load(inner),
            // その他（定数、変数など）はLoadを含まない
            _ => false,
        }
    }

    /// 式構造が同一かチェック（オフセット値は無視）
    fn is_structurally_equivalent(expr1: &AstNode, expr2: &AstNode) -> bool {
        match (expr1, expr2) {
            // 二項演算
            (AstNode::Add(a1, b1), AstNode::Add(a2, b2))
            | (AstNode::Mul(a1, b1), AstNode::Mul(a2, b2))
            | (AstNode::Max(a1, b1), AstNode::Max(a2, b2))
            | (AstNode::Idiv(a1, b1), AstNode::Idiv(a2, b2))
            | (AstNode::Rem(a1, b1), AstNode::Rem(a2, b2))
            | (AstNode::Lt(a1, b1), AstNode::Lt(a2, b2))
            | (AstNode::Le(a1, b1), AstNode::Le(a2, b2))
            | (AstNode::Gt(a1, b1), AstNode::Gt(a2, b2))
            | (AstNode::Ge(a1, b1), AstNode::Ge(a2, b2))
            | (AstNode::Eq(a1, b1), AstNode::Eq(a2, b2))
            | (AstNode::Ne(a1, b1), AstNode::Ne(a2, b2)) => {
                Self::is_structurally_equivalent(a1, a2) && Self::is_structurally_equivalent(b1, b2)
            }

            // 単項演算
            (AstNode::Recip(a1), AstNode::Recip(a2))
            | (AstNode::Sqrt(a1), AstNode::Sqrt(a2))
            | (AstNode::Log2(a1), AstNode::Log2(a2))
            | (AstNode::Exp2(a1), AstNode::Exp2(a2))
            | (AstNode::Sin(a1), AstNode::Sin(a2)) => Self::is_structurally_equivalent(a1, a2),

            // Fma
            (
                AstNode::Fma {
                    a: a1,
                    b: b1,
                    c: c1,
                },
                AstNode::Fma {
                    a: a2,
                    b: b2,
                    c: c2,
                },
            ) => {
                Self::is_structurally_equivalent(a1, a2)
                    && Self::is_structurally_equivalent(b1, b2)
                    && Self::is_structurally_equivalent(c1, c2)
            }

            // Cast
            (AstNode::Cast(a1, dt1), AstNode::Cast(a2, dt2)) => {
                dt1 == dt2 && Self::is_structurally_equivalent(a1, a2)
            }

            // Load: 同じポインタならOK（オフセットは無視）
            (AstNode::Load { ptr: p1, .. }, AstNode::Load { ptr: p2, .. }) => {
                format!("{:?}", p1) == format!("{:?}", p2)
            }

            // 定数: 型が同じならOK（値は無視）
            (AstNode::Const(l1), AstNode::Const(l2)) => {
                std::mem::discriminant(l1) == std::mem::discriminant(l2)
            }

            // 変数: 名前が同じならOK
            (AstNode::Var(n1), AstNode::Var(n2)) => n1 == n2,

            _ => false,
        }
    }

    /// Block内の文をベクトル化した新しいBlockを生成
    fn vectorize_block(
        &self,
        statements: &[AstNode],
        scope: &crate::ast::Scope,
    ) -> Option<AstNode> {
        let groups = self.extract_store_groups(statements);
        if groups.is_empty() {
            return None;
        }

        trace!(
            "VectorizationSuggester: Found {} contiguous access groups",
            groups.len()
        );

        // 各グループに対して最適なSIMD幅を選択
        let mut new_statements = statements.to_vec();
        let mut indices_to_remove: Vec<usize> = Vec::new();
        let mut vectorized_any = false;

        for group in &groups {
            // グループサイズに合うSIMD幅を探す
            let group_size = group.statement_indices.len();
            let width = self
                .available_widths
                .iter()
                .copied()
                .find(|&w| w <= group_size);

            if let Some(width) = width {
                // 式構造が同一かチェック
                let first_idx = group.statement_indices[0];
                if let AstNode::Store { value, .. } = &statements[first_idx] {
                    let first_value = value.as_ref();

                    trace!(
                        "vectorize_block: group_size={}, width={}, base_offset={:?}, first_value={:?}",
                        group_size, width, group.base_offset, first_value
                    );

                    let all_equivalent = group.statement_indices.iter().all(|&idx| {
                        if let AstNode::Store { value, .. } = &statements[idx] {
                            Self::is_structurally_equivalent(first_value, value)
                        } else {
                            false
                        }
                    });

                    let has_load = Self::contains_load(first_value);
                    trace!(
                        "vectorize_block: all_equivalent={}, has_load={}",
                        all_equivalent, has_load
                    );

                    // Loadを含まない式（純粋な定数など）はベクトル化しない
                    // これらはベクトル化してもパフォーマンス向上にならず、
                    // 不正なコード（float2をfloat*に代入など）を生成する可能性がある
                    if all_equivalent && has_load {
                        // ベクトル化を試行（失敗した場合はスキップ）
                        let result =
                            self.try_vectorize_expr(first_value, width, &group.base_offset);
                        trace!(
                            "vectorize_block: try_vectorize_expr result={:?}",
                            result.is_some()
                        );
                        if let Some(vectorized_value) = result {
                            trace!("vectorize_block: vectorized_value={:?}", vectorized_value);
                            let vectorized_store = AstNode::Store {
                                ptr: Box::new(group.ptr.clone()),
                                offset: Box::new(group.base_offset.clone()),
                                value: Box::new(vectorized_value),
                            };

                            // 最初の文を置き換え、残りを削除リストに追加
                            new_statements[first_idx] = vectorized_store;
                            for &idx in group.statement_indices.iter().skip(1).take(width - 1) {
                                indices_to_remove.push(idx);
                            }

                            vectorized_any = true;
                            trace!(
                                "VectorizationSuggester: Vectorized {} statements with width {}",
                                width, width
                            );
                        }
                    }
                }
            }
        }

        if !vectorized_any {
            return None;
        }

        // 削除対象のインデックスをソートして逆順で削除
        indices_to_remove.sort();
        indices_to_remove.dedup();
        for idx in indices_to_remove.into_iter().rev() {
            new_statements.remove(idx);
        }

        Some(AstNode::Block {
            statements: new_statements,
            scope: Box::new(scope.clone()),
        })
    }

    /// AST全体を走査してベクトル化候補を収集
    fn collect_vectorization_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        match ast {
            // Blockノードでベクトル化を試みる
            AstNode::Block { statements, scope } => {
                // このBlock自体をベクトル化
                if let Some(vectorized) = self.vectorize_block(statements, scope.as_ref()) {
                    candidates.push(vectorized);
                }

                // 子ノードを再帰的に探索
                for (i, stmt) in statements.iter().enumerate() {
                    for vectorized_stmt in self.collect_vectorization_candidates(stmt) {
                        let mut new_stmts = statements.clone();
                        new_stmts[i] = vectorized_stmt;
                        candidates.push(AstNode::Block {
                            statements: new_stmts,
                            scope: scope.clone(),
                        });
                    }
                }
            }

            // Range内のBlockを探索
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => {
                for vectorized_body in self.collect_vectorization_candidates(body) {
                    candidates.push(AstNode::Range {
                        var: var.clone(),
                        start: start.clone(),
                        step: step.clone(),
                        stop: stop.clone(),
                        body: Box::new(vectorized_body),
                    });
                }
            }

            // Function内を探索
            AstNode::Function {
                name,
                params,
                return_type,
                body,
            } => {
                for vectorized_body in self.collect_vectorization_candidates(body) {
                    candidates.push(AstNode::Function {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(vectorized_body),
                    });
                }
            }

            // Kernel内を探索
            AstNode::Kernel {
                name,
                params,
                return_type,
                body,
                default_grid_size,
                default_thread_group_size,
            } => {
                for vectorized_body in self.collect_vectorization_candidates(body) {
                    candidates.push(AstNode::Kernel {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(vectorized_body),
                        default_grid_size: default_grid_size.clone(),
                        default_thread_group_size: default_thread_group_size.clone(),
                    });
                }
            }

            // Program内の各関数を探索
            AstNode::Program {
                functions,
                execution_waves,
            } => {
                for (i, func) in functions.iter().enumerate() {
                    for vectorized_func in self.collect_vectorization_candidates(func) {
                        let mut new_functions = functions.clone();
                        new_functions[i] = vectorized_func;
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

impl AstSuggester for VectorizationSuggester {
    fn name(&self) -> &str {
        "Vectorization"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        trace!("VectorizationSuggester: Generating vectorization suggestions");
        let candidates = self.collect_vectorization_candidates(ast);
        let suggestions = super::deduplicate_candidates(candidates);
        debug!(
            "VectorizationSuggester: Generated {} unique suggestions",
            suggestions.len()
        );

        suggestions
            .into_iter()
            .map(|ast| AstSuggestResult::with_description(ast, self.name(), "vectorize SIMD"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Scope;
    use crate::ast::helper::{const_int, load, store, var};

    /// 連続オフセット検出のテスト
    #[test]
    fn test_contiguous_offsets() {
        assert!(VectorizationSuggester::are_offsets_contiguous(&[
            0, 1, 2, 3
        ]));
        assert!(VectorizationSuggester::are_offsets_contiguous(&[
            3, 2, 1, 0
        ])); // 順序は問わない
        assert!(!VectorizationSuggester::are_offsets_contiguous(&[
            0, 2, 4, 6
        ])); // ギャップあり
        assert!(!VectorizationSuggester::are_offsets_contiguous(&[0])); // 1要素
    }

    /// オフセット分解のテスト
    #[test]
    fn test_decompose_offset() {
        // 単純な定数
        let offset = const_int(5);
        let (base, c) = VectorizationSuggester::decompose_offset(&offset).unwrap();
        assert!(matches!(base, AstNode::Const(Literal::I64(0))));
        assert_eq!(c, 5);

        // base + const
        let offset = var("i") + const_int(3);
        let (base, c) = VectorizationSuggester::decompose_offset(&offset).unwrap();
        assert!(matches!(base, AstNode::Var(_)));
        assert_eq!(c, 3);
    }

    /// 単純なStore文のベクトル化テスト
    #[test]
    fn test_simple_store_vectorization() {
        let suggester = VectorizationSuggester::with_widths(vec![4, 2]);

        // 4つの連続Store文を含むBlock
        let statements = vec![
            store(
                var("out"),
                const_int(0),
                load(var("in"), const_int(0), DType::F32),
            ),
            store(
                var("out"),
                const_int(1),
                load(var("in"), const_int(1), DType::F32),
            ),
            store(
                var("out"),
                const_int(2),
                load(var("in"), const_int(2), DType::F32),
            ),
            store(
                var("out"),
                const_int(3),
                load(var("in"), const_int(3), DType::F32),
            ),
        ];

        let block = AstNode::Block {
            statements,
            scope: Box::new(Scope::default()),
        };

        let suggestions = suggester.suggest(&block);

        // ベクトル化候補が生成されるはず
        assert!(
            !suggestions.is_empty(),
            "Should generate vectorization suggestions"
        );

        // ベクトル化後の文数が減っているはず
        if let AstNode::Block {
            statements: new_stmts,
            ..
        } = &suggestions[0].ast
        {
            assert!(
                new_stmts.len() < 4,
                "Vectorized block should have fewer statements"
            );
        }
    }

    /// 構造的等価性チェックのテスト
    #[test]
    fn test_structural_equivalence() {
        // 同じ構造の式
        let expr1 = load(var("in"), const_int(0), DType::F32) + const_int(1);
        let expr2 = load(var("in"), const_int(1), DType::F32) + const_int(1);
        assert!(VectorizationSuggester::is_structurally_equivalent(
            &expr1, &expr2
        ));

        // 異なる構造の式
        let expr3 = load(var("in"), const_int(0), DType::F32) * const_int(2);
        assert!(!VectorizationSuggester::is_structurally_equivalent(
            &expr1, &expr3
        ));
    }

    /// ベクトル化対象外のケース
    #[test]
    fn test_non_contiguous_not_vectorized() {
        let suggester = VectorizationSuggester::new();

        // 非連続なオフセット
        let statements = vec![
            store(
                var("out"),
                const_int(0),
                load(var("in"), const_int(0), DType::F32),
            ),
            store(
                var("out"),
                const_int(2), // ギャップ
                load(var("in"), const_int(2), DType::F32),
            ),
        ];

        let block = AstNode::Block {
            statements,
            scope: Box::new(Scope::default()),
        };

        let suggestions = suggester.suggest(&block);

        // 非連続なのでベクトル化されない
        assert!(suggestions.is_empty());
    }
}
