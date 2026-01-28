//! 共有メモリ最適化のためのSuggester実装
//!
//! GPUの共有メモリを活用して、グローバルメモリアクセスを削減します。
//! 複数スレッドがデータを再利用するパターンを検出し、
//! 共有メモリへのプリロードと同期を挿入します。

use crate::ast::{AstNode, DType, Literal, ParallelInfo, ParallelKind, Scope};
use crate::opt::ast::{AstSuggestResult, AstSuggester};
use log::{debug, trace};

#[cfg(test)]
use crate::ast::{AddressSpace, Mutability, VarDecl, VarKind};

/// 共有メモリ最適化候補
#[derive(Debug, Clone)]
struct SharedMemoryCandidate {
    /// Load対象のポインタ変数名
    ptr_var: String,
    /// Load対象のデータ型
    dtype: DType,
    /// 内側ループ変数（再利用の軸）
    inner_loop_var: String,
    /// 再利用回数（内側ループの反復回数）
    reuse_count: i64,
    /// 共有メモリに格納するサイズ（要素数）
    shared_size: i64,
    /// 並列ループ変数
    parallel_var: String,
    /// 並列ループのサイズ
    parallel_size: i64,
}

/// 共有メモリ最適化を提案するSuggester
///
/// GpuThread並列化されたループ内で、内側ループで再利用されるデータを
/// 共有メモリにプリロードするパターンを検出・提案します。
pub struct SharedMemorySuggester {
    /// 最小再利用回数（これ以上でなければ変換しない）
    min_reuse_count: i64,
    /// 最大共有メモリサイズ（バイト）
    max_shared_size: usize,
}

impl Default for SharedMemorySuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedMemorySuggester {
    /// 新しいSharedMemorySuggesterを作成
    pub fn new() -> Self {
        Self {
            min_reuse_count: 4,
            max_shared_size: 48 * 1024, // 48KB (typical shared memory size)
        }
    }

    /// 最小再利用回数を設定
    pub fn with_min_reuse_count(mut self, count: i64) -> Self {
        self.min_reuse_count = count;
        self
    }

    /// 最大共有メモリサイズを設定
    pub fn with_max_shared_size(mut self, size: usize) -> Self {
        self.max_shared_size = size;
        self
    }

    /// 式がループ変数に依存しているかチェック
    fn depends_on_var(expr: &AstNode, var_name: &str) -> bool {
        match expr {
            AstNode::Var(name) => name == var_name,
            AstNode::Add(a, b)
            | AstNode::Mul(a, b)
            | AstNode::Rem(a, b)
            | AstNode::Idiv(a, b)
            | AstNode::Max(a, b)
            | AstNode::BitwiseAnd(a, b)
            | AstNode::BitwiseOr(a, b)
            | AstNode::BitwiseXor(a, b)
            | AstNode::LeftShift(a, b)
            | AstNode::RightShift(a, b)
            | AstNode::Lt(a, b)
            | AstNode::And(a, b) => {
                Self::depends_on_var(a, var_name) || Self::depends_on_var(b, var_name)
            }
            AstNode::Recip(a)
            | AstNode::Sqrt(a)
            | AstNode::Log2(a)
            | AstNode::Exp2(a)
            | AstNode::Sin(a)
            | AstNode::Floor(a)
            | AstNode::BitwiseNot(a)
            | AstNode::Not(a)
            | AstNode::Cast(a, _) => Self::depends_on_var(a, var_name),
            AstNode::Select {
                cond,
                then_val,
                else_val,
            } => {
                Self::depends_on_var(cond, var_name)
                    || Self::depends_on_var(then_val, var_name)
                    || Self::depends_on_var(else_val, var_name)
            }
            AstNode::Fma { a, b, c } => {
                Self::depends_on_var(a, var_name)
                    || Self::depends_on_var(b, var_name)
                    || Self::depends_on_var(c, var_name)
            }
            AstNode::Load { ptr, offset, .. } => {
                Self::depends_on_var(ptr, var_name) || Self::depends_on_var(offset, var_name)
            }
            _ => false,
        }
    }

    /// ループの反復回数を取得（定数の場合）
    fn get_loop_count(start: &AstNode, stop: &AstNode) -> Option<i64> {
        match (start, stop) {
            (AstNode::Const(Literal::I64(s)), AstNode::Const(Literal::I64(e))) => Some(e - s),
            _ => None,
        }
    }

    /// Load式からポインタ変数名を抽出
    fn extract_ptr_var(ptr: &AstNode) -> Option<String> {
        match ptr {
            AstNode::Var(name) => Some(name.clone()),
            _ => None,
        }
    }

    /// AST内のLoad操作を収集
    fn collect_loads(node: &AstNode, loads: &mut Vec<(String, DType, AstNode)>) {
        match node {
            AstNode::Load {
                ptr, offset, dtype, ..
            } => {
                if let Some(ptr_var) = Self::extract_ptr_var(ptr) {
                    loads.push((ptr_var, dtype.clone(), *offset.clone()));
                }
            }
            _ => {
                for child in node.children() {
                    Self::collect_loads(child, loads);
                }
            }
        }
    }

    /// 並列ループ内の内側ループで再利用されるLoadを検出
    fn find_shared_memory_candidates(
        &self,
        parallel_var: &str,
        parallel_size: i64,
        body: &AstNode,
    ) -> Vec<SharedMemoryCandidate> {
        let mut candidates = Vec::new();

        // bodyから内側のRangeループを探す
        self.find_inner_loops_with_loads(parallel_var, parallel_size, body, &mut candidates);

        candidates
    }

    /// 内側ループとそのLoadを再帰的に探索
    fn find_inner_loops_with_loads(
        &self,
        parallel_var: &str,
        parallel_size: i64,
        node: &AstNode,
        candidates: &mut Vec<SharedMemoryCandidate>,
    ) {
        match node {
            AstNode::Range {
                var: inner_var,
                start,
                stop,
                body,
                ..
            } => {
                // 内側ループの反復回数を取得
                if let Some(reuse_count) = Self::get_loop_count(start, stop) {
                    if reuse_count >= self.min_reuse_count {
                        // このループ内のLoadを収集
                        let mut loads = Vec::new();
                        Self::collect_loads(body, &mut loads);

                        // 内側ループ変数に依存しないLoadを候補として追加
                        for (ptr_var, dtype, offset) in loads {
                            if !Self::depends_on_var(&offset, inner_var) {
                                // 並列ループ変数にのみ依存するLoadが候補
                                let shared_size = parallel_size;
                                let element_size = dtype.size_in_bytes();

                                if (shared_size as usize * element_size) <= self.max_shared_size {
                                    trace!(
                                        "Found shared memory candidate: {} (reuse={}, size={})",
                                        ptr_var, reuse_count, shared_size
                                    );
                                    candidates.push(SharedMemoryCandidate {
                                        ptr_var,
                                        dtype,
                                        inner_loop_var: inner_var.clone(),
                                        reuse_count,
                                        shared_size,
                                        parallel_var: parallel_var.to_string(),
                                        parallel_size,
                                    });
                                }
                            }
                        }
                    }
                }

                // 更に内側のループも探索
                self.find_inner_loops_with_loads(parallel_var, parallel_size, body, candidates);
            }
            AstNode::Block { statements, .. } => {
                for stmt in statements {
                    self.find_inner_loops_with_loads(parallel_var, parallel_size, stmt, candidates);
                }
            }
            AstNode::If {
                then_body,
                else_body,
                ..
            } => {
                self.find_inner_loops_with_loads(
                    parallel_var,
                    parallel_size,
                    then_body,
                    candidates,
                );
                if let Some(else_b) = else_body {
                    self.find_inner_loops_with_loads(
                        parallel_var,
                        parallel_size,
                        else_b,
                        candidates,
                    );
                }
            }
            _ => {}
        }
    }

    /// LoadをSharedLoadに置換
    fn replace_load_with_shared_load(
        node: &AstNode,
        ptr_var: &str,
        shared_name: &str,
        parallel_var: &str,
        dtype: &DType,
    ) -> AstNode {
        match node {
            AstNode::Load {
                ptr,
                offset,
                count,
                dtype: load_dtype,
            } => {
                if let Some(var_name) = Self::extract_ptr_var(ptr) {
                    if var_name == ptr_var {
                        // SharedLoadに変換
                        // オフセットから並列変数部分だけを抽出
                        let shared_offset = Box::new(AstNode::Var(parallel_var.to_string()));
                        return AstNode::SharedLoad {
                            ptr: Box::new(AstNode::Var(shared_name.to_string())),
                            offset: shared_offset,
                            dtype: load_dtype.clone(),
                        };
                    }
                }
                // 変換しない場合はそのまま返す
                AstNode::Load {
                    ptr: Box::new(Self::replace_load_with_shared_load(
                        ptr,
                        ptr_var,
                        shared_name,
                        parallel_var,
                        dtype,
                    )),
                    offset: Box::new(Self::replace_load_with_shared_load(
                        offset,
                        ptr_var,
                        shared_name,
                        parallel_var,
                        dtype,
                    )),
                    count: *count,
                    dtype: load_dtype.clone(),
                }
            }
            _ => {
                // 再帰的に子ノードを変換
                node.map_children(&|child| {
                    Self::replace_load_with_shared_load(
                        child,
                        ptr_var,
                        shared_name,
                        parallel_var,
                        dtype,
                    )
                })
            }
        }
    }

    /// 共有メモリ最適化を適用した新しいASTを生成
    fn apply_shared_memory_optimization(
        &self,
        kernel: &AstNode,
        candidate: &SharedMemoryCandidate,
    ) -> Option<AstNode> {
        if let AstNode::Kernel {
            name,
            params,
            return_type,
            body,
            default_grid_size,
            default_thread_group_size,
        } = kernel
        {
            let shared_name = format!("shared_{}", candidate.ptr_var);

            // 新しいKernelボディを構築
            // 1. SharedAlloc
            // 2. 協調ロード（各スレッドが1要素をロード）
            // 3. Barrier (プリロード後)
            // 4. 元のボディ（LoadをSharedLoadに置換）

            let shared_alloc = AstNode::SharedAlloc {
                name: shared_name.clone(),
                dtype: candidate.dtype.clone(),
                size: Box::new(AstNode::Const(Literal::I64(candidate.shared_size))),
            };

            // 協調ロード: shared[local_id] = global[offset + local_id]
            let preload = self.generate_preload(
                &candidate.ptr_var,
                &shared_name,
                &candidate.parallel_var,
                &candidate.dtype,
            );

            let barrier = AstNode::Barrier;

            // 元のボディをSharedLoadに置換
            let transformed_body = Self::replace_load_with_shared_load(
                body,
                &candidate.ptr_var,
                &shared_name,
                &candidate.parallel_var,
                &candidate.dtype,
            );

            // 新しいボディを組み立て
            let new_body = AstNode::Block {
                statements: vec![shared_alloc, preload, barrier, transformed_body],
                scope: Box::new(Scope::new()),
            };

            Some(AstNode::Kernel {
                name: name.clone(),
                params: params.clone(),
                return_type: return_type.clone(),
                body: Box::new(new_body),
                default_grid_size: default_grid_size.clone(),
                default_thread_group_size: default_thread_group_size.clone(),
            })
        } else {
            None
        }
    }

    /// 協調ロードを生成
    fn generate_preload(
        &self,
        ptr_var: &str,
        shared_name: &str,
        parallel_var: &str,
        dtype: &DType,
    ) -> AstNode {
        // shared[local_id] = global[base + local_id]
        // ここではシンプルに、parallel_varの値をそのままオフセットとして使用
        AstNode::SharedStore {
            ptr: Box::new(AstNode::Var(shared_name.to_string())),
            offset: Box::new(AstNode::Var(parallel_var.to_string())),
            value: Box::new(AstNode::Load {
                ptr: Box::new(AstNode::Var(ptr_var.to_string())),
                offset: Box::new(AstNode::Var(parallel_var.to_string())),
                count: 1,
                dtype: dtype.clone(),
            }),
        }
    }

    /// Kernel内のGpuThread並列ループを検出して候補を提案
    fn suggest_for_kernel(&self, kernel: &AstNode) -> Vec<AstSuggestResult> {
        let mut results = Vec::new();

        if let AstNode::Kernel { body, .. } = kernel {
            self.find_parallel_loops_and_suggest(kernel, body, &mut results);
        }

        results
    }

    /// GpuThread並列ループを探索
    fn find_parallel_loops_and_suggest(
        &self,
        kernel: &AstNode,
        node: &AstNode,
        results: &mut Vec<AstSuggestResult>,
    ) {
        match node {
            AstNode::Range {
                var,
                start,
                stop,
                body,
                parallel,
                ..
            } => {
                if parallel.is_parallel && matches!(parallel.kind, ParallelKind::GpuThread) {
                    // GpuThreadループの場合、共有メモリ候補を探す
                    if let Some(parallel_size) = Self::get_loop_count(start, stop) {
                        let candidates =
                            self.find_shared_memory_candidates(var, parallel_size, body);

                        for candidate in candidates {
                            debug!(
                                "Suggesting shared memory for {} (reuse={}x)",
                                candidate.ptr_var, candidate.reuse_count
                            );

                            if let Some(optimized) =
                                self.apply_shared_memory_optimization(kernel, &candidate)
                            {
                                results.push(AstSuggestResult::with_description(
                                    optimized,
                                    "SharedMemorySuggester",
                                    format!(
                                        "共有メモリ最適化: {} を共有メモリにプリロード (再利用{}回)",
                                        candidate.ptr_var, candidate.reuse_count
                                    ),
                                ));
                            }
                        }
                    }
                }

                // ネストしたループも探索
                self.find_parallel_loops_and_suggest(kernel, body, results);
            }
            AstNode::Block { statements, .. } => {
                for stmt in statements {
                    self.find_parallel_loops_and_suggest(kernel, stmt, results);
                }
            }
            AstNode::If {
                then_body,
                else_body,
                ..
            } => {
                self.find_parallel_loops_and_suggest(kernel, then_body, results);
                if let Some(else_b) = else_body {
                    self.find_parallel_loops_and_suggest(kernel, else_b, results);
                }
            }
            _ => {}
        }
    }

    /// Program全体から候補を提案
    fn suggest_for_program(&self, program: &AstNode) -> Vec<AstSuggestResult> {
        let mut results = Vec::new();

        if let AstNode::Program { functions, .. } = program {
            for func in functions {
                if let AstNode::Kernel { .. } = func {
                    results.extend(self.suggest_for_kernel(func));
                }
            }
        }

        results
    }
}

impl AstSuggester for SharedMemorySuggester {
    fn name(&self) -> &str {
        "SharedMemorySuggester"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        match ast {
            AstNode::Program { .. } => self.suggest_for_program(ast),
            AstNode::Kernel { .. } => self.suggest_for_kernel(ast),
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_kernel() -> AstNode {
        // テスト用のカーネル:
        // kernel test(a: *f32, b: *f32, c: *f32) {
        //   for local_id in 0..256 (GpuThread) {
        //     for k in 0..64 {
        //       c[local_id] += a[local_id] * b[k]
        //     }
        //   }
        // }
        let load_a = AstNode::Load {
            ptr: Box::new(AstNode::Var("a".to_string())),
            offset: Box::new(AstNode::Var("local_id".to_string())),
            count: 1,
            dtype: DType::F32,
        };
        let load_b = AstNode::Load {
            ptr: Box::new(AstNode::Var("b".to_string())),
            offset: Box::new(AstNode::Var("k".to_string())),
            count: 1,
            dtype: DType::F32,
        };
        let mul = AstNode::Mul(Box::new(load_a), Box::new(load_b));

        let inner_loop = AstNode::Range {
            var: "k".to_string(),
            start: Box::new(AstNode::Const(Literal::I64(0))),
            step: Box::new(AstNode::Const(Literal::I64(1))),
            stop: Box::new(AstNode::Const(Literal::I64(64))),
            body: Box::new(AstNode::AtomicAdd {
                ptr: Box::new(AstNode::Var("c".to_string())),
                offset: Box::new(AstNode::Var("local_id".to_string())),
                value: Box::new(mul),
                dtype: DType::F32,
            }),
            parallel: ParallelInfo::default(),
        };

        let parallel_loop = AstNode::Range {
            var: "local_id".to_string(),
            start: Box::new(AstNode::Const(Literal::I64(0))),
            step: Box::new(AstNode::Const(Literal::I64(1))),
            stop: Box::new(AstNode::Const(Literal::I64(256))),
            body: Box::new(inner_loop),
            parallel: ParallelInfo {
                is_parallel: true,
                kind: ParallelKind::GpuThread,
                reductions: vec![],
            },
        };

        let one = || Box::new(AstNode::Const(Literal::I64(1)));

        AstNode::Kernel {
            name: Some("test".to_string()),
            params: vec![
                VarDecl {
                    name: "a".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32), AddressSpace::Global),
                    kind: VarKind::Normal,
                    mutability: Mutability::Immutable,
                },
                VarDecl {
                    name: "b".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32), AddressSpace::Global),
                    kind: VarKind::Normal,
                    mutability: Mutability::Immutable,
                },
                VarDecl {
                    name: "c".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32), AddressSpace::Global),
                    kind: VarKind::Normal,
                    mutability: Mutability::Mutable,
                },
            ],
            return_type: DType::Tuple(vec![]),
            body: Box::new(parallel_loop),
            default_grid_size: [one(), one(), one()],
            default_thread_group_size: [one(), one(), one()],
        }
    }

    #[test]
    fn test_depends_on_var() {
        let expr = AstNode::Add(
            Box::new(AstNode::Var("x".to_string())),
            Box::new(AstNode::Const(Literal::I64(1))),
        );
        assert!(SharedMemorySuggester::depends_on_var(&expr, "x"));
        assert!(!SharedMemorySuggester::depends_on_var(&expr, "y"));
    }

    #[test]
    fn test_get_loop_count() {
        let start = AstNode::Const(Literal::I64(0));
        let stop = AstNode::Const(Literal::I64(64));
        assert_eq!(
            SharedMemorySuggester::get_loop_count(&start, &stop),
            Some(64)
        );
    }

    #[test]
    fn test_suggest_finds_candidate() {
        let kernel = create_test_kernel();
        let suggester = SharedMemorySuggester::new();
        let suggestions = suggester.suggest(&kernel);

        // a[local_id]は内側ループ変数kに依存しないので候補になるはず
        assert!(
            !suggestions.is_empty(),
            "Should find at least one shared memory candidate"
        );

        // 提案されたASTにSharedAllocが含まれていることを確認
        let first_suggestion = &suggestions[0].ast;
        let has_shared_alloc = contains_shared_alloc(first_suggestion);
        assert!(has_shared_alloc, "Suggested AST should contain SharedAlloc");
    }

    fn contains_shared_alloc(node: &AstNode) -> bool {
        match node {
            AstNode::SharedAlloc { .. } => true,
            _ => node.children().iter().any(|c| contains_shared_alloc(c)),
        }
    }
}
