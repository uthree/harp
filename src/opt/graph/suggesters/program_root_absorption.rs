//! ProgramRoot Absorption Suggester
//!
//! Kernel(Function)ノードをProgramRootノードに吸収するSuggester。
//! LoweringSuggesterで生成されたKernel(Function)ノードを順次ProgramRootに取り込み、
//! 最終的に ProgramRoot(Program)のみの構成を目指します。
//!
//! # 処理フロー
//! 1. ProgramRootの**直接の**子ノードからKernel(Function)を検出
//! 2. 吸収時にProgramにKernelを追加し、main関数を更新
//!
//! # 設計方針
//! - Viewノードは透過しない（ViewMergeSuggesterに委譲）
//! - ProgramRootの直接の子ノードのみを吸収対象とする
//! - `ProgramRoot -> View -> Kernel`パターンは、先にViewMergeSuggesterが
//!   `ProgramRoot -> Kernel[view適用]`に変換してからこのSuggesterで吸収する

use crate::ast::helper::{block, const_int, function, var};
use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, VarDecl, VarKind};
use crate::graph::ops::InputBufferMeta;
use crate::graph::ops::custom_placeholders as ph;
use crate::graph::{DType as GraphDType, Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// Kernel(Function)ノードをProgramRootに吸収するSuggester
pub struct ProgramRootAbsorptionSuggester;

/// カーネル情報（AST + 入力バッファ名のマッピング）
struct KernelInfo {
    /// カーネルのAST
    ast: AstNode,
    /// 入力バッファ名のリスト（カーネルパラメータ順）
    input_buffer_names: Vec<String>,
}

impl ProgramRootAbsorptionSuggester {
    pub fn new() -> Self {
        Self
    }

    /// ノードが出力 Buffer かどうかを判定
    fn is_output_buffer(node: &GraphNode) -> bool {
        matches!(&node.op, GraphOp::Buffer { name } if name.starts_with("output"))
    }

    /// src から入力ノードのみを取得（出力 Buffer を除外）
    fn get_input_nodes(src: &[GraphNode]) -> Vec<&GraphNode> {
        src.iter().filter(|n| !Self::is_output_buffer(n)).collect()
    }

    /// ProgramRootノードから吸収可能なKernel(Function)を検出
    ///
    /// ProgramRootの子ノードからKernel(Function)を探します。
    /// Viewノードは1レベルのみ透過します（ViewMergeSuggesterと連携）。
    ///
    /// # 設計方針
    /// - 直接の子がKernel(Function)の場合は即座に吸収
    /// - 直接の子がViewの場合は、Viewの先にあるKernelを探す（1レベルのみ）
    /// - 深いネストのViewは透過しない（ViewMergeSuggesterに委譲）
    fn find_absorbable_customs(&self, graph: &Graph) -> Vec<GraphNode> {
        let sink = match graph.program_root() {
            Some(s) => s,
            None => return vec![],
        };

        let mut absorbable = Vec::new();
        let mut visited = HashSet::new();

        // ProgramRootの直接の子ノードをチェック
        for src in &sink.src {
            Self::collect_customs_shallow(src, &mut absorbable, &mut visited);
        }

        absorbable
    }

    /// 浅い探索でKernel(Function/Kernel/Program)を収集
    ///
    /// Viewノードは1レベルのみ透過し、その先のKernelを収集します。
    /// 複数のKernelがチェーンしている場合、依存順（リーフ優先）で収集します。
    ///
    /// # Note
    /// - Kernel(Function): Sequential戦略で生成（LoweringSuggester）
    /// - Kernel(Kernel): Parallel戦略で生成（LoweringSuggester）
    /// - Kernel(Program): KernelMergeSuggesterで作成
    fn collect_customs_shallow(
        node: &GraphNode,
        result: &mut Vec<GraphNode>,
        visited: &mut HashSet<*const GraphNodeData>,
    ) {
        let ptr = node.as_ptr();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        match &node.op {
            // Kernel(Function), Kernel(Kernel), またはKernel(Program)を検出
            GraphOp::Kernel {
                ast: AstNode::Function { .. } | AstNode::Kernel { .. } | AstNode::Program { .. },
                ..
            } => {
                // 先にこのKernelのsrcを探索（依存関係順）
                for src in &node.src {
                    Self::collect_customs_shallow(src, result, visited);
                }
                // 重複チェック
                if !result.iter().any(|n| n.as_ptr() == ptr) {
                    log::debug!("ProgramRootAbsorption: found absorbable Kernel");
                    result.push(node.clone());
                }
            }
            // Viewノードは透過（srcを探索）
            GraphOp::View(_) => {
                for src in &node.src {
                    Self::collect_customs_shallow(src, result, visited);
                }
            }
            // Buffer/Const等は無視
            _ => {}
        }
    }

    /// Kernel(Function)またはKernel(Program)をProgramRootに吸収
    fn absorb_custom(&self, graph: &Graph, custom_node: &GraphNode) -> Option<Graph> {
        let sink = graph.program_root()?;

        // KernelのASTを取得
        let custom_ast = match &custom_node.op {
            GraphOp::Kernel { ast, .. } => ast,
            _ => return None,
        };

        // 現在のProgramRootからProgram情報を取得
        let (mut program_functions, entry_point, output_names, execution_order) = match &sink.op {
            GraphOp::ProgramRoot { ast, outputs } => {
                if let AstNode::Program {
                    functions,
                    entry_point,
                    execution_order,
                } = ast
                {
                    (
                        functions.clone(),
                        entry_point.clone(),
                        outputs.clone(),
                        execution_order.clone(),
                    )
                } else {
                    (vec![], "harp_main".to_string(), outputs.clone(), vec![])
                }
            }
            _ => return None,
        };

        // 使用済み名前を収集
        let mut used_names: HashSet<String> = program_functions
            .iter()
            .filter_map(|f| match f {
                AstNode::Kernel { name: Some(n), .. } => Some(n.clone()),
                AstNode::Function { name: Some(n), .. } => Some(n.clone()),
                _ => None,
            })
            .collect();

        // ASTの種類に応じて処理を分岐
        let kernel_infos = match custom_ast {
            AstNode::Function { .. } => {
                // Kernel(Function)をKernelに変換して追加
                let kernel_info = self.create_kernel_from_function(
                    graph,
                    custom_node,
                    custom_ast,
                    &mut used_names,
                );
                program_functions.push(kernel_info.ast.clone());
                vec![kernel_info]
            }
            AstNode::Kernel { name, body, .. } => {
                // Kernel(Kernel)はすでにカーネル形式なので、直接追加
                // 名前の重複を避ける
                let base_name = name.clone().unwrap_or_else(|| "kernel".to_string());
                let unique_name = Self::make_unique_name(&base_name, &used_names);
                used_names.insert(unique_name.clone());

                // 入力バッファ名を取得
                let input_buffer_names = self.get_input_buffer_names(custom_node);

                // 入力形状を取得してshape変数を置換
                let input_shape = self.get_input_shape(custom_node);
                let mut shape_substitutions: HashMap<String, AstNode> = HashMap::new();
                for (axis, expr) in input_shape.iter().enumerate() {
                    let placeholder_name = ph::shape(axis);
                    let ast_expr: AstNode = expr.clone().into();
                    shape_substitutions.insert(placeholder_name, ast_expr);
                }
                let substituted_body = body.substitute_vars(&shape_substitutions);

                // 名前とbodyを更新したカーネルを作成
                let updated_kernel = match custom_ast {
                    AstNode::Kernel {
                        params,
                        return_type,
                        default_grid_size,
                        default_thread_group_size,
                        ..
                    } => AstNode::Kernel {
                        name: Some(unique_name.clone()),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(substituted_body),
                        default_grid_size: default_grid_size.clone(),
                        default_thread_group_size: default_thread_group_size.clone(),
                    },
                    _ => custom_ast.clone(),
                };
                program_functions.push(updated_kernel.clone());

                vec![KernelInfo {
                    ast: updated_kernel,
                    input_buffer_names,
                }]
            }
            AstNode::Program { functions, .. } => {
                // Kernel(Program)の全関数をProgramRootのProgramに追加
                // harp_main以外の関数を追加（harp_mainは後で再生成）
                let mut infos = Vec::new();
                for func in functions {
                    match func {
                        AstNode::Kernel { name: Some(n), .. }
                        | AstNode::Function { name: Some(n), .. }
                            if n != "harp_main" =>
                        {
                            // 名前の重複を避ける
                            let unique_name = Self::make_unique_name(n, &used_names);
                            used_names.insert(unique_name.clone());

                            // 関数/カーネルを名前を変更してコピー
                            let renamed = Self::rename_function(func, &unique_name);
                            program_functions.push(renamed.clone());

                            // KernelInfoを生成（input_buffer_namesはProgramから取得できないので空）
                            infos.push(KernelInfo {
                                ast: renamed,
                                input_buffer_names: vec![],
                            });
                        }
                        _ => {}
                    }
                }
                infos
            }
            _ => return None,
        };

        // 新しいProgramRootのsrcを構築（custom_nodeを削除し、その入力を直接接続）
        let new_src = self.rebuild_sink_src(sink, custom_node);

        // main関数を生成/更新（カーネル引数マッピング情報を渡す）
        let main_fn = self.generate_main_function(
            graph,
            &new_src,
            &program_functions,
            &output_names,
            &kernel_infos,
        );

        // 既存のmain関数を削除して新しいものを追加
        program_functions
            .retain(|f| !matches!(f, AstNode::Function { name: Some(n), .. } if n == "harp_main"));
        program_functions.push(main_fn);

        // 新しいProgramを作成
        let new_program = AstNode::Program {
            functions: program_functions,
            entry_point,
            execution_order,
        };

        // 新しいProgramRootノードを作成
        let new_sink = GraphNode::new(
            sink.dtype.clone(),
            GraphOp::ProgramRoot {
                ast: new_program,
                outputs: output_names,
            },
            new_src,
            sink.view.clone(),
        );

        // 新しいグラフを構築
        Some(self.rebuild_graph_with_sink(graph, new_sink))
    }

    /// 関数/カーネルの名前を変更
    fn rename_function(func: &AstNode, new_name: &str) -> AstNode {
        match func {
            AstNode::Kernel {
                params,
                return_type,
                body,
                default_grid_size,
                default_thread_group_size,
                ..
            } => AstNode::Kernel {
                name: Some(new_name.to_string()),
                params: params.clone(),
                return_type: return_type.clone(),
                body: body.clone(),
                default_grid_size: default_grid_size.clone(),
                default_thread_group_size: default_thread_group_size.clone(),
            },
            AstNode::Function {
                params,
                return_type,
                body,
                ..
            } => AstNode::Function {
                name: Some(new_name.to_string()),
                params: params.clone(),
                return_type: return_type.clone(),
                body: body.clone(),
            },
            other => other.clone(),
        }
    }

    /// ProgramRootのsrcを再構築（吸収されたKernelノードを出力バッファで置き換え）
    ///
    /// BufferAbsorption適用後: Kernel.src = [output_buffer]
    /// BufferAbsorption適用前: Kernel.src = [input0, input1, ..., output_buffer]
    fn rebuild_sink_src(&self, sink: &GraphNode, absorbed_custom: &GraphNode) -> Vec<GraphNode> {
        let mut new_src = Vec::new();
        let mut node_map: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();

        // BufferAbsorption適用済みかどうかをチェック
        let has_input_buffers = matches!(
            &absorbed_custom.op,
            GraphOp::Kernel {
                input_buffers: Some(_),
                ..
            }
        );

        // 吸収されたノードの置き換え先を決定
        // - BufferAbsorption後: srcの全体（output_bufferのみ）を使用
        // - BufferAbsorption前: 入力ノード（出力バッファを除く）の最初を使用
        let replacement_nodes: Vec<GraphNode> = if has_input_buffers {
            // BufferAbsorption後: srcをそのまま使用（output_bufferのみが含まれる）
            absorbed_custom.src.clone()
        } else {
            // BufferAbsorption前（フォールバック）: 入力ノードを取得
            Self::get_input_nodes(&absorbed_custom.src)
                .iter()
                .map(|n| (*n).clone())
                .collect()
        };

        fn rebuild_node(
            node: &GraphNode,
            absorbed_ptr: *const GraphNodeData,
            replacement_nodes: &[GraphNode],
            node_map: &mut HashMap<*const GraphNodeData, GraphNode>,
            visited: &mut HashSet<*const GraphNodeData>,
        ) -> Option<GraphNode> {
            let ptr = node.as_ptr();

            // キャッシュをチェック
            if let Some(mapped) = node_map.get(&ptr) {
                return Some(mapped.clone());
            }

            // 吸収されたノードの場合、置き換えノードを返す
            // BufferAbsorption後: output_bufferで置き換え
            // BufferAbsorption前: 最初の入力ノードで置き換え
            if ptr == absorbed_ptr {
                if let Some(first) = replacement_nodes.first() {
                    node_map.insert(ptr, first.clone());
                    return Some(first.clone());
                }
                // 置き換えノードがない場合はNoneを返す（このノードはスキップされる）
                return None;
            }

            // Bufferノードはそのまま返す
            if matches!(node.op, GraphOp::Buffer { .. }) {
                return Some(node.clone());
            }

            // 循環参照を避ける
            if visited.contains(&ptr) {
                return Some(node.clone());
            }
            visited.insert(ptr);

            // 再帰的に子ノードを再構築
            let new_children: Vec<GraphNode> = node
                .src
                .iter()
                .filter_map(|src| {
                    rebuild_node(src, absorbed_ptr, replacement_nodes, node_map, visited)
                })
                .collect();

            // 子ノードに変更がなければ元のノードを返す
            let children_changed = new_children.len() != node.src.len()
                || new_children
                    .iter()
                    .zip(&node.src)
                    .any(|(a, b)| a.as_ptr() != b.as_ptr());

            if !children_changed {
                return Some(node.clone());
            }

            // 新しいノードを作成
            let new_node = GraphNode::new(
                node.dtype.clone(),
                node.op.clone(),
                new_children,
                node.view.clone(),
            );
            node_map.insert(ptr, new_node.clone());
            Some(new_node)
        }

        let absorbed_ptr = absorbed_custom.as_ptr();
        let mut visited = HashSet::new();
        let mut added_ptrs: HashSet<*const GraphNodeData> = HashSet::new();

        // ProgramRootのsrcを再構築（重複を排除）
        for src in &sink.src {
            if let Some(rebuilt) = rebuild_node(
                src,
                absorbed_ptr,
                &replacement_nodes,
                &mut node_map,
                &mut visited,
            ) {
                // 同じノード（ポインタ）が既に追加されている場合はスキップ
                let rebuilt_ptr = rebuilt.as_ptr();
                if !added_ptrs.contains(&rebuilt_ptr) {
                    added_ptrs.insert(rebuilt_ptr);
                    new_src.push(rebuilt);
                }
            }
        }

        new_src
    }

    /// Kernelノードから入力バッファ名を取得
    fn get_input_buffer_names(&self, node: &GraphNode) -> Vec<String> {
        // BufferAbsorptionで取り込まれたinput_buffersを優先的に使用
        if let GraphOp::Kernel {
            input_buffers: Some(buffers),
            ..
        } = &node.op
        {
            return buffers.iter().map(|m| m.name.clone()).collect();
        }

        // BufferAbsorption前（フォールバック）: srcから入力を取得
        Self::get_input_nodes(&node.src)
            .iter()
            .map(|src| match &src.op {
                GraphOp::Buffer { name } => name.clone(),
                _ => format!("intermediate_{}", src.as_ptr() as usize),
            })
            .collect()
    }

    /// Kernelノードから入力形状を取得
    fn get_input_shape(&self, node: &GraphNode) -> Vec<crate::graph::shape::Expr> {
        // BufferAbsorptionで取り込まれたinput_buffersを優先的に使用
        if let GraphOp::Kernel {
            input_buffers: Some(buffers),
            ..
        } = &node.op
            && let Some(first_buffer) = buffers.first()
        {
            return first_buffer.shape.clone();
        }

        // BufferAbsorption前（フォールバック）: srcから入力を取得
        let input_nodes = Self::get_input_nodes(&node.src);
        if !input_nodes.is_empty() {
            return input_nodes[0].view.shape().to_vec();
        }

        // フォールバック: ノード自体の形状を使用
        node.view.shape().to_vec()
    }

    /// Kernel関数を作成
    fn create_kernel_from_function(
        &self,
        _graph: &Graph,
        node: &GraphNode,
        func_ast: &AstNode,
        used_names: &mut HashSet<String>,
    ) -> KernelInfo {
        // BufferAbsorptionで取り込まれたinput_buffersを優先的に使用
        let (input_buffer_metas, input_shape) = if let GraphOp::Kernel {
            input_buffers: Some(buffers),
            ..
        } = &node.op
        {
            // BufferAbsorption後: input_buffersに形状を含むメタデータがある
            // 入力形状は最初のバッファから取得（Shape変数のためカーネルパラメータに必要）
            let shape = if let Some(first_buffer) = buffers.first() {
                first_buffer.shape.clone()
            } else {
                node.view.shape().to_vec()
            };
            (buffers.clone(), shape)
        } else {
            // BufferAbsorption前（フォールバック）: srcから入力を取得
            let input_nodes = Self::get_input_nodes(&node.src);
            let shape = if !input_nodes.is_empty() {
                input_nodes[0].view.shape().to_vec()
            } else {
                node.view.shape().to_vec()
            };
            // srcからInputBufferMetaを構築
            let metas: Vec<InputBufferMeta> = input_nodes
                .iter()
                .map(|src| {
                    let name = match &src.op {
                        GraphOp::Buffer { name } => name.clone(),
                        _ => format!("intermediate_{}", src.as_ptr() as usize),
                    };
                    InputBufferMeta {
                        name,
                        dtype: src.dtype.clone(),
                        shape: src.view.shape().to_vec(),
                    }
                })
                .collect();
            (metas, shape)
        };

        // 入力バッファ名を収集（カーネルパラメータ順）
        let input_buffer_names: Vec<String> =
            input_buffer_metas.iter().map(|m| m.name.clone()).collect();

        // パラメータを生成
        let mut params = Vec::new();

        // 入力バッファー - input_buffersの順序で生成（bodyのプレースホルダーと一致させる）
        for (i, meta) in input_buffer_metas.iter().enumerate() {
            params.push(VarDecl {
                name: ph::input(i),
                dtype: Self::graph_dtype_to_ast_ptr(&meta.dtype),
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            });
        }

        // 出力バッファー
        params.push(VarDecl {
            name: ph::OUTPUT.to_string(),
            dtype: Self::graph_dtype_to_ast_ptr(&node.dtype),
            mutability: Mutability::Mutable,
            kind: VarKind::Normal,
        });

        // Shape変数
        for expr in input_shape.iter() {
            if let crate::graph::shape::Expr::Var(var_name) = expr {
                params.push(VarDecl {
                    name: var_name.clone(),
                    dtype: AstDType::Int,
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                });
            }
        }

        // ボディを取得
        let body = if let AstNode::Function { body, .. } = func_ast {
            let mut shape_substitutions: HashMap<String, AstNode> = HashMap::new();
            for (axis, expr) in input_shape.iter().enumerate() {
                let placeholder_name = ph::shape(axis);
                let ast_expr: AstNode = expr.clone().into();
                shape_substitutions.insert(placeholder_name, ast_expr);
            }
            body.substitute_vars(&shape_substitutions)
        } else {
            AstNode::Block {
                statements: vec![],
                scope: Box::new(Scope::new()),
            }
        };

        // 元のFunction名を取得し、重複を避けてユニークな名前を生成
        let base_name = if let AstNode::Function { name: Some(n), .. } = func_ast {
            n.clone()
        } else {
            "kernel".to_string()
        };
        let kernel_name = Self::make_unique_name(&base_name, used_names);
        used_names.insert(kernel_name.clone());

        // デフォルト1D dispatch設定（64スレッド/グループ, 1グリッド）
        let one = const_int(1);
        let ast = AstNode::Kernel {
            name: Some(kernel_name),
            params,
            return_type: AstDType::Tuple(vec![]),
            body: Box::new(body),
            default_grid_size: [
                Box::new(one.clone()),
                Box::new(one.clone()),
                Box::new(one.clone()),
            ],
            default_thread_group_size: [
                Box::new(const_int(64)),
                Box::new(one.clone()),
                Box::new(one),
            ],
        };

        KernelInfo {
            ast,
            input_buffer_names,
        }
    }

    /// 重複を避けてユニークな名前を生成
    fn make_unique_name(base_name: &str, used_names: &HashSet<String>) -> String {
        if !used_names.contains(base_name) {
            return base_name.to_string();
        }

        let mut counter = 1;
        loop {
            let candidate = format!("{}__{}", base_name, counter);
            if !used_names.contains(&candidate) {
                return candidate;
            }
            counter += 1;
        }
    }

    /// main関数を生成
    fn generate_main_function(
        &self,
        graph: &Graph,
        _sink_src: &[GraphNode],
        kernels: &[AstNode],
        output_names: &[String],
        kernel_infos: &[KernelInfo],
    ) -> AstNode {
        let mut params: Vec<VarDecl> = Vec::new();
        let mut param_names: HashSet<String> = HashSet::new();

        // 入力バッファーのパラメータを追加（メタデータから、ソート順）
        let mut sorted_inputs: Vec<_> =
            graph.input_metas().iter().map(|m| m.name.clone()).collect();
        sorted_inputs.sort();

        // バッファ名からmainパラメータ名へのマッピングを構築
        let mut buffer_to_main_param: HashMap<String, String> = HashMap::new();
        for (i, name) in sorted_inputs.iter().enumerate() {
            if let Some(meta) = graph.input_metas().iter().find(|m| &m.name == name) {
                let param_name = format!("input{}", i);
                if !param_names.contains(&param_name) {
                    params.push(VarDecl {
                        name: param_name.clone(),
                        dtype: Self::graph_dtype_to_ast_ptr(&meta.dtype),
                        mutability: Mutability::Immutable,
                        kind: VarKind::Normal,
                    });
                    param_names.insert(param_name.clone());
                    buffer_to_main_param.insert(name.clone(), param_name);
                }
            }
        }

        // 出力バッファーのパラメータを追加（メタデータから）
        for (i, name) in output_names.iter().enumerate() {
            if let Some(meta) = graph.output_metas().iter().find(|m| &m.name == name) {
                let param_name = format!("output{}", i);
                if !param_names.contains(&param_name) {
                    params.push(VarDecl {
                        name: param_name.clone(),
                        dtype: Self::graph_dtype_to_ast_ptr(&meta.dtype),
                        mutability: Mutability::Mutable,
                        kind: VarKind::Normal,
                    });
                    param_names.insert(param_name);
                }
            }
        }

        // main関数のbody
        let mut statements: Vec<AstNode> = Vec::new();
        let scope = Scope::new();

        // 各カーネルを呼び出す
        for kernel in kernels {
            let kernel_name = Self::get_kernel_name(kernel);

            // main関数自体はスキップ
            if kernel_name == "harp_main" {
                continue;
            }

            // カーネル情報からこのカーネルの入力バッファ名を取得
            let kernel_info = kernel_infos.iter().find(|k| {
                if let AstNode::Kernel { name: Some(n), .. } = &k.ast {
                    n == &kernel_name
                } else {
                    false
                }
            });

            let mut args: Vec<AstNode> = Vec::new();

            // カーネルの入力パラメータ数を取得
            let input_count = Self::get_kernel_input_count(kernel);

            if let Some(info) = kernel_info {
                // カーネルパラメータ順（Kernel.src順）でmainパラメータにマッピング
                for buffer_name in &info.input_buffer_names {
                    if let Some(main_param) = buffer_to_main_param.get(buffer_name) {
                        args.push(var(main_param.clone()));
                    }
                    // 中間結果など、input_metasにないバッファはスキップ
                }
            }

            // KernelInfoがない場合、またはマッピングで十分な引数が生成されなかった場合はフォールバック
            if args.len() < input_count {
                args.clear();
                // カーネルのパラメータ数に合わせてソート順で入力を渡す
                for i in 0..input_count.min(sorted_inputs.len()) {
                    args.push(var(format!("input{}", i)));
                }
            }

            // 出力バッファー
            if !output_names.is_empty() {
                args.push(var("output0"));
            }

            statements.push(AstNode::Call {
                name: kernel_name,
                args,
            });

            // バリアを挿入（最後のカーネル以外）
            statements.push(AstNode::Barrier);
        }

        // 最後のバリアを削除
        if statements.last() == Some(&AstNode::Barrier) {
            statements.pop();
        }

        let body = block(statements, scope);

        function(Some("harp_main"), params, AstDType::Tuple(vec![]), body)
    }

    /// カーネル名を取得
    fn get_kernel_name(kernel: &AstNode) -> String {
        match kernel {
            AstNode::Kernel { name: Some(n), .. } => n.clone(),
            AstNode::Function { name: Some(n), .. } => n.clone(),
            _ => "unknown".to_string(),
        }
    }

    /// カーネルの入力パラメータ数を取得（出力パラメータを除く）
    fn get_kernel_input_count(kernel: &AstNode) -> usize {
        match kernel {
            AstNode::Kernel { params, .. } | AstNode::Function { params, .. } => {
                // 最後のパラメータは出力バッファなので除く
                if params.is_empty() {
                    0
                } else {
                    params.len() - 1
                }
            }
            _ => 0,
        }
    }

    /// 新しいProgramRootノードでグラフを再構築
    fn rebuild_graph_with_sink(&self, graph: &Graph, new_sink: GraphNode) -> Graph {
        let mut new_graph = Graph::new();

        // 入力・出力メタデータをコピー
        new_graph.copy_input_metas_from(graph);
        new_graph.copy_output_metas_from(graph);

        // 新しいProgramRootをセット
        new_graph.set_program_root(new_sink);

        // shape変数のデフォルト値をコピー
        for (name, value) in graph.shape_var_defaults() {
            new_graph.set_shape_var_default(name.clone(), *value);
        }

        new_graph
    }

    fn graph_dtype_to_ast(dtype: &GraphDType) -> AstDType {
        match dtype {
            GraphDType::Bool => AstDType::Bool,
            GraphDType::I32 => AstDType::Int,
            GraphDType::F32 => AstDType::F32,
            GraphDType::Complex => AstDType::F32,
            GraphDType::Unknown => AstDType::F32,
        }
    }

    fn graph_dtype_to_ast_ptr(dtype: &GraphDType) -> AstDType {
        AstDType::Ptr(Box::new(Self::graph_dtype_to_ast(dtype)))
    }
}

impl Default for ProgramRootAbsorptionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for ProgramRootAbsorptionSuggester {
    fn name(&self) -> &'static str {
        "ProgramRootAbsorption"
    }

    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        // ProgramRootがない場合は何もしない
        if graph.program_root().is_none() {
            log::debug!("ProgramRootAbsorptionSuggester: no ProgramRoot node found");
            return vec![];
        }

        let absorbable = self.find_absorbable_customs(graph);

        log::debug!(
            "ProgramRootAbsorptionSuggester: found {} absorbable Kernel nodes",
            absorbable.len()
        );

        let mut suggestions = Vec::new();

        for custom in absorbable {
            if let Some(new_graph) = self.absorb_custom(graph, &custom) {
                log::debug!("ProgramRootAbsorptionSuggester: absorbed Kernel");
                suggestions.push(new_graph);
            }
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DType;

    #[test]
    fn test_sink_absorption_basic() {
        let suggester = ProgramRootAbsorptionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        let c = a + b;
        graph.output("c", c);

        // ProgramRootが存在することを確認
        assert!(graph.program_root().is_some());

        // まだKernelノードがないので、吸収対象はない
        let absorbable = suggester.find_absorbable_customs(&graph);
        assert!(absorbable.is_empty());
    }

    #[test]
    fn test_find_absorbable_with_custom() {
        use crate::ast::helper::wildcard;

        let suggester = ProgramRootAbsorptionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        // Kernel(Function)を作成
        let custom = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        graph.output("c", custom);

        eprintln!("ProgramRoot exists: {:?}", graph.program_root().is_some());
        if let Some(sink) = graph.program_root() {
            eprintln!("ProgramRoot src count: {}", sink.src.len());
            for (i, src) in sink.src.iter().enumerate() {
                eprintln!("  src[{}]: {:?}", i, src.op);
            }
        }

        // Kernel(Function)が吸収対象として検出される
        let absorbable = suggester.find_absorbable_customs(&graph);
        eprintln!("Found {} absorbable Kernel nodes", absorbable.len());
        for (i, node) in absorbable.iter().enumerate() {
            eprintln!("  absorbable[{}]: {:?}", i, node.op);
        }
        assert_eq!(absorbable.len(), 1);
    }

    #[test]
    fn test_absorb_custom_into_sink() {
        use crate::ast::helper::wildcard;

        let suggester = ProgramRootAbsorptionSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        // Kernel(Function)を作成
        let custom = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        graph.output("c", custom);

        // suggest()を呼び出し
        let suggestions = suggester.suggest(&graph);
        eprintln!(
            "Got {} suggestions from ProgramRootAbsorption",
            suggestions.len()
        );

        // 1つの提案があるはず
        assert!(
            !suggestions.is_empty(),
            "Should have at least one suggestion"
        );

        let new_graph = &suggestions[0];
        eprintln!(
            "New graph sink exists: {:?}",
            new_graph.program_root().is_some()
        );

        if let Some(sink) = new_graph.program_root()
            && let GraphOp::ProgramRoot { ast, outputs } = &sink.op
        {
            eprintln!("Outputs: {:?}", outputs);
            if let crate::ast::AstNode::Program { functions, .. } = ast {
                eprintln!("Program has {} functions", functions.len());
                for (i, func) in functions.iter().enumerate() {
                    let name = match func {
                        crate::ast::AstNode::Kernel { name, .. } => name.clone(),
                        crate::ast::AstNode::Function { name, .. } => name.clone(),
                        _ => None,
                    };
                    eprintln!("  function[{}]: {:?}", i, name);
                }
            }
        }
    }

    #[test]
    fn test_lowering_then_sink_absorption() {
        use crate::opt::graph::suggesters::LoweringSuggester;

        let lowering = LoweringSuggester::new();
        let sink_absorber = ProgramRootAbsorptionSuggester::new();

        // シンプルなElementwise演算グラフ
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);
        let c = a + b;
        graph.output("c", c);

        eprintln!("=== Initial Graph ===");
        eprintln!("ProgramRoot exists: {:?}", graph.program_root().is_some());

        // LoweringSuggesterを適用
        let lowered = lowering.suggest(&graph);
        eprintln!("\n=== After Lowering ===");
        eprintln!("Got {} suggestions from Lowering", lowered.len());

        assert!(!lowered.is_empty(), "Lowering should produce suggestions");
        let lowered_graph = &lowered[0];

        eprintln!(
            "Lowered graph sink exists: {:?}",
            lowered_graph.program_root().is_some()
        );
        if let Some(sink) = lowered_graph.program_root() {
            eprintln!("ProgramRoot src count: {}", sink.src.len());
            for (i, src) in sink.src.iter().enumerate() {
                let op_name = match &src.op {
                    GraphOp::Kernel { .. } => "Kernel".to_string(),
                    GraphOp::Buffer { name } => format!("Buffer({})", name),
                    GraphOp::Elementwise { op, .. } => format!("Elementwise({:?})", op),
                    _ => format!("{:?}", src.op),
                };
                eprintln!("  src[{}]: {}", i, op_name);
            }
        }

        // ProgramRootAbsorptionを適用
        let absorbed = sink_absorber.suggest(lowered_graph);
        eprintln!("\n=== After ProgramRootAbsorption ===");
        eprintln!(
            "Got {} suggestions from ProgramRootAbsorption",
            absorbed.len()
        );

        if absorbed.is_empty() {
            // 吸収対象が見つからない場合、理由を調べる
            let absorbable = sink_absorber.find_absorbable_customs(lowered_graph);
            eprintln!("Absorbable nodes: {}", absorbable.len());
        }

        assert!(
            !absorbed.is_empty(),
            "ProgramRootAbsorption should produce suggestions"
        );

        // 吸収後のProgramRootの状態を確認
        let absorbed_graph = &absorbed[0];
        eprintln!("\n=== Absorbed Graph Details ===");
        eprintln!(
            "Input metas: {:?}",
            absorbed_graph
                .input_metas()
                .iter()
                .map(|m| &m.name)
                .collect::<Vec<_>>()
        );
        if let Some(sink) = absorbed_graph.program_root() {
            eprintln!("ProgramRoot src count: {}", sink.src.len());
            for (i, src) in sink.src.iter().enumerate() {
                let op_name = match &src.op {
                    GraphOp::Kernel { .. } => "Kernel".to_string(),
                    GraphOp::Buffer { name } => format!("Buffer({})", name),
                    _ => format!("{:?}", src.op),
                };
                eprintln!("  src[{}]: {}", i, op_name);
            }
            if let GraphOp::ProgramRoot { ast, outputs } = &sink.op {
                eprintln!("Outputs: {:?}", outputs);
                if let AstNode::Program { functions, .. } = ast {
                    eprintln!("Program functions: {}", functions.len());
                    for f in functions {
                        match f {
                            AstNode::Function { name, params, .. } => {
                                eprintln!("  Function {:?}: {} params", name, params.len());
                                for p in params {
                                    eprintln!("    - {}", p.name);
                                }
                            }
                            AstNode::Kernel { name, params, .. } => {
                                eprintln!("  Kernel {:?}: {} params", name, params.len());
                                for p in params {
                                    eprintln!("    - {}", p.name);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_full_optimization_with_beam_search() {
        use crate::backend::pipeline::{MultiPhaseConfig, create_multi_phase_optimizer};
        use crate::opt::graph::GraphOptimizer;

        // 複数の演算を含むグラフ: reduce(a + b) + c
        // Reduceがあると融合されずに2つのKernelノードになる
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 5]);
        let b = graph.input("b", DType::F32, vec![10, 5]);
        let c = graph.input("c", DType::F32, vec![10]);
        let sum = &a + &b;
        let reduced = sum.reduce_sum(1); // [10]
        let result = &reduced + &c;
        graph.output("result", result);

        eprintln!("=== Initial Graph ===");
        eprintln!("ProgramRoot exists: {:?}", graph.program_root().is_some());

        // マルチフェーズ最適化を使用
        let config = MultiPhaseConfig::new()
            .with_beam_width(4)
            .with_max_steps(20)
            .with_progress(false)
            .with_collect_logs(false);

        let optimizer = create_multi_phase_optimizer(config);
        let (optimized, history) = optimizer.optimize_with_history(graph);

        eprintln!("\n=== Optimization History ===");
        for snapshot in history.snapshots() {
            let sink_info = if let Some(sink) = snapshot.graph.program_root() {
                if let GraphOp::ProgramRoot { ast, outputs } = &sink.op {
                    let func_count = if let crate::ast::AstNode::Program { functions, .. } = ast {
                        functions.len()
                    } else {
                        0
                    };
                    format!("ProgramRoot(outputs={:?}, funcs={})", outputs, func_count)
                } else {
                    "Unknown".to_string()
                }
            } else {
                "No ProgramRoot".to_string()
            };
            eprintln!(
                "Step {}: {} - cost={:.2} - {}",
                snapshot.step, snapshot.description, snapshot.cost, sink_info
            );
        }

        eprintln!("\n=== Final Graph ===");
        eprintln!(
            "ProgramRoot exists: {:?}",
            optimized.program_root().is_some()
        );

        if let Some(sink) = optimized.program_root() {
            eprintln!("ProgramRoot src count: {}", sink.src.len());
            for (i, src) in sink.src.iter().enumerate() {
                let op_name = match &src.op {
                    GraphOp::Buffer { name } => format!("Buffer({})", name),
                    GraphOp::Kernel { .. } => "Kernel".to_string(),
                    _ => format!("{:?}", std::mem::discriminant(&src.op)),
                };
                eprintln!("  src[{}]: {}", i, op_name);
            }
            if let GraphOp::ProgramRoot { ast, outputs } = &sink.op {
                eprintln!("Outputs: {:?}", outputs);
                if let crate::ast::AstNode::Program { functions, .. } = ast {
                    eprintln!("Program has {} functions", functions.len());
                    for (i, func) in functions.iter().enumerate() {
                        let name = match func {
                            crate::ast::AstNode::Kernel { name, .. } => name.clone(),
                            crate::ast::AstNode::Function { name, .. } => name.clone(),
                            _ => None,
                        };
                        eprintln!("  function[{}]: {:?}", i, name);
                    }
                    // 最終的にカーネルとmain関数があることを確認
                    assert!(
                        functions.len() >= 2,
                        "Should have at least kernel and harp_main"
                    );
                }
            }
        } else {
            panic!("Optimized graph should have ProgramRoot node");
        }
    }

    #[test]
    fn test_multiple_outputs_absorption() {
        use crate::ast::helper::wildcard;
        use crate::backend::pipeline::{MultiPhaseConfig, create_multi_phase_optimizer};
        use crate::opt::graph::GraphOptimizer;

        // 複数出力のグラフを作成: x = a + b, y = x + 1
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        // x = a + b (Kernel)
        let x = a.custom_elementwise_binary(b, wildcard("0") + wildcard("1"));
        // y = x + 1 (Kernel) - 使用できるcustom_elementwiseがないので、通常のelementwise演算を使用
        let y = &x + 1.0f32;

        graph.output("x", x.clone());
        graph.output("y", y);

        eprintln!("\n=== Multiple Outputs Test ===");
        eprintln!("ProgramRoot exists: {:?}", graph.program_root().is_some());
        if let Some(sink) = graph.program_root() {
            eprintln!("ProgramRoot src count: {}", sink.src.len());
            if let GraphOp::ProgramRoot { outputs, .. } = &sink.op {
                eprintln!("Outputs: {:?}", outputs);
            }
        }

        // ProgramRootAbsorptionで吸収可能なノードを検出
        let suggester = ProgramRootAbsorptionSuggester::new();
        let absorbable = suggester.find_absorbable_customs(&graph);
        eprintln!("\nAbsorbable Kernel nodes: {}", absorbable.len());
        for (i, node) in absorbable.iter().enumerate() {
            if let GraphOp::Kernel { ast, .. } = &node.op {
                let name = match ast {
                    crate::ast::AstNode::Function { name, .. } => name.clone(),
                    _ => None,
                };
                eprintln!("  [{}] Kernel(Function): {:?}", i, name);
            }
        }

        // 少なくとも1つのノードが吸収可能であるべき
        assert!(
            !absorbable.is_empty(),
            "Should find at least one absorbable Kernel node for multiple outputs"
        );

        // マルチフェーズ最適化を使用
        let config = MultiPhaseConfig::new()
            .with_beam_width(4)
            .with_max_steps(50)
            .with_progress(false)
            .with_collect_logs(false);

        let optimizer = create_multi_phase_optimizer(config);
        let (optimized, history) = optimizer.optimize_with_history(graph);

        eprintln!("\n=== Optimization History ===");
        for snapshot in history.snapshots() {
            let sink_info = if let Some(sink) = snapshot.graph.program_root() {
                if let GraphOp::ProgramRoot { ast, outputs } = &sink.op {
                    let func_count = if let crate::ast::AstNode::Program { functions, .. } = ast {
                        functions.len()
                    } else {
                        0
                    };
                    format!("outputs={:?}, funcs={}", outputs, func_count)
                } else {
                    "Unknown".to_string()
                }
            } else {
                "No ProgramRoot".to_string()
            };
            eprintln!(
                "Step {}: {} - cost={:.2} - {}",
                snapshot.step, snapshot.description, snapshot.cost, sink_info
            );
        }

        // 最終グラフを確認
        eprintln!("\n=== Final Graph ===");
        if let Some(sink) = optimized.program_root()
            && let GraphOp::ProgramRoot { ast, outputs } = &sink.op
        {
            eprintln!("Outputs: {:?}", outputs);
            if let crate::ast::AstNode::Program { functions, .. } = ast {
                eprintln!(
                    "Program has {} functions (including harp_main)",
                    functions.len()
                );
                for (i, func) in functions.iter().enumerate() {
                    let name = match func {
                        crate::ast::AstNode::Kernel { name, .. } => name.clone(),
                        crate::ast::AstNode::Function { name, .. } => name.clone(),
                        _ => None,
                    };
                    eprintln!("  function[{}]: {:?}", i, name);
                }
                // 最低でも1つのカーネル + harp_main = 2つの関数があるべき
                // ビームサーチはコストベースで最適化するため、ProgramRootAbsorption後にコストが上がる場合は
                // 吸収前の状態が選ばれる可能性がある
                assert!(
                    functions.len() >= 2,
                    "Should have at least 1 kernel + harp_main, got {}",
                    functions.len()
                );
            }
        }
    }
}
