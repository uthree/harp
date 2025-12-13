//! 最終的に生成されたコードを表示するビューア
//!
//! グラフ最適化の結果として生成されたKernel(Program)ノードのコードを表示します。

use crate::renderer_selector::{render_with_type, renderer_selector_ui, RendererType};
use harp::graph::{Graph, GraphNode, GraphOp};
use harp::opt::graph::OptimizationHistory;
use std::collections::HashSet;

/// コードビューアアプリケーション
///
/// 最適化後のグラフから生成されたコードを表示します。
pub struct CodeViewerApp {
    /// 最適化履歴
    optimization_history: Option<OptimizationHistory>,
    /// 現在のレンダラータイプ
    renderer_type: RendererType,
    /// キャッシュされた最終コード
    cached_code: Option<String>,
    /// キャッシュされたAST（レンダラー変更時に再レンダリング用）
    cached_ast: Option<harp::ast::AstNode>,
}

impl Default for CodeViewerApp {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeViewerApp {
    /// 新しいCodeViewerAppを作成（デフォルトでCRendererを使用）
    pub fn new() -> Self {
        Self::with_renderer_type(RendererType::default())
    }

    /// 指定されたレンダラータイプでCodeViewerAppを作成
    pub fn with_renderer_type(renderer_type: RendererType) -> Self {
        Self {
            optimization_history: None,
            renderer_type,
            cached_code: None,
            cached_ast: None,
        }
    }

    /// レンダラータイプを設定
    pub fn set_renderer_type(&mut self, renderer_type: RendererType) {
        if self.renderer_type != renderer_type {
            self.renderer_type = renderer_type;
            // キャッシュされたASTがあれば再レンダリング
            if let Some(ref ast) = self.cached_ast {
                self.cached_code = Some(render_with_type(ast, renderer_type));
            }
        }
    }

    /// 現在のレンダラータイプを取得
    pub fn renderer_type(&self) -> RendererType {
        self.renderer_type
    }

    /// 最適化履歴を読み込む
    pub fn load_history(&mut self, history: OptimizationHistory) {
        if history.is_empty() {
            log::warn!("Attempted to load empty optimization history");
            return;
        }

        // 最終ステップのグラフからASTを抽出
        if let Some(last_snapshot) = history.snapshots().last() {
            self.cached_ast = self.extract_ast_from_graph(&last_snapshot.graph);
            // キャッシュされたASTからコードを生成
            if let Some(ref ast) = self.cached_ast {
                self.cached_code = Some(render_with_type(ast, self.renderer_type));
            } else {
                self.cached_code = None;
            }
        }

        self.optimization_history = Some(history);

        log::info!("Optimization history loaded for code viewer");
    }

    /// グラフを直接読み込む
    pub fn load_graph(&mut self, graph: Graph) {
        self.cached_ast = self.extract_ast_from_graph(&graph);
        // キャッシュされたASTからコードを生成
        if let Some(ref ast) = self.cached_ast {
            self.cached_code = Some(render_with_type(ast, self.renderer_type));
        } else {
            self.cached_code = None;
        }
        self.optimization_history = None;

        log::info!("Graph loaded for code viewer");
    }

    /// 最適化済みASTを直接読み込む
    ///
    /// グラフ履歴からの抽出をバイパスして、AST最適化後のProgramを直接設定します。
    /// これにより、AST最適化の結果がCode Viewerに正しく反映されます。
    pub fn load_optimized_ast(&mut self, ast: harp::ast::AstNode) {
        self.cached_ast = Some(ast.clone());
        self.cached_code = Some(render_with_type(&ast, self.renderer_type));
        log::info!("Optimized AST loaded for code viewer");
    }

    /// グラフからProgramRoot(Program)、Kernel(Program)またはKernel(Function)のASTを抽出
    fn extract_ast_from_graph(&self, graph: &Graph) -> Option<harp::ast::AstNode> {
        // 1. ProgramRootノードのProgramを最優先で確認
        if let Some(sink) = graph.program_root() {
            if let harp::graph::GraphOp::ProgramRoot { ast, .. } = &sink.op {
                if let harp::ast::AstNode::Program { functions, .. } = ast {
                    if !functions.is_empty() {
                        return Some(ast.clone());
                    }
                }
            }
        }

        // 2. Kernel(Program/Function)を走査してASTを収集
        let mut visited = HashSet::new();
        let mut program_ast = None;
        let mut function_asts = Vec::new();

        // ProgramRootがある場合はProgramRootのsrcから、ない場合はoutputsから走査
        if let Some(sink) = graph.program_root() {
            for src in &sink.src {
                Self::collect_custom_nodes(src, &mut visited, &mut program_ast, &mut function_asts);
            }
        } else {
            for output in graph.outputs().values() {
                Self::collect_custom_nodes(
                    output,
                    &mut visited,
                    &mut program_ast,
                    &mut function_asts,
                );
            }
        }

        // Kernel(Program)があればそれを優先
        if let Some(ast) = program_ast {
            return Some(ast);
        }

        // Kernel(Function)が複数ある場合は全て連結してProgram化
        if !function_asts.is_empty() {
            // 最初のASTを返す（複数ある場合は後で処理）
            // TODO: 複数のFunctionをProgramにまとめる
            return Some(function_asts.remove(0));
        }

        None
    }

    /// Kernelノードを再帰的に収集
    fn collect_custom_nodes(
        node: &GraphNode,
        visited: &mut HashSet<*const harp::graph::GraphNodeData>,
        program_ast: &mut Option<harp::ast::AstNode>,
        function_asts: &mut Vec<harp::ast::AstNode>,
    ) {
        let ptr = node.as_ptr();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        // 入力ノードを先に処理
        for src in &node.src {
            Self::collect_custom_nodes(src, visited, program_ast, function_asts);
        }

        // Kernelノードの場合はASTを収集
        if let GraphOp::Kernel { ast, .. } = &node.op {
            // Programかどうかを判定
            if Self::is_program(ast) {
                *program_ast = Some(ast.clone());
            } else {
                function_asts.push(ast.clone());
            }
        }
    }

    /// ASTがProgramかどうかを判定
    fn is_program(ast: &harp::ast::AstNode) -> bool {
        use harp::ast::AstNode;
        match ast {
            AstNode::Program { .. } => true,
            AstNode::Block { statements, .. } => {
                // Block内にKernelが複数あるか確認
                statements
                    .iter()
                    .filter(|node| matches!(node, AstNode::Kernel { .. }))
                    .count()
                    > 1
            }
            _ => false,
        }
    }

    /// UIを描画
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Code Viewer");
        ui.separator();

        // レンダラー選択と統計情報
        ui.horizontal(|ui| {
            // レンダラー選択
            if renderer_selector_ui(ui, &mut self.renderer_type) {
                // レンダラーが変更されたら再レンダリング
                if let Some(ref ast) = self.cached_ast {
                    self.cached_code = Some(render_with_type(ast, self.renderer_type));
                }
            }

            ui.separator();

            if let Some(ref history) = self.optimization_history {
                ui.label("Optimization Steps:");
                ui.label(format!("{}", history.len()));

                ui.separator();

                // 最終コストを表示
                if let Some(last) = history.snapshots().last() {
                    ui.label("Final Cost:");
                    ui.label(format!("{:.2}", last.cost));
                }
            }
        });
        ui.separator();

        // コード表示
        if let Some(ref code) = self.cached_code {
            ui.horizontal(|ui| {
                // クリップボードにコピーボタン
                if ui.button("Copy to Clipboard").clicked() {
                    ui.output_mut(|o| o.copied_text = code.clone());
                    log::info!("Code copied to clipboard");
                }

                ui.separator();

                // 行数を表示
                let line_count = code.lines().count();
                ui.label(format!("{} lines", line_count));
            });

            ui.separator();

            // シンタックスハイライト付きでコードを表示
            egui::ScrollArea::both()
                .id_salt("final_code_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let theme = egui_extras::syntax_highlighting::CodeTheme::from_memory(
                        ui.ctx(),
                        ui.style(),
                    );

                    let highlighted_code = egui_extras::syntax_highlighting::highlight(
                        ui.ctx(),
                        ui.style(),
                        &theme,
                        code,
                        "c", // C言語風のシンタックスハイライト
                    );

                    ui.add(egui::Label::new(highlighted_code).selectable(true));
                });
        } else {
            ui.label("No code available.");
            ui.label("Load an optimized graph to view the generated code.");
            ui.add_space(20.0);

            ui.label("Tips:");
            ui.label("  - The graph must be fully lowered (contain Kernel nodes)");
            ui.label("  - Use single-stage or unified optimization for best results");
        }
    }
}
