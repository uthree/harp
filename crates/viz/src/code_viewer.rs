//! 最終的に生成されたコードを表示するビューア
//!
//! グラフ最適化の結果として生成されたCustom(Program)ノードのコードを表示します。

use harp::ast::renderer::render_ast_with;
use harp::backend::c::CRenderer;
use harp::backend::c_like::CLikeRenderer;
use harp::graph::{Graph, GraphNode, GraphOp};
use harp::opt::graph::OptimizationHistory;
use std::collections::HashSet;

/// コードビューアアプリケーション
///
/// 最適化後のグラフから生成されたコードを表示します。
pub struct CodeViewerApp<R = CRenderer>
where
    R: CLikeRenderer + Clone,
{
    /// 最適化履歴
    optimization_history: Option<OptimizationHistory>,
    /// レンダラー
    renderer: R,
    /// キャッシュされた最終コード
    cached_code: Option<String>,
}

impl Default for CodeViewerApp<CRenderer> {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeViewerApp<CRenderer> {
    /// 新しいCodeViewerAppを作成（デフォルトでCRendererを使用）
    pub fn new() -> Self {
        Self::with_renderer(CRenderer::new())
    }
}

impl<R> CodeViewerApp<R>
where
    R: CLikeRenderer + Clone,
{
    /// カスタムレンダラーを使用してCodeViewerAppを作成
    pub fn with_renderer(renderer: R) -> Self {
        Self {
            optimization_history: None,
            renderer,
            cached_code: None,
        }
    }

    /// 最適化履歴を読み込む
    pub fn load_history(&mut self, history: OptimizationHistory) {
        if history.is_empty() {
            log::warn!("Attempted to load empty optimization history");
            return;
        }

        // 最終ステップのグラフからコードを抽出
        if let Some(last_snapshot) = history.snapshots().last() {
            self.cached_code = self.extract_code_from_graph(&last_snapshot.graph);
        }

        self.optimization_history = Some(history);

        log::info!("Optimization history loaded for code viewer");
    }

    /// グラフを直接読み込む
    pub fn load_graph(&mut self, graph: Graph) {
        self.cached_code = self.extract_code_from_graph(&graph);
        self.optimization_history = None;

        log::info!("Graph loaded for code viewer");
    }

    /// グラフからCustom(Program)またはCustom(Function)のコードを抽出
    fn extract_code_from_graph(&self, graph: &Graph) -> Option<String> {
        let mut visited = HashSet::new();
        let mut program_ast = None;
        let mut function_asts = Vec::new();

        // 全ノードを走査してCustomノードを収集
        for output in graph.outputs().values() {
            Self::collect_custom_nodes(output, &mut visited, &mut program_ast, &mut function_asts);
        }

        // Custom(Program)があればそれを優先
        if let Some(ast) = program_ast {
            return Some(render_ast_with(&ast, &self.renderer));
        }

        // Custom(Function)が複数ある場合は全て連結（重複を除去）
        if !function_asts.is_empty() {
            // レンダリングして重複を除去
            let mut seen_codes = HashSet::new();
            let mut unique_codes = Vec::new();

            for ast in &function_asts {
                let code = render_ast_with(ast, &self.renderer);
                if !seen_codes.contains(&code) {
                    seen_codes.insert(code.clone());
                    unique_codes.push(code);
                }
            }

            if !unique_codes.is_empty() {
                return Some(unique_codes.join("\n\n// ================\n\n"));
            }
        }

        None
    }

    /// Customノードを再帰的に収集
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

        // Customノードの場合はASTを収集
        if let GraphOp::Custom { ast } = &node.op {
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

        // 統計情報
        if let Some(ref history) = self.optimization_history {
            ui.horizontal(|ui| {
                ui.label("Optimization Steps:");
                ui.label(format!("{}", history.len()));

                ui.separator();

                // 最終コストを表示
                if let Some(last) = history.snapshots().last() {
                    ui.label("Final Cost:");
                    ui.label(format!("{:.2}", last.cost));
                }
            });
            ui.separator();
        }

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
            ui.label("  - The graph must be fully lowered (contain Custom nodes)");
            ui.label("  - Use single-stage or unified optimization for best results");
        }
    }
}
