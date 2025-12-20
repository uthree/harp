//! Harp Visualization Library
//!
//! グラフ構造とコード生成を可視化するためのライブラリ

pub mod code_viewer;
pub mod diff_viewer;
pub mod graph_viewer;
pub mod renderer_selector;

pub use code_viewer::CodeViewerApp;
pub use diff_viewer::{
    show_collapsible_diff, show_resizable_diff, show_text_diff, DiffViewerConfig,
};
pub use graph_viewer::GraphViewerApp;
pub use renderer_selector::RendererType;

/// 可視化アプリケーション全体を統合するメインアプリ
///
/// レンダラーは実行時に切り替え可能です。
pub struct HarpVizApp {
    /// 現在のタブ
    current_tab: VizTab,
    /// グラフビューア
    graph_viewer: GraphViewerApp,
    /// コードビューア（AST最適化履歴を表示）
    code_viewer: CodeViewerApp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
enum VizTab {
    GraphViewer,
    CodeViewer,
    FinalCode,
}

impl Default for HarpVizApp {
    fn default() -> Self {
        Self::new()
    }
}

impl HarpVizApp {
    /// 新しいHarpVizAppを作成（デフォルトでCRendererを使用）
    pub fn new() -> Self {
        Self::with_renderer_type(RendererType::default())
    }

    /// 指定されたレンダラータイプでHarpVizAppを作成
    pub fn with_renderer_type(renderer_type: RendererType) -> Self {
        let mut graph_viewer = GraphViewerApp::new();
        graph_viewer.set_renderer_type(renderer_type);

        Self {
            current_tab: VizTab::GraphViewer,
            graph_viewer,
            code_viewer: CodeViewerApp::with_renderer_type(renderer_type),
        }
    }

    /// レンダラータイプを設定
    ///
    /// すべてのサブコンポーネントのレンダラーも更新されます。
    pub fn set_renderer_type(&mut self, renderer_type: RendererType) {
        self.graph_viewer.set_renderer_type(renderer_type);
        self.code_viewer.set_renderer_type(renderer_type);
    }

    /// 現在のレンダラータイプを取得
    pub fn renderer_type(&self) -> RendererType {
        self.code_viewer.renderer_type()
    }

    /// グラフ最適化履歴を読み込む
    pub fn load_graph_optimization_history(
        &mut self,
        history: harp::opt::graph::OptimizationHistory,
    ) {
        // コードビューアにも履歴を読み込む（最終コード表示用）
        self.code_viewer.load_history(history.clone());
        self.graph_viewer.load_history(history);
        // グラフビューアタブに切り替え
        self.current_tab = VizTab::GraphViewer;
    }

    /// 複数のグラフ最適化履歴をフェーズ名付きで結合して読み込む
    ///
    /// 2段階最適化（Phase 1: グラフ最適化、Phase 2: カーネルマージなど）の
    /// 履歴を1つのタイムラインとして表示するために使用します。
    ///
    /// # Arguments
    /// * `histories` - (フェーズ名, 履歴) のタプルのベクター
    ///
    /// # Example
    /// ```ignore
    /// app.load_combined_graph_histories(vec![
    ///     ("Graph Opt".to_string(), phase1_history),
    ///     ("Kernel Merge".to_string(), phase2_history),
    /// ]);
    /// ```
    pub fn load_combined_graph_histories(
        &mut self,
        histories: Vec<(String, harp::opt::graph::OptimizationHistory)>,
    ) {
        let combined = harp::opt::graph::OptimizationHistory::from_phases(&histories);
        // コードビューアにも履歴を読み込む（最終コード表示用）
        self.code_viewer.load_history(combined.clone());
        self.graph_viewer.load_history(combined);
        self.current_tab = VizTab::GraphViewer;
    }

    /// グラフを読み込む
    pub fn load_graph(&mut self, graph: harp::graph::Graph) {
        // コードビューアにもグラフを読み込む
        self.code_viewer.load_graph(graph.clone());
        self.graph_viewer.load_graph(graph);
        self.current_tab = VizTab::GraphViewer;
    }

    /// 最適化済みASTを読み込む
    ///
    /// AST最適化後のProgramを直接Code Viewerに設定します。
    /// グラフ履歴からの抽出をバイパスするため、AST最適化の結果が
    /// 正しくCode Viewerに反映されます。
    ///
    /// # Example
    /// ```ignore
    /// let (optimized_program, _) = pipeline.optimize_graph_with_all_histories(graph)?;
    /// app.load_optimized_ast(optimized_program);
    /// ```
    pub fn load_optimized_ast(&mut self, ast: harp::ast::AstNode) {
        self.code_viewer.load_optimized_ast(ast);
    }

    /// AST最適化履歴を読み込む
    ///
    /// AST最適化の各ステップをCode Viewerで可視化できるようにします。
    /// Code Viewerタブに切り替え、AST履歴表示モードを有効にします。
    ///
    /// # Example
    /// ```ignore
    /// let (optimized_ast, ast_history) = ast_optimizer.optimize_with_history(ast);
    /// app.load_ast_optimization_history(ast_history);
    /// ```
    pub fn load_ast_optimization_history(&mut self, history: harp::opt::ast::OptimizationHistory) {
        self.code_viewer.load_ast_history(history);
        // Code Viewerタブに切り替え
        self.current_tab = VizTab::CodeViewer;
    }

    /// グラフ最適化履歴を読み込んでvisualizerを起動
    ///
    /// グラフ最適化履歴を指定して可視化ウィンドウを起動します。
    ///
    /// # Note
    /// AST最適化は現在グラフ最適化に統合されているため、
    /// 別途AST最適化履歴を指定する必要はありません。
    ///
    /// # 例
    /// ```ignore
    /// use harp_viz::HarpVizApp;
    ///
    /// let (optimized_graph, history) = optimizer.optimize_with_history(graph);
    ///
    /// HarpVizApp::run_with_history(history)?;
    /// ```
    pub fn run_with_history(
        graph_history: harp::opt::graph::OptimizationHistory,
    ) -> Result<(), eframe::Error> {
        let mut app = Self::new();
        app.load_graph_optimization_history(graph_history);

        let native_options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1200.0, 800.0])
                .with_title("Harp Visualizer"),
            ..Default::default()
        };

        eframe::run_native(
            "Harp Visualizer",
            native_options,
            Box::new(|_cc| Ok(Box::new(app))),
        )
    }
}

impl eframe::App for HarpVizApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.heading("Harp Visualizer");
                ui.separator();

                if ui
                    .selectable_label(self.current_tab == VizTab::GraphViewer, "Graph Viewer")
                    .clicked()
                {
                    self.current_tab = VizTab::GraphViewer;
                }

                if ui
                    .selectable_label(self.current_tab == VizTab::CodeViewer, "Code Viewer")
                    .clicked()
                {
                    self.current_tab = VizTab::CodeViewer;
                }

                if ui
                    .selectable_label(self.current_tab == VizTab::FinalCode, "Final Code")
                    .clicked()
                {
                    self.current_tab = VizTab::FinalCode;
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.current_tab {
            VizTab::GraphViewer => {
                self.graph_viewer.ui(ui);
            }
            VizTab::CodeViewer => {
                self.code_viewer.ui(ui);
            }
            VizTab::FinalCode => {
                self.code_viewer.ui_final_code_tab(ui);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harp_viz_app_creation() {
        let app = HarpVizApp::new();
        assert_eq!(app.current_tab, VizTab::GraphViewer);
    }

    #[test]
    fn test_renderer_type_selection() {
        let mut app = HarpVizApp::new();
        assert_eq!(app.renderer_type(), RendererType::CLike); // CLikeがデフォルト

        // 別のタイプに変更しても戻せることをテスト
        let types = RendererType::all();
        if types.len() > 1 {
            app.set_renderer_type(types[1]);
            assert_eq!(app.renderer_type(), types[1]);

            app.set_renderer_type(RendererType::CLike);
            assert_eq!(app.renderer_type(), RendererType::CLike);
        }
    }
}
