//! Harp Visualization Library
//!
//! グラフ構造とパフォーマンス統計を可視化するためのライブラリ

pub mod ast_viewer;
pub mod graph_viewer;
pub mod perf_viewer;

pub use ast_viewer::AstViewerApp;
pub use graph_viewer::GraphViewerApp;
pub use perf_viewer::PerfViewerApp;

/// 可視化アプリケーション全体を統合するメインアプリ
pub struct HarpVizApp {
    /// 現在のタブ
    current_tab: VizTab,
    /// グラフビューア
    graph_viewer: GraphViewerApp,
    /// ASTビューア
    ast_viewer: AstViewerApp,
    /// パフォーマンスビューア
    perf_viewer: PerfViewerApp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VizTab {
    GraphViewer,
    AstViewer,
    PerfViewer,
}

impl Default for HarpVizApp {
    fn default() -> Self {
        Self::new()
    }
}

impl HarpVizApp {
    /// 新しいHarpVizAppを作成
    pub fn new() -> Self {
        Self {
            current_tab: VizTab::GraphViewer,
            graph_viewer: GraphViewerApp::new(),
            ast_viewer: AstViewerApp::new(),
            perf_viewer: PerfViewerApp::new(),
        }
    }

    /// グラフ最適化履歴を読み込む
    pub fn load_graph_optimization_history(
        &mut self,
        history: harp::opt::graph::OptimizationHistory,
    ) {
        self.graph_viewer.load_history(history);
        // グラフビューアタブに切り替え
        self.current_tab = VizTab::GraphViewer;
    }

    /// AST最適化履歴を読み込む
    pub fn load_ast_optimization_history(&mut self, history: harp::opt::ast::OptimizationHistory) {
        self.ast_viewer.load_history(history);
        // ASTビューアタブに切り替え
        self.current_tab = VizTab::AstViewer;
    }

    /// グラフを読み込む
    pub fn load_graph(&mut self, graph: harp::graph::Graph) {
        self.graph_viewer.load_graph(graph);
        self.current_tab = VizTab::GraphViewer;
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
                    .selectable_label(self.current_tab == VizTab::AstViewer, "AST Viewer")
                    .clicked()
                {
                    self.current_tab = VizTab::AstViewer;
                }

                if ui
                    .selectable_label(self.current_tab == VizTab::PerfViewer, "Performance")
                    .clicked()
                {
                    self.current_tab = VizTab::PerfViewer;
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.current_tab {
            VizTab::GraphViewer => {
                self.graph_viewer.ui(ui);
            }
            VizTab::AstViewer => {
                self.ast_viewer.ui(ui);
            }
            VizTab::PerfViewer => {
                self.perf_viewer.ui(ui);
            }
        });
    }
}
