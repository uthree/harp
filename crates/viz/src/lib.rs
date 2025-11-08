//! Harp Visualization Library
//!
//! グラフ構造とパフォーマンス統計を可視化するためのライブラリ

pub mod graph_viewer;
pub mod perf_viewer;

pub use graph_viewer::GraphViewerApp;
pub use perf_viewer::PerfViewerApp;

/// 可視化アプリケーション全体を統合するメインアプリ
pub struct HarpVizApp {
    /// 現在のタブ
    current_tab: VizTab,
    /// グラフビューア
    graph_viewer: GraphViewerApp,
    /// パフォーマンスビューア
    perf_viewer: PerfViewerApp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VizTab {
    GraphViewer,
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
            perf_viewer: PerfViewerApp::new(),
        }
    }

    /// 最適化履歴を読み込む
    pub fn load_optimization_history(&mut self, history: harp::opt::graph::OptimizationHistory) {
        self.graph_viewer.load_history(history);
        // グラフビューアタブに切り替え
        self.current_tab = VizTab::GraphViewer;
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
            VizTab::PerfViewer => {
                self.perf_viewer.ui(ui);
            }
        });
    }
}
