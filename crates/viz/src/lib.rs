//! Harp Visualization Library
//!
//! グラフ構造とパフォーマンス統計を可視化するためのライブラリ

pub mod ast_viewer;
pub mod diff_viewer;
pub mod graph_viewer;
pub mod perf_viewer;

pub use ast_viewer::AstViewerApp;
pub use diff_viewer::{
    show_collapsible_diff, show_resizable_diff, show_text_diff, DiffViewerConfig,
};
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

    /// 複数のFunction用のAST最適化履歴を読み込む
    pub fn load_multiple_ast_histories(
        &mut self,
        histories: std::collections::HashMap<String, harp::opt::ast::OptimizationHistory>,
    ) {
        self.ast_viewer.load_multiple_histories(histories);
        // ASTビューアタブに切り替え
        self.current_tab = VizTab::AstViewer;
    }

    /// グラフを読み込む
    pub fn load_graph(&mut self, graph: harp::graph::Graph) {
        self.graph_viewer.load_graph(graph);
        self.current_tab = VizTab::GraphViewer;
    }

    /// GenericPipelineから最適化履歴を読み込む
    ///
    /// GenericPipelineに保存されているグラフとAST両方の最適化履歴を読み込みます。
    /// 履歴が存在する場合、適切なタブに切り替えます。
    ///
    /// # 型パラメータ
    /// * `R` - Rendererの型
    /// * `C` - Compilerの型
    pub fn load_from_pipeline<R, C>(&mut self, pipeline: &harp::backend::GenericPipeline<R, C>)
    where
        R: harp::backend::Renderer,
        C: harp::backend::Compiler<CodeRepr = R::CodeRepr>,
    {
        // グラフ最適化履歴を読み込む
        if let Some(graph_history) = pipeline.last_graph_optimization_history() {
            self.load_graph_optimization_history(graph_history.clone());
        }

        // AST最適化履歴を読み込む
        if let Some(ast_history) = pipeline.last_ast_optimization_history() {
            self.load_ast_optimization_history(ast_history.clone());
        }
    }

    /// GenericPipelineから最適化履歴を読み込んで所有権を移動
    ///
    /// `load_from_pipeline`と異なり、Pipelineから履歴を取り出して所有権を移動します。
    /// Pipeline内の履歴はクリアされます。
    ///
    /// # 型パラメータ
    /// * `R` - Rendererの型
    /// * `C` - Compilerの型
    pub fn take_from_pipeline<R, C>(&mut self, pipeline: &mut harp::backend::GenericPipeline<R, C>)
    where
        R: harp::backend::Renderer,
        C: harp::backend::Compiler<CodeRepr = R::CodeRepr>,
    {
        // グラフ最適化履歴を取得
        if let Some(graph_history) = pipeline.take_graph_optimization_history() {
            self.load_graph_optimization_history(graph_history);
        }

        // AST最適化履歴を取得
        if let Some(ast_history) = pipeline.take_ast_optimization_history() {
            self.load_ast_optimization_history(ast_history);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use harp::backend::{Buffer, Compiler, Kernel, KernelSignature, Renderer};

    // テスト用のダミー実装
    struct DummyRenderer;

    impl Renderer for DummyRenderer {
        type CodeRepr = String;
        type Option = ();

        fn render(&self, _program: &harp::ast::AstNode) -> Self::CodeRepr {
            "dummy code".to_string()
        }

        fn is_available(&self) -> bool {
            true
        }
    }

    #[derive(Debug, Clone)]
    struct DummyBuffer;

    impl Buffer for DummyBuffer {
        fn shape(&self) -> Vec<usize> {
            vec![]
        }

        fn dtype(&self) -> harp::ast::DType {
            harp::ast::DType::F32
        }

        fn to_bytes(&self) -> Vec<u8> {
            vec![]
        }

        fn from_bytes(&mut self, _bytes: &[u8]) -> Result<(), String> {
            Ok(())
        }

        fn byte_len(&self) -> usize {
            0
        }
    }

    #[derive(Debug, Clone)]
    struct DummyKernel;

    impl Kernel for DummyKernel {
        type Buffer = DummyBuffer;

        fn signature(&self) -> KernelSignature {
            KernelSignature::empty()
        }
    }

    struct DummyCompiler;

    impl Compiler for DummyCompiler {
        type CodeRepr = String;
        type Buffer = DummyBuffer;
        type Kernel = DummyKernel;
        type Option = ();

        fn new() -> Self {
            Self
        }

        fn is_available(&self) -> bool {
            true
        }

        fn compile(&mut self, _code: &Self::CodeRepr) -> Self::Kernel {
            DummyKernel
        }

        fn create_buffer(&self, _shape: Vec<usize>, _element_size: usize) -> Self::Buffer {
            DummyBuffer
        }
    }

    #[test]
    fn test_harp_viz_app_creation() {
        let app = HarpVizApp::new();
        assert_eq!(app.current_tab, VizTab::GraphViewer);
    }

    #[test]
    fn test_load_from_pipeline() {
        use harp::backend::GenericPipeline;
        use harp::graph::Graph;
        use harp::opt::graph::{OptimizationHistory as GraphOptHistory, OptimizationSnapshot};

        let renderer = DummyRenderer;
        let compiler = DummyCompiler;
        let mut pipeline = GenericPipeline::new(renderer, compiler);

        // グラフ最適化履歴を作成してPipelineに設定
        let graph = Graph::new();
        let snapshot = OptimizationSnapshot::new(0, graph.clone(), 1.0, "Initial".to_string());
        let mut history = GraphOptHistory::new();
        history.add_snapshot(snapshot);
        pipeline.set_graph_optimization_history(history);

        // HarpVizAppに読み込む
        let mut viz_app = HarpVizApp::new();
        viz_app.load_from_pipeline(&pipeline);

        // Pipelineには履歴が残っている
        assert!(pipeline.last_graph_optimization_history().is_some());
    }

    #[test]
    fn test_take_from_pipeline() {
        use harp::backend::GenericPipeline;
        use harp::graph::Graph;
        use harp::opt::graph::{OptimizationHistory as GraphOptHistory, OptimizationSnapshot};

        let renderer = DummyRenderer;
        let compiler = DummyCompiler;
        let mut pipeline = GenericPipeline::new(renderer, compiler);

        // グラフ最適化履歴を作成してPipelineに設定
        let graph = Graph::new();
        let snapshot = OptimizationSnapshot::new(0, graph.clone(), 1.0, "Initial".to_string());
        let mut history = GraphOptHistory::new();
        history.add_snapshot(snapshot);
        pipeline.set_graph_optimization_history(history);

        // HarpVizAppに読み込んで所有権を移動
        let mut viz_app = HarpVizApp::new();
        viz_app.take_from_pipeline(&mut pipeline);

        // Pipelineから履歴がクリアされている
        assert!(pipeline.last_graph_optimization_history().is_none());
    }
}
