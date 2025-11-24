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
#[allow(clippy::enum_variant_names)]
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
        if let Some(graph_history) = &pipeline.histories.graph {
            self.load_graph_optimization_history(graph_history.clone());
        }

        // AST最適化履歴を読み込む
        if let Some(ast_history) = &pipeline.histories.ast {
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
        if let Some(graph_history) = pipeline.histories.graph.take() {
            self.load_graph_optimization_history(graph_history);
        }

        // AST最適化履歴を取得
        if let Some(ast_history) = pipeline.histories.ast.take() {
            self.load_ast_optimization_history(ast_history);
        }
    }

    /// Pipelineから直接visualizerを起動するヘルパー関数
    ///
    /// この関数は、GenericPipelineに保存されている最適化履歴を読み込んで、
    /// 可視化ウィンドウを起動します。
    ///
    /// # 例
    /// ```ignore
    /// use harp::backend::GenericPipeline;
    /// use harp_viz::HarpVizApp;
    ///
    /// let mut pipeline = GenericPipeline::new(renderer, compiler);
    /// pipeline.enable_graph_optimization = true;
    /// pipeline.enable_ast_optimization = true;
    /// pipeline.collect_histories = true;
    ///
    /// // グラフをコンパイル（最適化履歴が記録される）
    /// let kernel = pipeline.compile_graph(graph)?;
    ///
    /// // 可視化ウィンドウを起動
    /// HarpVizApp::run_from_pipeline(&pipeline)?;
    /// ```
    ///
    /// # 型パラメータ
    /// * `R` - Rendererの型
    /// * `C` - Compilerの型
    pub fn run_from_pipeline<R, C>(
        pipeline: &harp::backend::GenericPipeline<R, C>,
    ) -> Result<(), eframe::Error>
    where
        R: harp::backend::Renderer,
        C: harp::backend::Compiler<CodeRepr = R::CodeRepr>,
    {
        let mut app = Self::new();
        app.load_from_pipeline(pipeline);

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

    /// 最適化履歴を読み込んでvisualizerを起動
    ///
    /// グラフとASTの最適化履歴を個別に指定して可視化ウィンドウを起動します。
    ///
    /// # 例
    /// ```ignore
    /// use harp_viz::HarpVizApp;
    ///
    /// let (optimized_graph, graph_history) = graph_optimizer.optimize_with_history(graph);
    /// let (optimized_ast, ast_history) = ast_optimizer.optimize_with_history(ast);
    ///
    /// HarpVizApp::run_with_histories(Some(graph_history), Some(ast_history))?;
    /// ```
    pub fn run_with_histories(
        graph_history: Option<harp::opt::graph::OptimizationHistory>,
        ast_history: Option<harp::opt::ast::OptimizationHistory>,
    ) -> Result<(), eframe::Error> {
        let mut app = Self::new();

        if let Some(history) = graph_history {
            app.load_graph_optimization_history(history);
        }

        if let Some(history) = ast_history {
            app.load_ast_optimization_history(history);
        }

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
    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    #[derive(Debug, Clone)]
    struct DummyKernel;

    impl Kernel for DummyKernel {
        type Buffer = DummyBuffer;

        fn signature(&self) -> KernelSignature {
            KernelSignature::empty()
        }
    }

    #[allow(dead_code)]
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
}
