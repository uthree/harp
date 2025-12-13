//! Harp Visualization Library
//!
//! グラフ構造とパフォーマンス統計を可視化するためのライブラリ

pub mod code_viewer;
pub mod diff_viewer;
pub mod graph_viewer;
pub mod perf_viewer;
pub mod renderer_selector;

pub use code_viewer::CodeViewerApp;
pub use diff_viewer::{
    show_collapsible_diff, show_resizable_diff, show_text_diff, DiffViewerConfig,
};
pub use graph_viewer::GraphViewerApp;
pub use perf_viewer::PerfViewerApp;
pub use renderer_selector::RendererType;

/// 可視化アプリケーション全体を統合するメインアプリ
///
/// レンダラーは実行時に切り替え可能です。
pub struct HarpVizApp {
    /// 現在のタブ
    current_tab: VizTab,
    /// グラフビューア
    graph_viewer: GraphViewerApp,
    /// コードビューア（最終的な生成コードを表示）
    code_viewer: CodeViewerApp,
    /// パフォーマンスビューア
    perf_viewer: PerfViewerApp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
enum VizTab {
    GraphViewer,
    CodeViewer,
    PerfViewer,
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
            perf_viewer: PerfViewerApp::new(),
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

    /// GenericPipelineから最適化履歴を読み込む
    ///
    /// GenericPipelineに保存されているグラフ最適化履歴とAST最適化履歴を読み込みます。
    /// 2段階最適化（Phase 1 + Phase 2）の履歴は自動的に結合されて表示されます。
    /// 履歴が存在する場合、適切なタブに切り替えます。
    ///
    /// # 型パラメータ
    /// * `PR` - Rendererの型
    /// * `PC` - Compilerの型
    pub fn load_from_pipeline<PR, PC>(&mut self, pipeline: &harp::backend::GenericPipeline<PR, PC>)
    where
        PR: harp::backend::Renderer + Clone + 'static,
        PC: harp::backend::Compiler<CodeRepr = PR::CodeRepr> + Clone + 'static,
        PC::Buffer: 'static,
    {
        // グラフ最適化履歴を読み込む（Phase 1 + Phase 2 を結合）
        if let Some(combined_history) = pipeline.histories.combined_graph_history() {
            self.load_graph_optimization_history(combined_history);
        }

        // AST最適化履歴を読み込む
        if let Some(ref ast_history) = pipeline.histories.ast {
            self.code_viewer.load_ast_history(ast_history.clone());
        }
    }

    /// GenericPipelineから最適化履歴を読み込んで所有権を移動
    ///
    /// `load_from_pipeline`と異なり、Pipelineから履歴を取り出して所有権を移動します。
    /// Pipeline内の履歴はクリアされます。
    /// 2段階最適化の履歴は結合されて読み込まれます。
    ///
    /// # 型パラメータ
    /// * `PR` - Rendererの型
    /// * `PC` - Compilerの型
    pub fn take_from_pipeline<PR, PC>(
        &mut self,
        pipeline: &mut harp::backend::GenericPipeline<PR, PC>,
    ) where
        PR: harp::backend::Renderer + Clone + 'static,
        PC: harp::backend::Compiler<CodeRepr = PR::CodeRepr> + Clone + 'static,
        PC::Buffer: 'static,
    {
        // グラフ最適化履歴を取得（Phase 1 + Phase 2 を結合）
        if let Some(combined_history) = pipeline.histories.combined_graph_history() {
            self.load_graph_optimization_history(combined_history);
        }

        // AST最適化履歴を取得
        if let Some(ast_history) = pipeline.histories.ast.take() {
            self.code_viewer.load_ast_history(ast_history);
        }

        // 元の履歴をクリア
        pipeline.histories.graph = None;
        pipeline.histories.graph_phase2 = None;
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
    /// * `PR` - Rendererの型
    /// * `PC` - Compilerの型
    pub fn run_from_pipeline<PR, PC>(
        pipeline: &harp::backend::GenericPipeline<PR, PC>,
    ) -> Result<(), eframe::Error>
    where
        PR: harp::backend::Renderer + Clone + 'static,
        PC: harp::backend::Compiler<CodeRepr = PR::CodeRepr> + Clone + 'static,
        PC::Buffer: 'static,
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
            VizTab::CodeViewer => {
                self.code_viewer.ui(ui);
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
    #[derive(Clone)]
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
    #[derive(Clone)]
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

        fn compile(
            &mut self,
            _code: &Self::CodeRepr,
            _signature: harp::backend::KernelSignature,
        ) -> Self::Kernel {
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
    fn test_renderer_type_selection() {
        let mut app = HarpVizApp::new();
        assert_eq!(app.renderer_type(), RendererType::C);

        app.set_renderer_type(RendererType::OpenCL);
        assert_eq!(app.renderer_type(), RendererType::OpenCL);

        app.set_renderer_type(RendererType::Metal);
        assert_eq!(app.renderer_type(), RendererType::Metal);
    }
}
