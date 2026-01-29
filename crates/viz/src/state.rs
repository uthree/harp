//! Application state management

use egui_snarl::Snarl;

use eclat::ast::AstNode;
use eclat::ast::renderer::render_ast_with;
use eclat::backend::renderer::{CLikeRenderer, GenericRenderer};
use eclat::opt::ast::history::OptimizationHistory;

use crate::convert::{graph_to_snarl, VizNode};
use crate::graph_history::GraphOptimizationHistory;
use crate::highlight::CodeHighlighter;
use crate::panels::candidates::CandidateInfo;
use crate::panels::code_panel::StyledSpan;

/// View mode for the visualization
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    /// Show AST/code view
    #[default]
    Ast,
    /// Show graph view
    Graph,
}

/// Application state
pub struct AppState<R: CLikeRenderer + Clone = GenericRenderer> {
    /// AST optimization history
    pub ast_history: OptimizationHistory,
    /// Graph optimization history
    pub graph_history: GraphOptimizationHistory,
    /// Current view mode
    pub view_mode: ViewMode,
    /// Current step for AST view
    pub ast_step: usize,
    /// Current step for graph view
    pub graph_step: usize,
    /// Selected candidate index (0=selected, 1+=alternatives)
    pub selected_candidate: usize,
    /// Code highlighter
    highlighter: CodeHighlighter,
    /// Code renderer
    renderer: R,
    /// Cached Snarl graph for current graph step
    cached_snarl: Option<Snarl<VizNode>>,
    /// Last graph step the snarl was built for
    cached_snarl_step: Option<usize>,
}

impl AppState<GenericRenderer> {
    /// Create a new state with AST history only
    pub fn new(ast_history: OptimizationHistory) -> Self {
        Self {
            ast_history,
            graph_history: GraphOptimizationHistory::new(),
            view_mode: ViewMode::Ast,
            ast_step: 0,
            graph_step: 0,
            selected_candidate: 0,
            highlighter: CodeHighlighter::new(),
            renderer: GenericRenderer::new(),
            cached_snarl: None,
            cached_snarl_step: None,
        }
    }
}

impl<R: CLikeRenderer + Clone> AppState<R> {
    /// Create a new state with AST history and custom renderer
    pub fn with_renderer(ast_history: OptimizationHistory, renderer: R) -> Self {
        Self {
            ast_history,
            graph_history: GraphOptimizationHistory::new(),
            view_mode: ViewMode::Ast,
            ast_step: 0,
            graph_step: 0,
            selected_candidate: 0,
            highlighter: CodeHighlighter::new(),
            renderer,
            cached_snarl: None,
            cached_snarl_step: None,
        }
    }

    /// Create a new state with both AST and graph history
    pub fn with_both_histories(
        ast_history: OptimizationHistory,
        graph_history: GraphOptimizationHistory,
        renderer: R,
    ) -> Self {
        Self {
            ast_history,
            graph_history,
            view_mode: ViewMode::Ast,
            ast_step: 0,
            graph_step: 0,
            selected_candidate: 0,
            highlighter: CodeHighlighter::new(),
            renderer,
            cached_snarl: None,
            cached_snarl_step: None,
        }
    }

    /// Set the graph optimization history
    pub fn set_graph_history(&mut self, history: GraphOptimizationHistory) {
        self.graph_history = history;
        self.cached_snarl = None;
        self.cached_snarl_step = None;
    }

    /// Get the current step based on view mode
    pub fn current_step(&self) -> usize {
        match self.view_mode {
            ViewMode::Ast => self.ast_step,
            ViewMode::Graph => self.graph_step,
        }
    }

    /// Get the current step mutably
    pub fn current_step_mut(&mut self) -> &mut usize {
        match self.view_mode {
            ViewMode::Ast => &mut self.ast_step,
            ViewMode::Graph => &mut self.graph_step,
        }
    }

    /// Get total steps for current view mode
    pub fn total_steps(&self) -> usize {
        match self.view_mode {
            ViewMode::Ast => self.ast_history.len(),
            ViewMode::Graph => self.graph_history.len(),
        }
    }

    /// Get the current AST if available
    pub fn current_ast(&self) -> Option<&AstNode> {
        let snapshot = self.ast_history.get(self.ast_step)?;
        if self.selected_candidate == 0 {
            Some(&snapshot.ast)
        } else {
            snapshot
                .alternatives
                .get(self.selected_candidate - 1)
                .map(|alt| &alt.ast)
        }
    }

    /// Render current AST to source code
    pub fn render_current_code(&self) -> String {
        match self.current_ast() {
            Some(ast) => render_ast_with(ast, &self.renderer),
            None => String::from("(No AST available)"),
        }
    }

    /// Get highlighted code for current AST
    pub fn highlighted_code(&self) -> Vec<Vec<StyledSpan>> {
        let code = self.render_current_code();
        self.highlighter.highlight(&code)
    }

    /// Get candidates for the current AST step
    pub fn ast_candidates(&self) -> Vec<CandidateInfo> {
        let Some(snapshot) = self.ast_history.get(self.ast_step) else {
            return vec![];
        };

        let mut candidates = vec![CandidateInfo {
            cost: snapshot.cost,
            suggester_name: snapshot.suggester_name.clone(),
            description: snapshot.description.clone(),
        }];

        for alt in &snapshot.alternatives {
            candidates.push(CandidateInfo {
                cost: alt.cost,
                suggester_name: alt.suggester_name.clone(),
                description: String::new(),
            });
        }

        candidates
    }

    /// Get candidates for the current graph step
    pub fn graph_candidates(&self) -> Vec<CandidateInfo> {
        let Some(snapshot) = self.graph_history.get(self.graph_step) else {
            return vec![];
        };

        let mut candidates = vec![CandidateInfo {
            cost: snapshot.cost,
            suggester_name: snapshot.suggester_name.clone(),
            description: snapshot.description.clone(),
        }];

        for alt in &snapshot.alternatives {
            candidates.push(CandidateInfo {
                cost: alt.cost,
                suggester_name: alt.suggester_name.clone(),
                description: alt.description.clone(),
            });
        }

        candidates
    }

    /// Get the Snarl graph for the current graph step
    pub fn current_snarl(&mut self) -> &mut Snarl<VizNode> {
        // Check if we need to rebuild the cache
        if self.cached_snarl_step != Some(self.graph_step) {
            if let Some(snapshot) = self.graph_history.get(self.graph_step) {
                self.cached_snarl = Some(graph_to_snarl(&snapshot.roots));
                self.cached_snarl_step = Some(self.graph_step);
            } else {
                self.cached_snarl = Some(Snarl::new());
                self.cached_snarl_step = Some(self.graph_step);
            }
        }

        self.cached_snarl.as_mut().unwrap()
    }

    /// Get current cost
    pub fn current_cost(&self) -> Option<f32> {
        match self.view_mode {
            ViewMode::Ast => self.ast_history.get(self.ast_step).map(|s| s.cost),
            ViewMode::Graph => self.graph_history.get(self.graph_step).map(|s| s.cost),
        }
    }

    /// Navigate to next step
    pub fn next_step(&mut self) {
        let total = self.total_steps();
        let step = match self.view_mode {
            ViewMode::Ast => &mut self.ast_step,
            ViewMode::Graph => &mut self.graph_step,
        };
        if *step + 1 < total {
            *step += 1;
            self.selected_candidate = 0;
        }
    }

    /// Navigate to previous step
    pub fn prev_step(&mut self) {
        let step = match self.view_mode {
            ViewMode::Ast => &mut self.ast_step,
            ViewMode::Graph => &mut self.graph_step,
        };
        if *step > 0 {
            *step -= 1;
            self.selected_candidate = 0;
        }
    }

    /// Select next candidate
    pub fn next_candidate(&mut self) {
        let count = match self.view_mode {
            ViewMode::Ast => self.ast_candidates().len(),
            ViewMode::Graph => self.graph_candidates().len(),
        };
        if self.selected_candidate + 1 < count {
            self.selected_candidate += 1;
        }
    }

    /// Select previous candidate
    pub fn prev_candidate(&mut self) {
        if self.selected_candidate > 0 {
            self.selected_candidate -= 1;
        }
    }
}
