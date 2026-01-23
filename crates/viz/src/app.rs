//! Application state

use eclat::ast::AstNode;
use eclat::ast::renderer::render_ast_with;
use eclat::backend::renderer::{CLikeRenderer, GenericRenderer};
use eclat::opt::ast::history::{OptimizationHistory, OptimizationSnapshot};

use super::highlight::CodeHighlighter;

/// Visualization application state (generic renderer support)
pub struct App<R: CLikeRenderer + Clone = GenericRenderer> {
    /// Optimization history
    history: OptimizationHistory,
    /// Current step being displayed
    current_step: usize,
    /// Currently selected candidate (0=selected, 1+=alternatives)
    selected_candidate: usize,
    /// Code highlighter
    highlighter: CodeHighlighter,
    /// Quit flag
    should_quit: bool,
    /// Code renderer
    renderer: R,
    /// Code scroll offset
    scroll_offset: u16,
}

impl App<GenericRenderer> {
    /// Create a new application (default renderer)
    pub fn new(history: OptimizationHistory) -> Self {
        Self {
            history,
            current_step: 0,
            selected_candidate: 0,
            highlighter: CodeHighlighter::new(),
            should_quit: false,
            renderer: GenericRenderer::new(),
            scroll_offset: 0,
        }
    }
}

impl<R: CLikeRenderer + Clone> App<R> {
    /// Create an application with a custom renderer
    pub fn with_renderer(history: OptimizationHistory, renderer: R) -> Self {
        Self {
            history,
            current_step: 0,
            selected_candidate: 0,
            highlighter: CodeHighlighter::new(),
            should_quit: false,
            renderer,
            scroll_offset: 0,
        }
    }

    /// Get current step number
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get total number of steps
    pub fn total_steps(&self) -> usize {
        self.history.len()
    }

    /// Get selected candidate index
    pub fn selected_candidate(&self) -> usize {
        self.selected_candidate
    }

    /// Get current step's snapshot
    pub fn current_snapshot(&self) -> Option<&OptimizationSnapshot> {
        self.history.get(self.current_step)
    }

    /// Get currently selected AST
    pub fn current_ast(&self) -> Option<&AstNode> {
        let snapshot = self.current_snapshot()?;
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

    /// Highlight current code (spans per line)
    pub fn highlight_current_code(&self) -> Vec<Vec<(ratatui::style::Style, String)>> {
        let code = self.render_current_code();
        self.highlighter.highlight(&code)
    }

    /// Get number of candidates for current step (selected + alternatives)
    pub fn candidate_count(&self) -> usize {
        match self.current_snapshot() {
            Some(snapshot) => 1 + snapshot.alternatives.len(),
            None => 0,
        }
    }

    /// Go to next step
    pub fn next_step(&mut self) {
        if self.current_step + 1 < self.history.len() {
            self.current_step += 1;
            self.selected_candidate = 0;
            self.scroll_offset = 0;
        }
    }

    /// Go to previous step
    pub fn prev_step(&mut self) {
        if self.current_step > 0 {
            self.current_step -= 1;
            self.selected_candidate = 0;
            self.scroll_offset = 0;
        }
    }

    /// Select next candidate
    pub fn next_candidate(&mut self) {
        let count = self.candidate_count();
        if count > 0 && self.selected_candidate + 1 < count {
            self.selected_candidate += 1;
            self.scroll_offset = 0;
        }
    }

    /// Select previous candidate
    pub fn prev_candidate(&mut self) {
        if self.selected_candidate > 0 {
            self.selected_candidate -= 1;
            self.scroll_offset = 0;
        }
    }

    /// Get quit flag
    pub fn should_quit(&self) -> bool {
        self.should_quit
    }

    /// Set quit flag
    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    /// Get code scroll offset
    pub fn scroll_offset(&self) -> u16 {
        self.scroll_offset
    }

    /// Scroll code down
    pub fn scroll_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(1);
    }

    /// Scroll code up
    pub fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
    }

    /// Get reference to history
    pub fn history(&self) -> &OptimizationHistory {
        &self.history
    }
}
