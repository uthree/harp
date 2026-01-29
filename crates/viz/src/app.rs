//! Main application with eframe::App implementation

use eframe::egui;

use eclat::backend::renderer::{CLikeRenderer, GenericRenderer};
use eclat::opt::ast::history::OptimizationHistory;

use crate::graph_history::GraphOptimizationHistory;
use crate::panels::{CandidatesPanel, CodePanel, GraphPanel, TimelinePanel};
use crate::state::{AppState, ViewMode};

/// Main visualization application
pub struct VizApp<R: CLikeRenderer + Clone = GenericRenderer> {
    state: AppState<R>,
}

impl VizApp<GenericRenderer> {
    /// Create a new application with AST history
    pub fn new(ast_history: OptimizationHistory) -> Self {
        Self {
            state: AppState::new(ast_history),
        }
    }
}

impl<R: CLikeRenderer + Clone> VizApp<R> {
    /// Create a new application with custom renderer
    pub fn with_renderer(ast_history: OptimizationHistory, renderer: R) -> Self {
        Self {
            state: AppState::with_renderer(ast_history, renderer),
        }
    }

    /// Create an application with both AST and graph history
    pub fn with_both_histories(
        ast_history: OptimizationHistory,
        graph_history: GraphOptimizationHistory,
        renderer: R,
    ) -> Self {
        Self {
            state: AppState::with_both_histories(ast_history, graph_history, renderer),
        }
    }

    /// Set the graph optimization history
    pub fn set_graph_history(&mut self, history: GraphOptimizationHistory) {
        self.state.set_graph_history(history);
    }
}

impl<R: CLikeRenderer + Clone + 'static> eframe::App for VizApp<R> {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle keyboard shortcuts
        self.handle_keyboard(ctx);

        // Top panel with timeline and view mode selector
        egui::TopBottomPanel::top("timeline").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Timeline navigation
                let mut step = self.state.current_step();
                let total = self.state.total_steps();
                let cost = self.state.current_cost();
                ui.add(TimelinePanel::new(&mut step, total, cost));
                if step != self.state.current_step() {
                    *self.state.current_step_mut() = step;
                    self.state.selected_candidate = 0;
                }

                ui.separator();

                // View mode selector
                ui.label("View:");
                if ui
                    .selectable_label(self.state.view_mode == ViewMode::Ast, "AST")
                    .clicked()
                {
                    self.state.view_mode = ViewMode::Ast;
                }
                if ui
                    .selectable_label(self.state.view_mode == ViewMode::Graph, "Graph")
                    .clicked()
                {
                    self.state.view_mode = ViewMode::Graph;
                }
            });
        });

        // Right panel with candidates
        egui::SidePanel::right("candidates")
            .resizable(true)
            .default_width(200.0)
            .show(ctx, |ui| {
                ui.heading("Candidates");
                ui.separator();

                let candidates = match self.state.view_mode {
                    ViewMode::Ast => self.state.ast_candidates(),
                    ViewMode::Graph => self.state.graph_candidates(),
                };
                ui.add(CandidatesPanel::new(
                    &mut self.state.selected_candidate,
                    &candidates,
                ));
            });

        // Central panel with main content
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.state.view_mode {
                ViewMode::Ast => {
                    self.show_ast_view(ui);
                }
                ViewMode::Graph => {
                    self.show_graph_view(ui);
                }
            }
        });
    }
}

impl<R: CLikeRenderer + Clone> VizApp<R> {
    fn handle_keyboard(&mut self, ctx: &egui::Context) {
        ctx.input(|i| {
            // Navigation
            if i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::H) {
                self.state.prev_step();
            }
            if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::L) {
                self.state.next_step();
            }
            if i.key_pressed(egui::Key::ArrowUp) || i.key_pressed(egui::Key::K) {
                self.state.prev_candidate();
            }
            if i.key_pressed(egui::Key::ArrowDown) || i.key_pressed(egui::Key::J) {
                self.state.next_candidate();
            }

            // View mode shortcuts
            if i.key_pressed(egui::Key::Num1) {
                self.state.view_mode = ViewMode::Ast;
            }
            if i.key_pressed(egui::Key::Num2) {
                self.state.view_mode = ViewMode::Graph;
            }
        });
    }

    fn show_ast_view(&self, ui: &mut egui::Ui) {
        let highlighted = self.state.highlighted_code();
        let title = format!(
            "Code (Candidate {}/{})",
            self.state.selected_candidate + 1,
            self.state.ast_candidates().len().max(1)
        );
        ui.add(CodePanel::new(&highlighted, title));
    }

    fn show_graph_view(&mut self, ui: &mut egui::Ui) {
        let title = format!(
            "Graph (Step {}/{})",
            self.state.graph_step + 1,
            self.state.graph_history.len().max(1)
        );
        let snarl = self.state.current_snarl();
        ui.add(GraphPanel::new(snarl, title));
    }
}
