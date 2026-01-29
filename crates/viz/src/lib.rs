//! Visualization tools for Eclat optimization history
//!
//! This crate provides a GUI for visualizing the optimization history
//! of Eclat's AST and Graph optimizers using egui/eframe.

mod app;
mod convert;
mod graph_history;
mod highlight;
mod panels;
mod state;

pub use app::VizApp;
pub use convert::{VizNode, graph_to_snarl};
pub use graph_history::{GraphAlternativeCandidate, GraphOptimizationHistory, GraphOptimizationSnapshot};
pub use highlight::CodeHighlighter;
pub use state::{AppState, ViewMode};

use eclat::backend::renderer::{CLikeRenderer, GenericRenderer};
use eclat::opt::ast::history::OptimizationHistory;

/// Error type for visualization
#[derive(Debug)]
pub enum VizError {
    /// eframe error
    Eframe(eframe::Error),
}

impl std::fmt::Display for VizError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VizError::Eframe(e) => write!(f, "eframe error: {}", e),
        }
    }
}

impl std::error::Error for VizError {}

impl From<eframe::Error> for VizError {
    fn from(e: eframe::Error) -> Self {
        VizError::Eframe(e)
    }
}

/// Launch GUI to visualize AST optimization history (default renderer)
///
/// # Arguments
/// * `history` - AST optimization history to display
///
/// # Returns
/// * `Result<(), VizError>` - Ok on success, Err on failure
///
/// # Example
/// ```ignore
/// use eclat::opt::ast::history::OptimizationHistory;
/// use eclat_viz::run;
///
/// let history = OptimizationHistory::new();
/// // ... add snapshots to history ...
/// run(history)?;
/// ```
pub fn run(history: OptimizationHistory) -> Result<(), VizError> {
    run_with_renderer(history, GenericRenderer::new())
}

/// Launch GUI to visualize AST optimization history with a custom renderer
///
/// # Arguments
/// * `history` - AST optimization history to display
/// * `renderer` - Renderer to use
///
/// # Returns
/// * `Result<(), VizError>` - Ok on success, Err on failure
pub fn run_with_renderer<R>(history: OptimizationHistory, renderer: R) -> Result<(), VizError>
where
    R: CLikeRenderer + Clone + 'static,
{
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Eclat Optimization Visualizer"),
        ..Default::default()
    };

    eframe::run_native(
        "Eclat Viz",
        options,
        Box::new(|_cc| Ok(Box::new(VizApp::with_renderer(history, renderer)))),
    )?;

    Ok(())
}

/// Launch GUI to visualize both AST and Graph optimization history
///
/// # Arguments
/// * `ast_history` - AST optimization history
/// * `graph_history` - Graph optimization history
/// * `renderer` - Renderer to use
///
/// # Returns
/// * `Result<(), VizError>` - Ok on success, Err on failure
pub fn run_with_both<R>(
    ast_history: OptimizationHistory,
    graph_history: GraphOptimizationHistory,
    renderer: R,
) -> Result<(), VizError>
where
    R: CLikeRenderer + Clone + 'static,
{
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Eclat Optimization Visualizer"),
        ..Default::default()
    };

    eframe::run_native(
        "Eclat Viz",
        options,
        Box::new(|_cc| {
            Ok(Box::new(VizApp::with_both_histories(
                ast_history,
                graph_history,
                renderer,
            )))
        }),
    )?;

    Ok(())
}
