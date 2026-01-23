//! Visualization tools for Eclat optimization history
//!
//! This crate provides a TUI (Terminal User Interface) for visualizing
//! the optimization history of Eclat's AST optimizer.

mod app;
mod events;
mod highlight;
mod ui;

pub use app::App;
pub use highlight::CodeHighlighter;

use std::io::{self, IsTerminal};

#[cfg(unix)]
use std::fs::File;

use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::prelude::*;

use eclat::backend::renderer::{CLikeRenderer, GenericRenderer};
use eclat::opt::ast::history::OptimizationHistory;

/// TUI runtime error
#[derive(Debug)]
pub enum VizError {
    /// Terminal is not interactive
    NotATty,
    /// IO error
    Io(io::Error),
}

impl std::fmt::Display for VizError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VizError::NotATty => write!(
                f,
                "Not a terminal. Please run directly in a terminal:\n  cargo run --features viz --example viz_demo"
            ),
            VizError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for VizError {}

impl From<io::Error> for VizError {
    fn from(e: io::Error) -> Self {
        VizError::Io(e)
    }
}

/// Launch TUI to visualize optimization history (default renderer)
///
/// # Arguments
/// * `history` - Optimization history to display
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

/// Launch TUI to visualize optimization history with a custom renderer
///
/// # Arguments
/// * `history` - Optimization history to display
/// * `renderer` - Renderer to use (e.g., OpenCLRenderer)
///
/// # Returns
/// * `Result<(), VizError>` - Ok on success, Err on failure
///
/// # Example
/// ```ignore
/// use eclat::opt::ast::history::OptimizationHistory;
/// use eclat_backend_opencl::OpenCLRenderer;
/// use eclat_viz::run_with_renderer;
///
/// let history = OptimizationHistory::new();
/// let renderer = OpenCLRenderer::new();
/// run_with_renderer(history, renderer)?;
/// ```
pub fn run_with_renderer<R>(history: OptimizationHistory, renderer: R) -> Result<(), VizError>
where
    R: CLikeRenderer + Clone + 'static,
{
    // Check if stdout is a TTY
    let stdout = io::stdout();
    if stdout.is_terminal() {
        // If stdout is a TTY, use it directly
        run_with_stdout_generic(history, renderer)
    } else {
        // If stdout is not a TTY, try /dev/tty (Unix only)
        #[cfg(unix)]
        {
            run_with_tty_generic(history, renderer)
        }
        #[cfg(not(unix))]
        {
            Err(VizError::NotATty)
        }
    }
}

/// Run TUI using stdout (generic)
fn run_with_stdout_generic<R>(history: OptimizationHistory, renderer: R) -> Result<(), VizError>
where
    R: CLikeRenderer + Clone + 'static,
{
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::with_renderer(history, renderer);
    let result = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

/// Run TUI using /dev/tty (Unix only, generic)
#[cfg(unix)]
fn run_with_tty_generic<R>(history: OptimizationHistory, renderer: R) -> Result<(), VizError>
where
    R: CLikeRenderer + Clone + 'static,
{
    let tty = File::options()
        .read(true)
        .write(true)
        .open("/dev/tty")
        .map_err(|_| VizError::NotATty)?;

    if !std::io::IsTerminal::is_terminal(&tty) {
        return Err(VizError::NotATty);
    }

    enable_raw_mode()?;
    let mut tty = tty;
    execute!(tty, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(tty);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::with_renderer(history, renderer);
    let result = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

/// Main loop (generic)
fn run_app<B: Backend, R: CLikeRenderer + Clone>(
    terminal: &mut Terminal<B>,
    app: &mut App<R>,
) -> Result<(), VizError> {
    loop {
        terminal.draw(|f| ui::draw(f, app))?;

        if let Some(action) = events::handle_events()? {
            match action {
                events::Action::Quit => break,
                events::Action::NextStep => app.next_step(),
                events::Action::PrevStep => app.prev_step(),
                events::Action::NextCandidate => app.next_candidate(),
                events::Action::PrevCandidate => app.prev_candidate(),
            }
        }

        if app.should_quit() {
            break;
        }
    }

    Ok(())
}
