//! 最適化履歴可視化モジュール
//!
//! ratatuiを使用してターミナルUIで最適化の履歴を可視化する。

mod app;
mod events;
mod highlight;
mod ui;

pub use app::App;
pub use highlight::CodeHighlighter;
// run_with_renderer is a public function defined in this module

use std::io::{self, IsTerminal};

#[cfg(unix)]
use std::fs::File;

use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::prelude::*;

use crate::opt::ast::history::OptimizationHistory;
use crate::renderer::c_like::{CLikeRenderer, GenericRenderer};

/// TUI実行時のエラー
#[derive(Debug)]
pub enum VizError {
    /// ターミナルがインタラクティブでない
    NotATty,
    /// IOエラー
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

/// 最適化履歴を可視化するTUIを起動（デフォルトレンダラー）
///
/// # Arguments
/// * `history` - 表示する最適化履歴
///
/// # Returns
/// * `Result<(), VizError>` - 成功時はOk、エラー時はErr
///
/// # Example
/// ```ignore
/// use harp::opt::ast::history::OptimizationHistory;
/// use harp::viz::run;
///
/// let history = OptimizationHistory::new();
/// // ... add snapshots to history ...
/// run(history)?;
/// ```
pub fn run(history: OptimizationHistory) -> Result<(), VizError> {
    run_with_renderer(history, GenericRenderer::new())
}

/// カスタムレンダラーで最適化履歴を可視化するTUIを起動
///
/// # Arguments
/// * `history` - 表示する最適化履歴
/// * `renderer` - 使用するレンダラー（例：OpenCLRenderer）
///
/// # Returns
/// * `Result<(), VizError>` - 成功時はOk、エラー時はErr
///
/// # Example
/// ```ignore
/// use harp::opt::ast::history::OptimizationHistory;
/// use harp::renderer::opencl::OpenCLRenderer;
/// use harp::viz::run_with_renderer;
///
/// let history = OptimizationHistory::new();
/// let renderer = OpenCLRenderer::new();
/// run_with_renderer(history, renderer)?;
/// ```
pub fn run_with_renderer<R>(history: OptimizationHistory, renderer: R) -> Result<(), VizError>
where
    R: CLikeRenderer + Clone + 'static,
{
    // stdoutがTTYかチェック
    let stdout = io::stdout();
    if stdout.is_terminal() {
        // stdoutがTTYの場合は直接使用
        run_with_stdout_generic(history, renderer)
    } else {
        // stdoutがTTYでない場合は/dev/ttyを試す（Unix系のみ）
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

/// stdoutを使用してTUIを実行（ジェネリック）
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

/// /dev/ttyを使用してTUIを実行（Unix系のみ、ジェネリック）
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

/// メインループ（ジェネリック）
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
