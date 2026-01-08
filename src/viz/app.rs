//! アプリケーション状態

use crate::ast::AstNode;
use crate::ast::renderer::render_ast_with;
use crate::backend::renderer::{CLikeRenderer, GenericRenderer};
use crate::opt::ast::history::{OptimizationHistory, OptimizationSnapshot};

use super::highlight::CodeHighlighter;

/// 可視化アプリケーションの状態（ジェネリックレンダラー対応）
pub struct App<R: CLikeRenderer + Clone = GenericRenderer> {
    /// 最適化履歴
    history: OptimizationHistory,
    /// 現在表示中のステップ
    current_step: usize,
    /// 現在選択中の候補（0=選択された候補、1+=代替候補）
    selected_candidate: usize,
    /// コードハイライター
    highlighter: CodeHighlighter,
    /// 終了フラグ
    should_quit: bool,
    /// コードレンダラー
    renderer: R,
    /// コードのスクロール位置
    scroll_offset: u16,
}

impl App<GenericRenderer> {
    /// 新しいアプリケーションを作成（デフォルトレンダラー）
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
    /// カスタムレンダラーでアプリケーションを作成
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

    /// 現在のステップ番号を取得
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// 総ステップ数を取得
    pub fn total_steps(&self) -> usize {
        self.history.len()
    }

    /// 選択中の候補インデックスを取得
    pub fn selected_candidate(&self) -> usize {
        self.selected_candidate
    }

    /// 現在のステップのスナップショットを取得
    pub fn current_snapshot(&self) -> Option<&OptimizationSnapshot> {
        self.history.get(self.current_step)
    }

    /// 現在選択中のASTを取得
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

    /// 現在のASTをソースコードにレンダリング
    pub fn render_current_code(&self) -> String {
        match self.current_ast() {
            Some(ast) => render_ast_with(ast, &self.renderer),
            None => String::from("(No AST available)"),
        }
    }

    /// 現在のコードをハイライト（行ごとのスパン）
    pub fn highlight_current_code(&self) -> Vec<Vec<(ratatui::style::Style, String)>> {
        let code = self.render_current_code();
        self.highlighter.highlight(&code)
    }

    /// 現在のステップの候補数を取得（選択された候補 + 代替候補）
    pub fn candidate_count(&self) -> usize {
        match self.current_snapshot() {
            Some(snapshot) => 1 + snapshot.alternatives.len(),
            None => 0,
        }
    }

    /// 次のステップへ
    pub fn next_step(&mut self) {
        if self.current_step + 1 < self.history.len() {
            self.current_step += 1;
            self.selected_candidate = 0;
            self.scroll_offset = 0;
        }
    }

    /// 前のステップへ
    pub fn prev_step(&mut self) {
        if self.current_step > 0 {
            self.current_step -= 1;
            self.selected_candidate = 0;
            self.scroll_offset = 0;
        }
    }

    /// 次の候補を選択
    pub fn next_candidate(&mut self) {
        let count = self.candidate_count();
        if count > 0 && self.selected_candidate + 1 < count {
            self.selected_candidate += 1;
            self.scroll_offset = 0;
        }
    }

    /// 前の候補を選択
    pub fn prev_candidate(&mut self) {
        if self.selected_candidate > 0 {
            self.selected_candidate -= 1;
            self.scroll_offset = 0;
        }
    }

    /// 終了フラグを取得
    pub fn should_quit(&self) -> bool {
        self.should_quit
    }

    /// 終了フラグを設定
    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    /// コードのスクロール位置を取得
    pub fn scroll_offset(&self) -> u16 {
        self.scroll_offset
    }

    /// コードを下にスクロール
    pub fn scroll_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(1);
    }

    /// コードを上にスクロール
    pub fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
    }

    /// 履歴への参照を取得
    pub fn history(&self) -> &OptimizationHistory {
        &self.history
    }
}
