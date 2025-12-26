//! シンタックスハイライト

use ratatui::style::{Color, Modifier, Style};
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

/// コードハイライター
pub struct CodeHighlighter {
    syntax_set: SyntaxSet,
    theme_set: ThemeSet,
}

impl CodeHighlighter {
    /// 新しいハイライターを作成
    pub fn new() -> Self {
        Self {
            syntax_set: SyntaxSet::load_defaults_newlines(),
            theme_set: ThemeSet::load_defaults(),
        }
    }

    /// コードをハイライトして行ごとのスタイル付き文字列を返す
    ///
    /// 戻り値: 各行のスパンのベクタ（外側のVecが行、内側のVecがその行のスパン）
    pub fn highlight(&self, code: &str) -> Vec<Vec<(Style, String)>> {
        // C言語のシンタックスを使用（最もC-likeに近い）
        let syntax = self
            .syntax_set
            .find_syntax_by_extension("c")
            .unwrap_or_else(|| self.syntax_set.find_syntax_plain_text());

        let theme = &self.theme_set.themes["base16-ocean.dark"];
        let mut highlighter = HighlightLines::new(syntax, theme);

        let mut result = Vec::new();

        for line in LinesWithEndings::from(code) {
            let mut line_spans = Vec::new();
            match highlighter.highlight_line(line, &self.syntax_set) {
                Ok(ranges) => {
                    for (style, text) in ranges {
                        let ratatui_style = convert_syntect_style(&style);
                        // 改行文字を除去（ratatuiのLineは改行を含まない）
                        let text = text.trim_end_matches('\n').to_string();
                        if !text.is_empty() {
                            line_spans.push((ratatui_style, text));
                        }
                    }
                }
                Err(_) => {
                    // ハイライトに失敗した場合はプレーンテキストとして追加
                    let text = line.trim_end_matches('\n').to_string();
                    if !text.is_empty() {
                        line_spans.push((Style::default(), text));
                    }
                }
            }
            result.push(line_spans);
        }

        result
    }
}

impl Default for CodeHighlighter {
    fn default() -> Self {
        Self::new()
    }
}

/// syntectのスタイルをratatuiのスタイルに変換
fn convert_syntect_style(style: &syntect::highlighting::Style) -> Style {
    let fg = Color::Rgb(style.foreground.r, style.foreground.g, style.foreground.b);

    let mut ratatui_style = Style::default().fg(fg);

    if style.font_style.contains(FontStyle::BOLD) {
        ratatui_style = ratatui_style.add_modifier(Modifier::BOLD);
    }
    if style.font_style.contains(FontStyle::ITALIC) {
        ratatui_style = ratatui_style.add_modifier(Modifier::ITALIC);
    }
    if style.font_style.contains(FontStyle::UNDERLINE) {
        ratatui_style = ratatui_style.add_modifier(Modifier::UNDERLINED);
    }

    ratatui_style
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlighter_creation() {
        let highlighter = CodeHighlighter::new();
        assert!(!highlighter.syntax_set.syntaxes().is_empty());
    }

    #[test]
    fn test_highlight_simple_code() {
        let highlighter = CodeHighlighter::new();
        let code = "int x = 42;";
        let result = highlighter.highlight(code);
        // 1行のコード
        assert_eq!(result.len(), 1);
        assert!(!result[0].is_empty());
    }

    #[test]
    fn test_highlight_multiline() {
        let highlighter = CodeHighlighter::new();
        let code = "void foo() {\n    int x = 1;\n}\n";
        let result = highlighter.highlight(code);
        // 3行（void foo() {, int x = 1;, }）
        assert_eq!(result.len(), 3);
        // 最初の行にスパンがある
        assert!(!result[0].is_empty());
    }
}
