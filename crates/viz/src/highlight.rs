//! Syntax highlighting for egui

use egui::Color32;
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;

use crate::panels::code_panel::StyledSpan;

/// Code highlighter
pub struct CodeHighlighter {
    syntax_set: SyntaxSet,
    theme_set: ThemeSet,
}

impl CodeHighlighter {
    /// Create a new highlighter
    pub fn new() -> Self {
        Self {
            syntax_set: SyntaxSet::load_defaults_newlines(),
            theme_set: ThemeSet::load_defaults(),
        }
    }

    /// Highlight code and return styled strings per line
    ///
    /// Returns: Vector of lines, each containing spans for that line
    pub fn highlight(&self, code: &str) -> Vec<Vec<StyledSpan>> {
        // Use C syntax (most C-like)
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
                        let egui_span = convert_syntect_style(&style, text);
                        // Remove newline character
                        if !egui_span.text.is_empty() {
                            line_spans.push(egui_span);
                        }
                    }
                }
                Err(_) => {
                    // On highlight failure, add as plain text
                    let text = line.trim_end_matches('\n').to_string();
                    if !text.is_empty() {
                        line_spans.push(StyledSpan::plain(text));
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

/// Convert syntect style to StyledSpan
fn convert_syntect_style(style: &syntect::highlighting::Style, text: &str) -> StyledSpan {
    let color = Color32::from_rgb(style.foreground.r, style.foreground.g, style.foreground.b);
    let text = text.trim_end_matches('\n').to_string();

    let mut span = StyledSpan::new(text, color);
    span.bold = style.font_style.contains(FontStyle::BOLD);
    span.italic = style.font_style.contains(FontStyle::ITALIC);

    span
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
        // 1 line of code
        assert_eq!(result.len(), 1);
        assert!(!result[0].is_empty());
    }

    #[test]
    fn test_highlight_multiline() {
        let highlighter = CodeHighlighter::new();
        let code = "void foo() {\n    int x = 1;\n}\n";
        let result = highlighter.highlight(code);
        // 3 lines (void foo() {, int x = 1;, })
        assert_eq!(result.len(), 3);
        // First line has spans
        assert!(!result[0].is_empty());
    }
}
