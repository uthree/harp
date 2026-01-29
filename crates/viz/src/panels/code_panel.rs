//! Code panel for displaying syntax-highlighted source code

use egui::{Color32, Response, RichText, Ui, Widget};

/// A styled text span
#[derive(Clone)]
pub struct StyledSpan {
    pub text: String,
    pub color: Color32,
    pub bold: bool,
    pub italic: bool,
}

impl StyledSpan {
    pub fn new(text: String, color: Color32) -> Self {
        Self {
            text,
            color,
            bold: false,
            italic: false,
        }
    }

    pub fn plain(text: String) -> Self {
        Self {
            text,
            color: Color32::WHITE,
            bold: false,
            italic: false,
        }
    }
}

/// Code panel for displaying syntax-highlighted source code
pub struct CodePanel<'a> {
    lines: &'a [Vec<StyledSpan>],
    title: String,
}

impl<'a> CodePanel<'a> {
    pub fn new(lines: &'a [Vec<StyledSpan>], title: impl Into<String>) -> Self {
        Self {
            lines,
            title: title.into(),
        }
    }
}

impl Widget for CodePanel<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        egui::Frame::canvas(ui.style())
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.heading(&self.title);
                ui.separator();

                egui::ScrollArea::both()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        ui.style_mut().override_font_id =
                            Some(egui::FontId::monospace(14.0));

                        for (line_num, spans) in self.lines.iter().enumerate() {
                            ui.horizontal(|ui| {
                                // Remove spacing between labels
                                ui.spacing_mut().item_spacing.x = 0.0;

                                // Line number (with some padding after)
                                ui.label(
                                    RichText::new(format!("{:4} ", line_num + 1))
                                        .color(Color32::GRAY),
                                );

                                // Code spans - no spacing between them
                                for span in spans {
                                    let mut text = RichText::new(&span.text).color(span.color);
                                    if span.bold {
                                        text = text.strong();
                                    }
                                    if span.italic {
                                        text = text.italics();
                                    }
                                    ui.label(text);
                                }
                            });
                        }
                    });
            });

        ui.allocate_response(ui.available_size(), egui::Sense::hover())
    }
}
