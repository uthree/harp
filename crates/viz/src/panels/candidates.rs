//! Candidates panel for displaying optimization alternatives

use egui::{Response, RichText, Ui, Widget};

/// Information about a candidate
#[derive(Clone)]
pub struct CandidateInfo {
    pub cost: f32,
    pub suggester_name: Option<String>,
    pub description: String,
}

/// Candidates panel for displaying and selecting optimization alternatives
pub struct CandidatesPanel<'a> {
    selected: &'a mut usize,
    candidates: &'a [CandidateInfo],
}

impl<'a> CandidatesPanel<'a> {
    pub fn new(selected: &'a mut usize, candidates: &'a [CandidateInfo]) -> Self {
        Self {
            selected,
            candidates,
        }
    }
}

impl Widget for CandidatesPanel<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                for (i, candidate) in self.candidates.iter().enumerate() {
                    let is_selected = *self.selected == i;

                    let text = format!(
                        "[{}] cost={:.2} {}",
                        i,
                        candidate.cost,
                        candidate
                            .suggester_name
                            .as_deref()
                            .unwrap_or(if i == 0 { "(selected)" } else { "(alt)" })
                    );

                    let label = if is_selected {
                        RichText::new(format!("> {}", text)).strong()
                    } else {
                        RichText::new(format!("  {}", text))
                    };

                    if ui.selectable_label(is_selected, label).clicked() {
                        *self.selected = i;
                    }

                    // Show description tooltip
                    if !candidate.description.is_empty() {
                        ui.small(&candidate.description);
                    }
                }
            });

        ui.allocate_response(ui.available_size(), egui::Sense::hover())
    }
}
