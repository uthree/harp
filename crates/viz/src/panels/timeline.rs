//! Timeline panel for step navigation

use egui::{Response, Ui, Widget};

/// Timeline panel for navigating optimization steps
pub struct TimelinePanel<'a> {
    current_step: &'a mut usize,
    total_steps: usize,
    cost: Option<f32>,
}

impl<'a> TimelinePanel<'a> {
    pub fn new(current_step: &'a mut usize, total_steps: usize, cost: Option<f32>) -> Self {
        Self {
            current_step,
            total_steps,
            cost,
        }
    }
}

impl Widget for TimelinePanel<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        ui.horizontal(|ui| {
            // Previous step button
            if ui
                .add_enabled(*self.current_step > 0, egui::Button::new("◀"))
                .clicked()
            {
                *self.current_step = self.current_step.saturating_sub(1);
            }

            // Step indicator
            ui.label(format!(
                "Step {}/{}",
                *self.current_step + 1,
                self.total_steps
            ));

            // Next step button
            if ui
                .add_enabled(
                    *self.current_step + 1 < self.total_steps,
                    egui::Button::new("▶"),
                )
                .clicked()
            {
                *self.current_step += 1;
            }

            ui.separator();

            // Cost display
            if let Some(cost) = self.cost {
                ui.label(format!("Cost: {:.2e}", cost));
            } else {
                ui.label("Cost: N/A");
            }
        })
        .response
    }
}
