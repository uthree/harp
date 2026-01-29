//! Graph panel for displaying computation graphs using egui-snarl

use egui::{Color32, Response, Ui, Widget};
use egui_snarl::ui::{PinInfo, SnarlStyle, SnarlViewer};
use egui_snarl::Snarl;

use crate::convert::VizNode;

/// Graph panel for displaying computation graphs
pub struct GraphPanel<'a> {
    snarl: &'a mut Snarl<VizNode>,
    title: String,
}

impl<'a> GraphPanel<'a> {
    pub fn new(snarl: &'a mut Snarl<VizNode>, title: impl Into<String>) -> Self {
        Self {
            snarl,
            title: title.into(),
        }
    }
}

impl Widget for GraphPanel<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        egui::Frame::canvas(ui.style())
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.heading(&self.title);
                ui.separator();

                let style = SnarlStyle::default();
                self.snarl.show(&mut GraphVizViewer, &style, "graph", ui);
            });

        ui.allocate_response(ui.available_size(), egui::Sense::hover())
    }
}

/// Viewer implementation for VizNode
pub struct GraphVizViewer;

impl SnarlViewer<VizNode> for GraphVizViewer {
    fn title(&mut self, node: &VizNode) -> String {
        // Very compact: "Op [shape]" e.g. "Sum [64×64]"
        format!("{} [{}]", node.op_type, node.shape_str)
    }

    fn inputs(&mut self, node: &VizNode) -> usize {
        node.input_count
    }

    fn outputs(&mut self, _node: &VizNode) -> usize {
        1
    }

    fn show_input(
        &mut self,
        _pin: &egui_snarl::InPin,
        _ui: &mut Ui,
        _scale: f32,
        _snarl: &mut Snarl<VizNode>,
    ) -> PinInfo {
        PinInfo::circle().with_fill(Color32::LIGHT_BLUE)
    }

    fn show_output(
        &mut self,
        _pin: &egui_snarl::OutPin,
        _ui: &mut Ui,
        _scale: f32,
        _snarl: &mut Snarl<VizNode>,
    ) -> PinInfo {
        PinInfo::circle().with_fill(Color32::LIGHT_GREEN)
    }

    fn show_body(
        &mut self,
        node: egui_snarl::NodeId,
        _inputs: &[egui_snarl::InPin],
        _outputs: &[egui_snarl::OutPin],
        ui: &mut Ui,
        _scale: f32,
        snarl: &mut Snarl<VizNode>,
    ) {
        let viz_node = &snarl[node];

        // Only show name if present and different from op_type
        if let Some(ref name) = viz_node.name {
            if name != &viz_node.op_type {
                ui.small(name);
            }
        }
    }

    fn has_body(&mut self, node: &VizNode) -> bool {
        // Only show body if there's a custom name
        node.name.as_ref().map_or(false, |n| n != &node.op_type)
    }
}
