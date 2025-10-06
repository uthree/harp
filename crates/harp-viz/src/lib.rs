use eframe::egui;
use egui_snarl::ui::{PinInfo, SnarlStyle, SnarlViewer};
use egui_snarl::{InPin, InPinId, NodeId, OutPin, OutPinId, Snarl};
use harp::graph::{Graph, GraphNode};
use std::collections::HashMap;

pub struct GraphVisualizerApp {
    snarl: Snarl<GraphNodeData>,
    sample_graph_loaded: bool,
    // Optimization snapshots
    snapshots: Vec<harp::opt::graph::OptimizationSnapshot>,
    current_snapshot_index: usize,
}

impl GraphVisualizerApp {
    /// グローバルログから起動
    pub fn from_global_snapshots(snapshots: Vec<harp::opt::graph::OptimizationSnapshot>) -> Self {
        let mut snarl = Snarl::new();
        let current_index = if snapshots.is_empty() { 0 } else { 0 };

        // Convert initial graph to snarl
        if !snapshots.is_empty() {
            GraphVisualizerApp::convert_graph_to_snarl_static(&snapshots[0].graph, &mut snarl);
        }

        Self {
            snarl,
            sample_graph_loaded: !snapshots.is_empty(),
            snapshots,
            current_snapshot_index: current_index,
        }
    }
}

impl Default for GraphVisualizerApp {
    fn default() -> Self {
        let mut snarl = Snarl::new();

        // Create and load sample graph with optimization
        let mut graph = Graph::new();
        let a = graph.input(harp::ast::DType::F32, vec![10.into()]);
        let b = graph.input(harp::ast::DType::F32, vec![10.into()]);
        let c = graph.input(harp::ast::DType::F32, vec![10.into()]);

        let add = a.clone() + b.clone();
        let mul = add * c.clone();

        graph.output(mul);

        // Run optimization with logging
        use harp::opt::graph::GraphOptimizer;
        let mut optimizer = harp::opt::graph::GraphFusionOptimizer::new().with_logging();
        optimizer.optimize(&mut graph);

        // Get snapshots
        let snapshots = optimizer.snapshots.clone();
        let current_index = if snapshots.is_empty() { 0 } else { 0 };

        // Convert initial graph to snarl
        if !snapshots.is_empty() {
            GraphVisualizerApp::convert_graph_to_snarl_static(&snapshots[0].graph, &mut snarl);
        } else {
            GraphVisualizerApp::convert_graph_to_snarl_static(&graph, &mut snarl);
        }

        Self {
            snarl,
            sample_graph_loaded: true,
            snapshots,
            current_snapshot_index: current_index,
        }
    }
}

#[derive(Clone)]
pub struct GraphNodeData {
    label: String,
    dtype: String,
    view_info: String,
    num_inputs: usize,
}

impl SnarlViewer<GraphNodeData> for GraphVisualizerApp {
    fn title(&mut self, node: &GraphNodeData) -> String {
        node.label.clone()
    }

    fn outputs(&mut self, _node: &GraphNodeData) -> usize {
        1 // Each graph node has one output
    }

    fn inputs(&mut self, node: &GraphNodeData) -> usize {
        node.num_inputs
    }

    fn show_input(
        &mut self,
        _pin: &InPin,
        _ui: &mut egui::Ui,
        _snarl: &mut Snarl<GraphNodeData>,
    ) -> PinInfo {
        PinInfo::square().with_fill(egui::Color32::from_rgb(100, 150, 200))
    }

    fn show_output(
        &mut self,
        _pin: &OutPin,
        _ui: &mut egui::Ui,
        _snarl: &mut Snarl<GraphNodeData>,
    ) -> PinInfo {
        PinInfo::square().with_fill(egui::Color32::from_rgb(100, 200, 150))
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<GraphNodeData>) -> bool {
        false // Disable graph menu to prevent editing
    }

    fn has_node_menu(&mut self, _node: &GraphNodeData) -> bool {
        false // Disable node menu to prevent editing
    }

    fn connect(&mut self, _from: &OutPin, _to: &InPin, _snarl: &mut Snarl<GraphNodeData>) {
        // Do nothing - disable connecting edges
    }

    fn disconnect(&mut self, _from: &OutPin, _to: &InPin, _snarl: &mut Snarl<GraphNodeData>) {
        // Do nothing - disable disconnecting edges
    }

    fn show_graph_menu(
        &mut self,
        _pos: egui::Pos2,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<GraphNodeData>,
    ) {
        ui.label("Graph Menu");
        if ui.button("Load Sample Graph").clicked() {
            self.load_sample_graph(snarl);
            ui.close();
        }
    }

    fn has_body(&mut self, _node: &GraphNodeData) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<GraphNodeData>,
    ) {
        let node_data = &snarl[node];
        ui.vertical(|ui| {
            ui.label(format!("Type: {}", node_data.dtype));
            ui.label(format!("Shape: {}", node_data.view_info));
        });
    }
}

impl GraphVisualizerApp {
    fn load_sample_graph(&mut self, snarl: &mut Snarl<GraphNodeData>) {
        if self.sample_graph_loaded {
            return;
        }

        // Create a simple sample graph: (a + b) * c
        let mut graph = Graph::new();
        let a = graph.input(harp::ast::DType::F32, vec![10.into()]);
        let b = graph.input(harp::ast::DType::F32, vec![10.into()]);
        let c = graph.input(harp::ast::DType::F32, vec![10.into()]);

        let add = a.clone() + b.clone();
        let mul = add * c.clone();

        graph.output(mul);

        // Convert graph to snarl
        self.convert_graph_to_snarl(&graph, snarl);

        self.sample_graph_loaded = true;
    }

    fn convert_graph_to_snarl_static(graph: &Graph, snarl: &mut Snarl<GraphNodeData>) {
        Self::convert_graph_to_snarl_impl(graph, snarl);
    }

    fn convert_graph_to_snarl(&self, graph: &Graph, snarl: &mut Snarl<GraphNodeData>) {
        Self::convert_graph_to_snarl_impl(graph, snarl);
    }

    fn convert_graph_to_snarl_impl(graph: &Graph, snarl: &mut Snarl<GraphNodeData>) {
        // Replace snarl with a new empty one
        *snarl = Snarl::new();
        let mut node_map: HashMap<GraphNode, NodeId> = HashMap::new();
        let mut visited: HashMap<GraphNode, ()> = HashMap::new();
        let mut input_indices: HashMap<GraphNode, usize> = HashMap::new();
        let mut input_counter: usize = 0;

        // Process all output nodes
        for (output_idx, output_node) in graph.outputs.iter().enumerate() {
            Self::add_node_recursive(
                output_node,
                snarl,
                &mut node_map,
                &mut visited,
                &mut input_indices,
                &mut input_counter,
                egui::pos2(500.0, 100.0 + output_idx as f32 * 150.0),
            );
        }
    }

    fn add_node_recursive(
        graph_node: &GraphNode,
        snarl: &mut Snarl<GraphNodeData>,
        node_map: &mut HashMap<GraphNode, NodeId>,
        visited: &mut HashMap<GraphNode, ()>,
        input_indices: &mut HashMap<GraphNode, usize>,
        input_counter: &mut usize,
        pos: egui::Pos2,
    ) -> NodeId {
        // If already processed, return existing node
        if let Some(&node_id) = node_map.get(graph_node) {
            return node_id;
        }

        // Mark as visited
        visited.insert(graph_node.clone(), ());

        // Use Display trait from harp core for GraphOp and DType
        let label = format!("{}", graph_node.op);
        let dtype_str = format!("{}", graph_node.dtype);

        // Format view using Display trait
        let view_str = format!("{}", graph_node.view);

        // Get input count for this node
        let num_inputs = graph_node.input_nodes().len();

        // Create node data
        let node_data = GraphNodeData {
            label,
            dtype: dtype_str,
            view_info: view_str,
            num_inputs,
        };

        // Add this node to snarl
        let node_id = snarl.insert_node(pos, node_data);
        node_map.insert(graph_node.clone(), node_id);

        // Process input nodes recursively
        let input_nodes = graph_node.input_nodes();
        for (input_idx, input_node) in input_nodes.iter().enumerate() {
            let input_pos = egui::pos2(pos.x - 250.0, pos.y + input_idx as f32 * 100.0);
            let input_node_id = Self::add_node_recursive(
                input_node,
                snarl,
                node_map,
                visited,
                input_indices,
                input_counter,
                input_pos,
            );

            // Connect input to this node
            snarl.connect(
                OutPinId {
                    node: input_node_id,
                    output: 0,
                },
                InPinId {
                    node: node_id,
                    input: input_idx,
                },
            );
        }

        node_id
    }
}

impl eframe::App for GraphVisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Harp Computational Graph Visualizer");

            // Snapshot navigation controls
            if !self.snapshots.is_empty() {
                ui.horizontal(|ui| {
                    if ui.button("◀ Previous").clicked() && self.current_snapshot_index > 0 {
                        self.current_snapshot_index -= 1;
                        Self::convert_graph_to_snarl_static(
                            &self.snapshots[self.current_snapshot_index].graph,
                            &mut self.snarl,
                        );
                    }

                    ui.label(format!(
                        "Step {}/{}: {}",
                        self.current_snapshot_index + 1,
                        self.snapshots.len(),
                        self.snapshots[self.current_snapshot_index].description
                    ));

                    if ui.button("Next ▶").clicked()
                        && self.current_snapshot_index < self.snapshots.len() - 1
                    {
                        self.current_snapshot_index += 1;
                        Self::convert_graph_to_snarl_static(
                            &self.snapshots[self.current_snapshot_index].graph,
                            &mut self.snarl,
                        );
                    }
                });
            }

            if !self.sample_graph_loaded && ui.button("Load Sample Graph").clicked() {
                let mut temp_snarl = Snarl::new();
                std::mem::swap(&mut temp_snarl, &mut self.snarl);
                self.load_sample_graph(&mut temp_snarl);
                self.snarl = temp_snarl;
            }

            ui.separator();

            // Show the graph
            // We need to temporarily take ownership of snarl to avoid borrow conflicts
            let mut snarl = Snarl::new();
            std::mem::swap(&mut snarl, &mut self.snarl);
            snarl.show(self, &SnarlStyle::new(), egui::Id::new("graph_snarl"), ui);
            self.snarl = snarl;
        });
    }
}

/// グローバルログからビジュアライザーを起動
pub fn launch_with_global_snapshots(
    snapshots: Vec<harp::opt::graph::OptimizationSnapshot>,
) -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Harp Optimization Visualizer",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(GraphVisualizerApp::from_global_snapshots(
                snapshots,
            )))
        }),
    )
}
