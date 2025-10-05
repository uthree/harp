use eframe::egui;
use egui_snarl::ui::{PinInfo, SnarlStyle, SnarlViewer};
use egui_snarl::{InPin, InPinId, NodeId, OutPin, OutPinId, Snarl};
use harp::graph::{Graph, GraphNode, GraphOp};
use std::collections::HashMap;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Harp Computational Graph Visualizer",
        options,
        Box::new(|_cc| Ok(Box::new(GraphVisualizerApp::default()))),
    )
}

#[derive(Default)]
struct GraphVisualizerApp {
    snarl: Snarl<GraphNodeData>,
    sample_graph_loaded: bool,
}

#[derive(Clone)]
struct GraphNodeData {
    label: String,
    op_type: String,
    dtype: String,
    view_info: String,
}

impl SnarlViewer<GraphNodeData> for GraphVisualizerApp {
    fn title(&mut self, node: &GraphNodeData) -> String {
        node.label.clone()
    }

    fn outputs(&mut self, _node: &GraphNodeData) -> usize {
        1 // Each graph node has one output
    }

    fn inputs(&mut self, node: &GraphNodeData) -> usize {
        // Count inputs based on operation type
        match node.op_type.as_str() {
            "Input" | "Const" => 0,
            "Elementwise" | "View" | "Contiguous" | "Cast" => 1,
            "Reduce" | "Cumulative" => 1,
            "FusedElementwise" => 2, // Simplified - could be more
            _ => 1,
        }
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut egui::Ui,
        _snarl: &mut Snarl<GraphNodeData>,
    ) -> PinInfo {
        ui.label(format!("in_{}", pin.id.input));
        PinInfo::square().with_fill(egui::Color32::from_rgb(100, 150, 200))
    }

    fn show_output(
        &mut self,
        _pin: &OutPin,
        ui: &mut egui::Ui,
        _snarl: &mut Snarl<GraphNodeData>,
    ) -> PinInfo {
        ui.label("out");
        PinInfo::square().with_fill(egui::Color32::from_rgb(100, 200, 150))
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<GraphNodeData>) -> bool {
        true
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
        ui.label(format!("{}", node_data.dtype));
        ui.label(format!("{}", node_data.view_info));
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

    fn convert_graph_to_snarl(&self, graph: &Graph, snarl: &mut Snarl<GraphNodeData>) {
        // Replace snarl with a new empty one
        *snarl = Snarl::new();
        let mut node_map: HashMap<GraphNode, NodeId> = HashMap::new();
        let mut visited: HashMap<GraphNode, ()> = HashMap::new();

        // Process all output nodes
        for (output_idx, output_node) in graph.outputs.iter().enumerate() {
            self.add_node_recursive(
                output_node,
                snarl,
                &mut node_map,
                &mut visited,
                egui::pos2(500.0, 100.0 + output_idx as f32 * 150.0),
            );
        }
    }

    fn add_node_recursive(
        &self,
        graph_node: &GraphNode,
        snarl: &mut Snarl<GraphNodeData>,
        node_map: &mut HashMap<GraphNode, NodeId>,
        visited: &mut HashMap<GraphNode, ()>,
        pos: egui::Pos2,
    ) -> NodeId {
        // If already processed, return existing node
        if let Some(&node_id) = node_map.get(graph_node) {
            return node_id;
        }

        // Mark as visited
        visited.insert(graph_node.clone(), ());

        // Get operation type and label
        let (op_type, label) = match &graph_node.op {
            GraphOp::Input => ("Input".to_string(), "Input".to_string()),
            GraphOp::Const(_) => ("Const".to_string(), "Const".to_string()),
            GraphOp::Elementwise(op) => {
                let op_name = match op {
                    harp::graph::ops::ElementwiseOp::Add(_, _) => "Add",
                    harp::graph::ops::ElementwiseOp::Mul(_, _) => "Mul",
                    harp::graph::ops::ElementwiseOp::Max(_, _) => "Max",
                    harp::graph::ops::ElementwiseOp::Mod(_, _) => "Mod",
                    harp::graph::ops::ElementwiseOp::Neg(_) => "Neg",
                    harp::graph::ops::ElementwiseOp::Recip(_) => "Recip",
                    harp::graph::ops::ElementwiseOp::Sin(_) => "Sin",
                    harp::graph::ops::ElementwiseOp::Sqrt(_) => "Sqrt",
                    harp::graph::ops::ElementwiseOp::Log2(_) => "Log2",
                    harp::graph::ops::ElementwiseOp::Exp2(_) => "Exp2",
                };
                ("Elementwise".to_string(), op_name.to_string())
            }
            GraphOp::Reduce(op, axis, _) => {
                let op_name = match op {
                    harp::graph::ops::ReduceOp::Add => "Sum",
                    harp::graph::ops::ReduceOp::Mul => "Product",
                    harp::graph::ops::ReduceOp::Max => "Max",
                };
                ("Reduce".to_string(), format!("{}[{}]", op_name, axis))
            }
            GraphOp::Cumulative(op, axis, _) => {
                let op_name = match op {
                    harp::graph::ops::CumulativeOp::Add => "CumSum",
                    harp::graph::ops::CumulativeOp::Mul => "CumProd",
                    harp::graph::ops::CumulativeOp::Max => "CumMax",
                };
                ("Cumulative".to_string(), format!("{}[{}]", op_name, axis))
            }
            GraphOp::View(_) => ("View".to_string(), "View".to_string()),
            GraphOp::Contiguous(_) => ("Contiguous".to_string(), "Contiguous".to_string()),
            GraphOp::Cast(_, dtype) => {
                let dtype_str = match dtype {
                    harp::ast::DType::F32 => "F32",
                    harp::ast::DType::Usize => "Usize",
                    harp::ast::DType::Isize => "Isize",
                    harp::ast::DType::Void => "Void",
                    harp::ast::DType::Ptr(_) => "Ptr",
                    harp::ast::DType::Vec(_, _) => "Vec",
                };
                ("Cast".to_string(), format!("Cast({})", dtype_str))
            }
            GraphOp::FusedElementwise(_, _) => (
                "FusedElementwise".to_string(),
                "Fused".to_string(),
            ),
            GraphOp::FusedReduce(op, axes, _) => {
                let op_name = match op {
                    harp::graph::ops::ReduceOp::Add => "Sum",
                    harp::graph::ops::ReduceOp::Mul => "Product",
                    harp::graph::ops::ReduceOp::Max => "Max",
                };
                ("FusedReduce".to_string(), format!("Fused{}[{:?}]", op_name, axes))
            }
            GraphOp::FusedElementwiseReduce(_, _, op, axes) => {
                let op_name = match op {
                    harp::graph::ops::ReduceOp::Add => "Sum",
                    harp::graph::ops::ReduceOp::Mul => "Product",
                    harp::graph::ops::ReduceOp::Max => "Max",
                };
                ("FusedElementwiseReduce".to_string(), format!("FusedER-{}[{:?}]", op_name, axes))
            }
            GraphOp::FusedElementwiseCumulative(_, _, op) => {
                let op_name = match op {
                    harp::graph::ops::CumulativeOp::Add => "CumSum",
                    harp::graph::ops::CumulativeOp::Mul => "CumProd",
                    harp::graph::ops::CumulativeOp::Max => "CumMax",
                };
                ("FusedElementwiseCumulative".to_string(), format!("FusedEC-{}", op_name))
            }
        };

        // Format dtype
        let dtype_str = match &graph_node.dtype {
            harp::ast::DType::F32 => "F32",
            harp::ast::DType::Usize => "Usize",
            harp::ast::DType::Isize => "Isize",
            harp::ast::DType::Void => "Void",
            harp::ast::DType::Ptr(_) => "Ptr",
            harp::ast::DType::Vec(_, _) => "Vec",
        };

        // Format shape - convert Vec<Expr> to a simple string like [10, 20]
        let shape_exprs = graph_node.view.shape();
        let shape_str = if shape_exprs.is_empty() {
            "[]".to_string()
        } else {
            let shape_parts: Vec<String> = shape_exprs
                .iter()
                .map(|expr| {
                    // Try to extract simple integer values
                    use harp::graph::shape::Expr;
                    match expr {
                        Expr::Const(n) => n.to_string(),
                        Expr::Var(name) => name.clone(),
                        _ => "?".to_string(),
                    }
                })
                .collect();
            format!("[{}]", shape_parts.join(", "))
        };

        // Create node data
        let node_data = GraphNodeData {
            label,
            op_type: op_type.clone(),
            dtype: dtype_str.to_string(),
            view_info: shape_str,
        };

        // Add this node to snarl
        let node_id = snarl.insert_node(pos, node_data);
        node_map.insert(graph_node.clone(), node_id);

        // Process input nodes recursively
        let input_nodes = self.get_input_nodes(graph_node);
        for (input_idx, input_node) in input_nodes.iter().enumerate() {
            let input_pos = egui::pos2(pos.x - 250.0, pos.y + input_idx as f32 * 100.0);
            let input_node_id =
                self.add_node_recursive(input_node, snarl, node_map, visited, input_pos);

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

    fn get_input_nodes(&self, node: &GraphNode) -> Vec<GraphNode> {
        match &node.op {
            GraphOp::Input | GraphOp::Const(_) => vec![],
            GraphOp::View(input) | GraphOp::Contiguous(input) | GraphOp::Cast(input, _) => {
                vec![input.clone()]
            }
            GraphOp::Reduce(_, _, input) | GraphOp::Cumulative(_, _, input) => {
                vec![input.clone()]
            }
            GraphOp::Elementwise(op) => self.get_elementwise_inputs(op),
            GraphOp::FusedElementwise(_, inputs) => inputs.clone(),
            GraphOp::FusedReduce(_, _, input) => vec![input.clone()],
            GraphOp::FusedElementwiseReduce(_, inputs, _, _) => inputs.clone(),
            GraphOp::FusedElementwiseCumulative(_, inputs, _) => inputs.clone(),
        }
    }

    fn get_elementwise_inputs(&self, op: &harp::graph::ops::ElementwiseOp) -> Vec<GraphNode> {
        use harp::graph::ops::ElementwiseOp;
        match op {
            ElementwiseOp::Add(a, b)
            | ElementwiseOp::Mul(a, b)
            | ElementwiseOp::Max(a, b)
            | ElementwiseOp::Mod(a, b) => vec![a.clone(), b.clone()],
            ElementwiseOp::Neg(a)
            | ElementwiseOp::Recip(a)
            | ElementwiseOp::Sin(a)
            | ElementwiseOp::Sqrt(a)
            | ElementwiseOp::Log2(a)
            | ElementwiseOp::Exp2(a) => vec![a.clone()],
        }
    }
}

impl eframe::App for GraphVisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Harp Computational Graph Visualizer");

            if !self.sample_graph_loaded {
                if ui.button("Load Sample Graph").clicked() {
                    let mut temp_snarl = Snarl::new();
                    std::mem::swap(&mut temp_snarl, &mut self.snarl);
                    self.load_sample_graph(&mut temp_snarl);
                    self.snarl = temp_snarl;
                }
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
