//! グラフ構造を可視化するビューア

use egui_snarl::{InPin, InPinId, NodeId, OutPin, OutPinId, Snarl};
use harp::graph::{Graph, GraphNode};
use harp::opt::graph::OptimizationHistory;
use std::collections::{HashMap, HashSet};

/// グラフビューアアプリケーション
pub struct GraphViewerApp {
    /// egui-snarlのグラフ表現
    snarl: Snarl<GraphNodeView>,
    /// 読み込まれたHarpグラフ
    harp_graph: Option<Graph>,
    /// HarpのGraphNodeとSnarlのNodeIdのマッピング
    node_mapping: HashMap<*const harp::graph::GraphNodeData, NodeId>,
    /// 最適化履歴
    optimization_history: Option<OptimizationHistory>,
    /// 現在表示中のステップ
    current_step: usize,
    /// DOTテキストを表示するかどうか
    show_dot_text: bool,
}

/// egui-snarl用のノードビュー
#[derive(Clone)]
pub struct GraphNodeView {
    /// ノードの名前
    pub name: String,
    /// ノードの型
    pub op_type: String,
    /// 入力ピンの数
    pub num_inputs: usize,
    /// 出力ピンの数
    pub num_outputs: usize,
    /// 詳細情報
    pub details: NodeDetails,
}

/// ノードの詳細情報
#[derive(Clone)]
pub struct NodeDetails {
    /// データ型
    pub dtype: String,
    /// 形状
    pub shape: Vec<String>,
    /// 最適化戦略
    pub strategies: Vec<String>,
    /// 操作の詳細
    pub op_details: String,
}

impl Default for GraphViewerApp {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphViewerApp {
    /// 新しいGraphViewerAppを作成
    pub fn new() -> Self {
        Self {
            snarl: Snarl::new(),
            harp_graph: None,
            node_mapping: HashMap::new(),
            optimization_history: None,
            current_step: 0,
            show_dot_text: false,
        }
    }

    /// Harpのグラフを読み込む
    pub fn load_graph(&mut self, graph: Graph) {
        let num_outputs = graph.outputs().len();
        self.harp_graph = Some(graph);
        self.optimization_history = None;
        self.current_step = 0;

        // グラフをSnarlノードに変換
        self.convert_graph_to_snarl();

        log::info!("Graph loaded with {} outputs", num_outputs);
    }

    /// 最適化履歴を読み込む
    pub fn load_history(&mut self, history: OptimizationHistory) {
        if history.is_empty() {
            log::warn!("Attempted to load empty optimization history");
            return;
        }

        self.optimization_history = Some(history);
        self.current_step = 0;

        // 最初のステップのグラフを表示
        self.update_graph_from_step();
    }

    /// 現在のステップに基づいてグラフを更新
    fn update_graph_from_step(&mut self) {
        // 必要な情報を先に取得
        let (graph, step, description) = if let Some(ref history) = self.optimization_history {
            if let Some(snapshot) = history.get(self.current_step) {
                (
                    snapshot.graph.clone(),
                    snapshot.step,
                    snapshot.description.clone(),
                )
            } else {
                return;
            }
        } else {
            return;
        };

        self.harp_graph = Some(graph);

        // グラフをSnarlノードに変換
        self.convert_graph_to_snarl();

        log::info!("Updated to step {}: {}", step, description);
    }

    /// 次のステップに進む
    pub fn next_step(&mut self) {
        if let Some(ref history) = self.optimization_history {
            if self.current_step + 1 < history.len() {
                self.current_step += 1;
                self.update_graph_from_step();
            }
        }
    }

    /// 前のステップに戻る
    pub fn prev_step(&mut self) {
        if self.current_step > 0 {
            self.current_step -= 1;
            self.update_graph_from_step();
        }
    }

    /// 特定のステップにジャンプ
    pub fn goto_step(&mut self, step: usize) {
        if let Some(ref history) = self.optimization_history {
            if step < history.len() {
                self.current_step = step;
                self.update_graph_from_step();
            }
        }
    }

    /// GraphをSnarlノードに変換
    fn convert_graph_to_snarl(&mut self) {
        let graph = match &self.harp_graph {
            Some(g) => g.clone(),
            None => return,
        };

        self.snarl = Snarl::new();
        self.node_mapping.clear();

        // ノードの深さを事前に計算
        let depths = self.calculate_node_depths(&graph);

        // 深さごとにノードをカウント（水平位置計算用）
        let mut depth_counters: HashMap<usize, usize> = HashMap::new();

        // 訪問済みノードを追跡
        let mut visited = HashSet::new();

        // 出力ノードから開始してトラバース（位置情報付き）
        for (output_name, output_node) in graph.outputs() {
            self.traverse_and_add_node_with_layout(
                output_node,
                output_name,
                &mut visited,
                &depths,
                &mut depth_counters,
            );
        }

        // エッジを追加
        for output_node in graph.outputs().values() {
            self.add_edges(output_node, &mut HashSet::new());
        }
    }

    /// ノードの深さを計算（入力ノードからの距離）
    fn calculate_node_depths(
        &self,
        graph: &Graph,
    ) -> HashMap<*const harp::graph::GraphNodeData, usize> {
        let mut depths = HashMap::new();

        // 深さ優先探索で各ノードの深さを計算
        fn calculate_depth(
            node: &GraphNode,
            depths: &mut HashMap<*const harp::graph::GraphNodeData, usize>,
            visited: &mut HashSet<*const harp::graph::GraphNodeData>,
        ) -> usize {
            let node_ptr = node.as_ptr();

            // 既に計算済みならそれを返す
            if let Some(&depth) = depths.get(&node_ptr) {
                return depth;
            }

            // 循環参照のチェック
            if visited.contains(&node_ptr) {
                return 0;
            }
            visited.insert(node_ptr);

            // 入力ノードの深さは0
            if node.src.is_empty() {
                depths.insert(node_ptr, 0);
                return 0;
            }

            // 全ての入力ノードの深さの最大値 + 1
            let max_input_depth = node
                .src
                .iter()
                .map(|input| calculate_depth(input, depths, visited))
                .max()
                .unwrap_or(0);

            let depth = max_input_depth + 1;
            depths.insert(node_ptr, depth);
            depth
        }

        // 出力ノードから開始
        let mut visited_global = HashSet::new();
        for output_node in graph.outputs().values() {
            calculate_depth(output_node, &mut depths, &mut visited_global);
        }

        depths
    }

    /// ノードをトラバースしてSnarlに追加（階層レイアウト付き）
    fn traverse_and_add_node_with_layout(
        &mut self,
        node: &GraphNode,
        node_name: &str,
        visited: &mut HashSet<*const harp::graph::GraphNodeData>,
        depths: &HashMap<*const harp::graph::GraphNodeData, usize>,
        depth_counters: &mut HashMap<usize, usize>,
    ) {
        let node_ptr = node.as_ptr();

        // 既に訪問済みならスキップ
        if visited.contains(&node_ptr) {
            return;
        }
        visited.insert(node_ptr);

        // 入力ノードを先にトラバース
        for input_node in &node.src {
            self.traverse_and_add_node_with_layout(
                input_node,
                "",
                visited,
                depths,
                depth_counters,
            );
        }

        // このノードの深さを取得
        let depth = depths.get(&node_ptr).copied().unwrap_or(0);

        // この深さでのノード数を取得してインクリメント
        let index = depth_counters.entry(depth).or_insert(0);
        let horizontal_index = *index;
        *index += 1;

        // レイアウトパラメータ
        let horizontal_spacing = 200.0; // 深さ間の水平間隔
        let vertical_spacing = 100.0; // ノード間の垂直間隔
        let start_x = 50.0;
        let start_y = 100.0;

        // 位置を計算（横方向にレイアウト）
        let x = start_x + (depth as f32) * horizontal_spacing;
        let y = start_y + (horizontal_index as f32) * vertical_spacing;

        // このノードをSnarlに追加
        let node_view = self.create_node_view(node, node_name);
        let snarl_node_id = self.snarl.insert_node(egui::Pos2::new(x, y), node_view);
        self.node_mapping.insert(node_ptr, snarl_node_id);
    }

    /// GraphNodeからGraphNodeViewを作成
    fn create_node_view(&self, node: &GraphNode, name: &str) -> GraphNodeView {
        // 操作タイプを簡潔に表示
        let op_type = self.simplify_op_type(&node.op);
        let num_inputs = node.src.len();
        let num_outputs = 1; // Harpのノードは常に1つの出力を持つ

        // ノード名を決定（出力名がある場合はそれを使用、なければ操作タイプ）
        let display_name = if name.is_empty() {
            op_type.clone()
        } else {
            name.to_string()
        };

        // 詳細情報を収集
        let dtype = format!("{:?}", node.dtype);
        let shape: Vec<String> = node.view.shape().iter().map(|e| format!("{}", e)).collect();
        let strategies: Vec<String> = node
            .elementwise_strategies
            .iter()
            .enumerate()
            .map(|(i, s)| format!("axis {}: {:?}", i, s))
            .collect();
        let op_details = format!("{:?}", node.op);

        let details = NodeDetails {
            dtype,
            shape,
            strategies,
            op_details,
        };

        GraphNodeView {
            name: display_name,
            op_type,
            num_inputs,
            num_outputs,
            details,
        }
    }

    /// 操作タイプを簡潔な表記に変換
    fn simplify_op_type(&self, op: &harp::graph::GraphOp) -> String {
        use harp::graph::GraphOp;
        match op {
            GraphOp::Input => "Input".to_string(),
            GraphOp::Const(_) => "Const".to_string(),
            GraphOp::View(_) => "View".to_string(),
            GraphOp::Contiguous { .. } => "Contiguous".to_string(),
            GraphOp::Elementwise { op, .. } => format!("Elem({:?})", op),
            GraphOp::Reduce { op, axis, .. } => format!("Reduce({:?}, {})", op, axis),
            GraphOp::Cumulative { .. } => "Cumulative".to_string(),
            GraphOp::FusedElementwise { .. } => "FusedElem".to_string(),
            GraphOp::FusedElementwiseReduce { .. } => "FusedElemReduce".to_string(),
            GraphOp::FusedReduce { .. } => "FusedReduce".to_string(),
        }
    }

    /// エッジを追加
    fn add_edges(
        &mut self,
        node: &GraphNode,
        visited: &mut HashSet<*const harp::graph::GraphNodeData>,
    ) {
        let node_ptr = node.as_ptr();

        // 既に訪問済みならスキップ
        if visited.contains(&node_ptr) {
            return;
        }
        visited.insert(node_ptr);

        // このノードのSnarlノードIDを取得
        let to_node_id = match self.node_mapping.get(&node_ptr) {
            Some(&id) => id,
            None => return,
        };

        // 各入力ノードからこのノードへのエッジを追加
        for (input_idx, input_node) in node.src.iter().enumerate() {
            let from_node_ptr = input_node.as_ptr();
            if let Some(&from_node_id) = self.node_mapping.get(&from_node_ptr) {
                // エッジを追加（from_node_idの出力0からto_node_idの入力input_idxへ）
                let out_pin = OutPinId {
                    node: from_node_id,
                    output: 0,
                };
                let in_pin = InPinId {
                    node: to_node_id,
                    input: input_idx,
                };
                self.snarl.connect(out_pin, in_pin);
            }

            // 再帰的に入力ノードのエッジも追加
            self.add_edges(input_node, visited);
        }
    }

    /// UIを描画
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Graph Viewer");
            ui.add_space(20.0);

            // DOTテキスト表示トグルボタン
            let button_text = if self.show_dot_text {
                "Hide DOT Text"
            } else {
                "Show DOT Text"
            };
            if ui.button(button_text).clicked() {
                self.show_dot_text = !self.show_dot_text;
            }
        });
        ui.separator();

        // 最適化履歴がある場合はナビゲーションを表示
        if self.optimization_history.is_some() {
            let history_len = self.optimization_history.as_ref().unwrap().len();
            let current_step = self.current_step;

            // ナビゲーションボタン
            ui.horizontal(|ui| {
                // 前のステップボタン
                let prev_clicked = ui
                    .add_enabled(current_step > 0, egui::Button::new("◀ Prev"))
                    .clicked();

                // ステップ情報表示
                ui.label(format!("Step: {} / {}", current_step, history_len - 1));

                // 次のステップボタン
                let next_clicked = ui
                    .add_enabled(current_step + 1 < history_len, egui::Button::new("Next ▶"))
                    .clicked();

                if prev_clicked {
                    self.prev_step();
                } else if next_clicked {
                    self.next_step();
                }
            });

            // 現在のステップの説明とコストを表示
            if let Some(ref history) = self.optimization_history {
                if let Some(snapshot) = history.get(self.current_step) {
                    ui.horizontal(|ui| {
                        ui.label("Description:");
                        ui.label(&snapshot.description);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Cost:");
                        ui.label(format!("{:.2}", snapshot.cost));
                    });
                }
            }

            ui.separator();
        }

        if self.harp_graph.is_none() {
            ui.label("No graph loaded.");
            ui.label("Load a graph to visualize it here.");
            return;
        }

        // グラフ情報を表示
        if let Some(ref graph) = self.harp_graph {
            ui.horizontal(|ui| {
                ui.label("Outputs:");
                ui.label(graph.outputs().len().to_string());
            });

            // 出力ノードの詳細を折りたたみ表示
            ui.collapsing("Output Nodes", |ui| {
                for (name, _node) in graph.outputs() {
                    ui.label(format!("• {}", name));
                }
            });
        }

        ui.separator();

        // DOTテキストを表示する場合は横分割
        if self.show_dot_text {
            ui.columns(2, |columns| {
                // 左側: グラフビュー
                columns[0].vertical(|ui| {
                    ui.heading("Graph View");
                    ui.separator();
                    self.snarl.show(
                        &mut GraphNodeViewStyle,
                        &egui_snarl::ui::SnarlStyle::default(),
                        egui::Id::new("graph_viewer_snarl"),
                        ui,
                    );
                });

                // 右側: DOTテキスト
                columns[1].vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.heading("DOT Format");
                        ui.add_space(10.0);

                        // クリップボードにコピーボタン
                        if ui.button("Copy to Clipboard").clicked() {
                            if let Some(ref graph) = self.harp_graph {
                                let dot_text = graph.to_dot();
                                ui.output_mut(|o| o.copied_text = dot_text);
                                log::info!("DOT text copied to clipboard");
                            }
                        }
                    });
                    ui.separator();

                    if let Some(ref graph) = self.harp_graph {
                        let dot_text = graph.to_dot();

                        egui::ScrollArea::vertical()
                            .max_height(ui.available_height())
                            .show(ui, |ui| {
                                ui.add(
                                    egui::TextEdit::multiline(&mut dot_text.clone())
                                        .code_editor()
                                        .desired_width(f32::INFINITY)
                                );
                            });
                    } else {
                        ui.label("No graph loaded");
                    }
                });
            });
        } else {
            // 通常表示: グラフビューのみ
            self.snarl.show(
                &mut GraphNodeViewStyle,
                &egui_snarl::ui::SnarlStyle::default(),
                egui::Id::new("graph_viewer_snarl"),
                ui,
            );
        }
    }
}

/// egui-snarlのノードスタイル
struct GraphNodeViewStyle;

impl egui_snarl::ui::SnarlViewer<GraphNodeView> for GraphNodeViewStyle {
    fn title(&mut self, node: &GraphNodeView) -> String {
        node.name.clone()
    }

    fn inputs(&mut self, node: &GraphNodeView) -> usize {
        node.num_inputs
    }

    fn outputs(&mut self, node: &GraphNodeView) -> usize {
        node.num_outputs
    }

    fn show_header(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        _scale: f32,
        snarl: &mut Snarl<GraphNodeView>,
    ) {
        if let Some(node_data) = snarl.get_node(node) {
            // ノードのタイトルを表示
            ui.label(&node_data.name);

            // 詳細情報を折りたたみ表示
            ui.collapsing("Details", |ui| {
                ui.label(format!("Type: {}", node_data.op_type));
                ui.label(format!("DType: {}", node_data.details.dtype));

                if !node_data.details.shape.is_empty() {
                    ui.label(format!("Shape: [{}]", node_data.details.shape.join(", ")));
                }

                if !node_data.details.strategies.is_empty() {
                    ui.collapsing("Strategies", |ui| {
                        for strategy in &node_data.details.strategies {
                            ui.label(strategy);
                        }
                    });
                }

                ui.collapsing("Operation Details", |ui| {
                    ui.label(&node_data.details.op_details);
                });
            });
        }
    }

    fn show_input(
        &mut self,
        _pin: &InPin,
        ui: &mut egui::Ui,
        _scale: f32,
        _snarl: &mut Snarl<GraphNodeView>,
    ) -> egui_snarl::ui::PinInfo {
        ui.label("in");
        egui_snarl::ui::PinInfo::circle().with_fill(egui::Color32::from_rgb(100, 150, 200))
    }

    fn show_output(
        &mut self,
        _pin: &OutPin,
        ui: &mut egui::Ui,
        _scale: f32,
        _snarl: &mut Snarl<GraphNodeView>,
    ) -> egui_snarl::ui::PinInfo {
        ui.label("out");
        egui_snarl::ui::PinInfo::circle().with_fill(egui::Color32::from_rgb(200, 150, 100))
    }

    fn connect(&mut self, _from: &OutPin, _to: &InPin, _snarl: &mut Snarl<GraphNodeView>) {
        // 接続は許可しない（読み取り専用）
    }

    fn disconnect(&mut self, _from: &OutPin, _to: &InPin, _snarl: &mut Snarl<GraphNodeView>) {
        // 切断は許可しない（読み取り専用）
    }
}
