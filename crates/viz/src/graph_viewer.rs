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
    /// DOTテキストのdiffを表示するかどうか
    show_dot_diff: bool,
    /// コスト遷移グラフを表示するかどうか
    show_cost_graph: bool,
    /// ログを表示するかどうか
    show_logs: bool,
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
            show_dot_text: true,
            show_dot_diff: false,
            show_cost_graph: true,
            show_logs: true,
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
            self.traverse_and_add_node_with_layout(input_node, "", visited, depths, depth_counters);
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

    /// グラフ内のすべての動的shape変数を収集
    fn collect_shape_vars(&self, graph: &Graph) -> Vec<String> {
        use std::collections::BTreeSet;

        let mut vars = BTreeSet::new();

        fn collect_from_node(node: &GraphNode, vars: &mut BTreeSet<String>) {
            // このノードのshapeから変数を収集
            for expr in node.view.shape() {
                collect_from_expr(expr, vars);
            }

            // 再帰的に入力ノードも処理
            for src in &node.src {
                collect_from_node(src, vars);
            }
        }

        fn collect_from_expr(expr: &harp::graph::shape::Expr, vars: &mut BTreeSet<String>) {
            use harp::graph::shape::Expr;
            match expr {
                Expr::Var(name) => {
                    vars.insert(name.clone());
                }
                Expr::Add(l, r)
                | Expr::Sub(l, r)
                | Expr::Mul(l, r)
                | Expr::Div(l, r)
                | Expr::Rem(l, r) => {
                    collect_from_expr(l, vars);
                    collect_from_expr(r, vars);
                }
                Expr::Const(_) => {}
            }
        }

        // 出力ノードから開始
        for output_node in graph.outputs().values() {
            collect_from_node(output_node, &mut vars);
        }

        vars.into_iter().collect()
    }

    /// UIを描画
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // キーボード入力処理（左右矢印キー）
        if self.optimization_history.is_some() {
            ui.input(|i| {
                if i.key_pressed(egui::Key::ArrowLeft) {
                    self.prev_step();
                } else if i.key_pressed(egui::Key::ArrowRight) {
                    self.next_step();
                }
            });
        }

        ui.heading("Graph Viewer");
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
                        // 科学記数法で表示（小さい値でも読みやすく）
                        let cost_str = if snapshot.cost.abs() < 0.001 && snapshot.cost != 0.0 {
                            format!("{:.2e}", snapshot.cost)
                        } else {
                            format!("{:.6}", snapshot.cost)
                        };
                        ui.label(cost_str);
                    });
                }
            }

            ui.separator();
        }

        // コスト遷移グラフを表示（折りたたみ可能）
        if let Some(ref history) = self.optimization_history {
            egui::CollapsingHeader::new("Cost Transition")
                .default_open(true)
                .show(ui, |ui| {
                    // コストデータを収集
                    let cost_points: Vec<[f64; 2]> = (0..history.len())
                        .filter_map(|step| {
                            history
                                .get(step)
                                .map(|snapshot| [step as f64, snapshot.cost as f64])
                        })
                        .collect();

                    // プロットを表示
                    egui_plot::Plot::new("cost_plot")
                        .view_aspect(2.0)
                        .height(200.0)
                        .show(ui, |plot_ui| {
                            plot_ui.line(
                                egui_plot::Line::new(cost_points)
                                    .color(egui::Color32::from_rgb(100, 150, 250))
                                    .name("Cost"),
                            );

                            // 現在のステップを縦線で表示
                            let current_step = self.current_step as f64;
                            plot_ui.vline(
                                egui_plot::VLine::new(current_step)
                                    .color(egui::Color32::from_rgb(255, 100, 100))
                                    .name("Current Step"),
                            );
                        });
                });

            ui.separator();
        }

        // ログを表示（最適化履歴がある場合）- グラフビューの前に配置
        if let Some(ref history) = self.optimization_history {
            if let Some(snapshot) = history.get(self.current_step) {
                // 折りたたみ可能なセクションとして表示
                egui::CollapsingHeader::new(format!("Debug Logs ({} entries)", snapshot.logs.len()))
                    .default_open(false) // デフォルトで閉じた状態にして、画面を広く使う
                    .show(ui, |ui| {
                        if !snapshot.logs.is_empty() {
                            egui::ScrollArea::vertical()
                                .id_salt("graph_logs_scroll")
                                .max_height(200.0) // 高さを少し小さくして、他のコンテンツも見やすく
                                .show(ui, |ui| {
                                    for log_line in &snapshot.logs {
                                        // ログレベルに応じて色分け
                                        let color = if log_line.contains("[ERROR]") {
                                            egui::Color32::from_rgb(255, 100, 100)
                                        } else if log_line.contains("[WARN]") {
                                            egui::Color32::from_rgb(255, 200, 100)
                                        } else if log_line.contains("[DEBUG]") {
                                            egui::Color32::from_rgb(150, 150, 255)
                                        } else if log_line.contains("[TRACE]") {
                                            egui::Color32::GRAY
                                        } else {
                                            egui::Color32::WHITE
                                        };

                                        ui.colored_label(color, egui::RichText::new(log_line).monospace());
                                    }
                                });
                        } else {
                            ui.label("No logs captured for this step.");
                        }
                    });

                ui.separator();
            }
        }

        if self.harp_graph.is_none() {
            ui.label("No graph loaded.");
            ui.label("Load a graph to visualize it here.");
            return;
        }

        // グラフ情報を表示
        if let Some(ref graph) = self.harp_graph {
            // Inputs情報
            ui.horizontal(|ui| {
                ui.label("Inputs:");
                ui.label(graph.inputs().len().to_string());
            });

            // 入力ノードの詳細を折りたたみ表示
            ui.collapsing("Input Nodes", |ui| {
                // 名前順にソート
                let mut input_names: Vec<_> = graph.inputs().keys().cloned().collect();
                input_names.sort();

                for name in input_names {
                    if let Some(weak_input) = graph.inputs().get(&name) {
                        if let Some(rc_node) = weak_input.upgrade() {
                            let input_node = GraphNode::from_rc(rc_node);
                            let shape_str: Vec<String> = input_node
                                .view
                                .shape()
                                .iter()
                                .map(|e| format!("{}", e))
                                .collect();
                            ui.label(format!("• {} : [{}]", name, shape_str.join(", ")));
                        } else {
                            ui.label(format!("• {} : <dropped>", name));
                        }
                    }
                }
            });

            ui.add_space(5.0);

            // Outputs情報
            ui.horizontal(|ui| {
                ui.label("Outputs:");
                ui.label(graph.outputs().len().to_string());
            });

            // 出力ノードの詳細を折りたたみ表示
            ui.collapsing("Output Nodes", |ui| {
                // 名前順にソート
                let mut output_names: Vec<_> = graph.outputs().keys().cloned().collect();
                output_names.sort();

                for name in output_names {
                    if let Some(output_node) = graph.outputs().get(&name) {
                        let shape_str: Vec<String> = output_node
                            .view
                            .shape()
                            .iter()
                            .map(|e| format!("{}", e))
                            .collect();
                        ui.label(format!("• {} : [{}]", name, shape_str.join(", ")));
                    }
                }
            });

            ui.add_space(5.0);

            // Shape Variables情報
            let shape_vars = self.collect_shape_vars(graph);
            ui.horizontal(|ui| {
                ui.label("Shape Variables:");
                ui.label(shape_vars.len().to_string());
            });

            if !shape_vars.is_empty() {
                ui.collapsing("Shape Variables", |ui| {
                    for var in &shape_vars {
                        ui.label(format!("• {}", var));
                    }
                });
            }
        }

        ui.separator();

        // グラフビュー
        egui::CollapsingHeader::new("Graph View")
            .default_open(true)
            .show(ui, |ui| {
                self.snarl.show(
                    &mut GraphNodeViewStyle,
                    &egui_snarl::ui::SnarlStyle::default(),
                    egui::Id::new("graph_viewer_snarl"),
                    ui,
                );
            });

        ui.separator();

        // DOTテキスト（折りたたみ可能）
        egui::CollapsingHeader::new("DOT Format")
            .default_open(false)
            .show(ui, |ui| {
                if let Some(ref graph) = self.harp_graph {
                    // クリップボードにコピーボタン
                    if ui.button("📋 Copy to Clipboard").clicked() {
                        let dot_text = graph.to_dot();
                        ui.output_mut(|o| o.copied_text = dot_text);
                        log::info!("DOT text copied to clipboard");
                    }

                    ui.add_space(5.0);

                    // Diff表示（最適化履歴がある場合のみ、折りたたみ可能）
                    if self.optimization_history.is_some() && self.current_step > 0 {
                        egui::CollapsingHeader::new("Show Diff (Previous → Current)")
                            .default_open(false)
                            .show(ui, |ui| {
                                let current_dot = graph.to_dot();
                                let prev_dot = self.optimization_history.as_ref().and_then(|history| {
                                    history
                                        .get(self.current_step - 1)
                                        .map(|prev_snapshot| prev_snapshot.graph.to_dot())
                                });

                                if let Some(prev_text) = prev_dot {
                                    egui::ScrollArea::vertical()
                                        .max_height(300.0)
                                        .show(ui, |ui| {
                                            let diff = similar::TextDiff::from_lines(&prev_text, &current_dot);

                                            for change in diff.iter_all_changes() {
                                                let (color, prefix) = match change.tag() {
                                                    similar::ChangeTag::Delete => {
                                                        (egui::Color32::from_rgb(255, 200, 200), "-")
                                                    }
                                                    similar::ChangeTag::Insert => {
                                                        (egui::Color32::from_rgb(200, 255, 200), "+")
                                                    }
                                                    similar::ChangeTag::Equal => (egui::Color32::GRAY, " "),
                                                };

                                                ui.horizontal(|ui| {
                                                    ui.colored_label(color, format!("{} {}", prefix, change));
                                                });
                                            }
                                        });
                                }
                            });

                        ui.add_space(5.0);
                    }

                    // DOTテキスト本文
                    let current_dot = graph.to_dot();
                    egui::ScrollArea::vertical()
                        .max_height(400.0)
                        .show(ui, |ui| {
                            ui.add(
                                egui::TextEdit::multiline(&mut current_dot.clone())
                                    .code_editor()
                                    .desired_width(f32::INFINITY),
                            );
                        });
                } else {
                    ui.label("No graph loaded");
                }
            });
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
