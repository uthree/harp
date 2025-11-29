//! グラフ構造を可視化するビューア

use crate::renderer_selector::{render_with_type, renderer_selector_ui, RendererType};
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
    /// 選択中のノードID（詳細パネル表示用）
    selected_node: Option<NodeId>,
    /// サイドパネルの幅
    side_panel_width: f32,
    /// サイドパネルを表示するか
    show_side_panel: bool,
    /// 現在のレンダラータイプ
    renderer_type: RendererType,
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

    /// 操作の詳細
    pub op_details: String,
    /// AST（Customノードの場合、レンダラー変更時に再レンダリング用）
    pub ast: Option<harp::ast::AstNode>,
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
            selected_node: None,
            side_panel_width: 450.0,
            show_side_panel: true,
            renderer_type: RendererType::default(),
        }
    }

    /// レンダラータイプを設定
    pub fn set_renderer_type(&mut self, renderer_type: RendererType) {
        self.renderer_type = renderer_type;
    }

    /// 現在のレンダラータイプを取得
    pub fn renderer_type(&self) -> RendererType {
        self.renderer_type
    }

    /// Harpのグラフを読み込む
    pub fn load_graph(&mut self, graph: Graph) {
        let num_outputs = graph.outputs().len();
        self.harp_graph = Some(graph);
        self.optimization_history = None;
        self.current_step = 0;
        self.selected_node = None;

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
        self.selected_node = None;

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

    /// ノードが出力 Buffer かどうかを判定
    fn is_output_buffer(node: &GraphNode) -> bool {
        use harp::graph::GraphOp;
        matches!(&node.op, GraphOp::Buffer { name } if name.starts_with("output"))
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

        // 出力 Buffer ノードはスキップ（表示しない）
        if Self::is_output_buffer(node) {
            return;
        }

        // 既に訪問済みならスキップ
        if visited.contains(&node_ptr) {
            return;
        }
        visited.insert(node_ptr);

        // 入力ノードを先にトラバース（出力 Buffer を除外）
        for input_node in &node.src {
            if !Self::is_output_buffer(input_node) {
                self.traverse_and_add_node_with_layout(
                    input_node,
                    "",
                    visited,
                    depths,
                    depth_counters,
                );
            }
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

        let op_details = format!("{:?}", node.op);

        // Customノードの場合はASTを保存（レンダラー変更時に再レンダリング用）
        let ast = if let harp::graph::GraphOp::Custom { ast } = &node.op {
            Some(ast.clone())
        } else {
            None
        };

        let details = NodeDetails {
            dtype,
            shape,

            op_details,
            ast,
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
            GraphOp::Buffer { name } => name.clone(),
            GraphOp::Const(l) => format!("{:?}", l),
            GraphOp::View(_) => "View".to_string(),
            GraphOp::Contiguous => "Contiguous".to_string(),
            GraphOp::Elementwise { op, .. } => format!("{:?}", op),
            GraphOp::Reduce { op, axis, .. } => format!("{:?}Reduce({})", op, axis),
            GraphOp::Cumulative { .. } => "Cum".to_string(),
            GraphOp::FusedElementwise { .. } => "FusedE".to_string(),
            GraphOp::FusedElementwiseReduce { .. } => "FusedER".to_string(),
            GraphOp::FusedElementwiseCumulative { .. } => "FusedEC".to_string(),
            GraphOp::FusedReduce { .. } => "FusedR".to_string(),
            GraphOp::Custom { .. } => "Custom".to_string(),
            GraphOp::Pad { .. } => "Pad".to_string(),
            GraphOp::Slice { .. } => "Slice".to_string(),
            GraphOp::Fold { .. } => "Fold".to_string(),
            GraphOp::Rand => "Rand".to_string(),
            GraphOp::Concat { axis } => format!("Concat({})", axis),
            _ => "Unknown".to_string(),
        }
    }

    /// エッジを追加
    fn add_edges(
        &mut self,
        node: &GraphNode,
        visited: &mut HashSet<*const harp::graph::GraphNodeData>,
    ) {
        let node_ptr = node.as_ptr();

        // 出力 Buffer ノードはスキップ
        if Self::is_output_buffer(node) {
            return;
        }

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

        // 各入力ノードからこのノードへのエッジを追加（出力 Buffer を除外）
        // 出力 Buffer は node_mapping に登録されていないため実際にはスキップされるが、
        // 入力インデックスが正しくなるように明示的にフィルタリング
        let input_nodes: Vec<_> = node
            .src
            .iter()
            .filter(|n| !Self::is_output_buffer(n))
            .collect();
        for (input_idx, input_node) in input_nodes.iter().enumerate() {
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

    /// 全ノードのIDと名前のリストを取得
    fn get_node_list(&self) -> Vec<(NodeId, String, bool)> {
        let mut nodes = Vec::new();
        for (node_id, node_data) in self.snarl.node_ids() {
            let has_code = node_data.details.ast.is_some();
            nodes.push((node_id, node_data.name.clone(), has_code));
        }
        // 名前でソート（Customノードを先に）
        nodes.sort_by(|a, b| match (a.2, b.2) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.1.cmp(&b.1),
        });
        nodes
    }

    /// サイドパネルの内容を表示（ノード詳細）
    fn show_side_panel_content(&mut self, ui: &mut egui::Ui) {
        ui.heading("📝 Node Details");
        ui.separator();

        // ノード選択ドロップダウン
        let node_list = self.get_node_list();
        let custom_nodes: Vec<_> = node_list
            .iter()
            .filter(|(_, _, has_code)| *has_code)
            .collect();

        if custom_nodes.is_empty() {
            ui.label("No Custom nodes in the current graph.");
            return;
        }

        // 現在選択中のノード名を取得
        let current_name = self
            .selected_node
            .and_then(|id| self.snarl.get_node(id))
            .map(|n| n.name.clone())
            .unwrap_or_else(|| "Select a node...".to_string());

        ui.horizontal(|ui| {
            ui.label("Node:");
            egui::ComboBox::from_id_salt("node_selector")
                .selected_text(&current_name)
                .width(ui.available_width() - 60.0)
                .show_ui(ui, |ui| {
                    for (node_id, name, has_code) in &node_list {
                        let label = if *has_code {
                            format!("🔧 {}", name)
                        } else {
                            name.clone()
                        };
                        if ui
                            .selectable_value(&mut self.selected_node, Some(*node_id), label)
                            .clicked()
                        {
                            log::debug!("Selected node: {}", name);
                        }
                    }
                });

            // クリアボタン
            if self.selected_node.is_some() && ui.button("✕").clicked() {
                self.selected_node = None;
            }
        });

        ui.separator();

        // 選択されたノードの詳細を表示
        if let Some(node_id) = self.selected_node {
            if let Some(node_data) = self.snarl.get_node(node_id) {
                // ノード基本情報
                egui::Grid::new("node_info_grid")
                    .num_columns(2)
                    .spacing([10.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Name:");
                        ui.strong(&node_data.name);
                        ui.end_row();

                        ui.label("Type:");
                        ui.code(&node_data.op_type);
                        ui.end_row();

                        ui.label("DType:");
                        ui.label(&node_data.details.dtype);
                        ui.end_row();

                        if !node_data.details.shape.is_empty() {
                            ui.label("Shape:");
                            ui.label(format!("[{}]", node_data.details.shape.join(", ")));
                            ui.end_row();
                        }
                    });

                // コードがある場合は全画面で表示
                if let Some(ref ast) = node_data.details.ast {
                    // ASTをレンダリング
                    let code = render_with_type(ast, self.renderer_type);

                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.strong("Generated Code");
                        ui.label(format!("({} lines)", code.lines().count()));

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("📋 Copy").clicked() {
                                ui.output_mut(|o| o.copied_text = code.clone());
                                log::info!("Code copied to clipboard");
                            }
                        });
                    });

                    // レンダラー選択
                    ui.horizontal(|ui| {
                        renderer_selector_ui(ui, &mut self.renderer_type);
                    });

                    ui.separator();

                    // コード表示エリア（残りの高さを全て使用）
                    egui::ScrollArea::both()
                        .id_salt("node_code_scroll")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            let theme = egui_extras::syntax_highlighting::CodeTheme::from_memory(
                                ui.ctx(),
                                ui.style(),
                            );

                            let highlighted_code = egui_extras::syntax_highlighting::highlight(
                                ui.ctx(),
                                ui.style(),
                                &theme,
                                &code,
                                "c",
                            );

                            ui.add(egui::Label::new(highlighted_code).selectable(true));
                        });
                } else {
                    ui.add_space(10.0);
                    ui.label("This node does not have generated code.");
                }
            } else {
                ui.label("Selected node not found.");
                self.selected_node = None;
            }
        } else {
            ui.add_space(10.0);
            ui.label("Select a Custom node to view its details and generated code.");
            ui.add_space(10.0);

            // Customノードのクイック選択ボタン
            ui.label("Quick select:");
            for (node_id, name, has_code) in &custom_nodes {
                if *has_code && ui.button(format!("🔧 {}", name)).clicked() {
                    self.selected_node = Some(*node_id);
                }
            }
        }
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

        // ヘッダーとサイドパネルトグルボタン
        ui.horizontal(|ui| {
            ui.heading("Graph Viewer");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let toggle_text = if self.show_side_panel {
                    "Hide Details ▶"
                } else {
                    "◀ Show Details"
                };
                if ui.button(toggle_text).clicked() {
                    self.show_side_panel = !self.show_side_panel;
                }
            });
        });
        ui.separator();

        // サイドパネルを先に表示（右側）
        if self.show_side_panel {
            egui::SidePanel::right("node_details_panel")
                .default_width(self.side_panel_width)
                .min_width(300.0)
                .max_width(800.0)
                .resizable(true)
                .show_inside(ui, |ui| {
                    self.show_side_panel_content(ui);
                });
        }

        // メインコンテンツ（残りのスペース）
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
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
                            .add_enabled(
                                current_step + 1 < history_len,
                                egui::Button::new("Next ▶"),
                            )
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
                            // Suggester名を表示
                            if let Some(ref suggester_name) = snapshot.suggester_name {
                                ui.horizontal(|ui| {
                                    ui.label("Suggester:");
                                    ui.label(
                                        egui::RichText::new(suggester_name)
                                            .color(egui::Color32::from_rgb(100, 200, 150))
                                            .strong(),
                                    );
                                });
                            }

                            ui.horizontal(|ui| {
                                ui.label("Description:");
                                ui.label(&snapshot.description);
                            });
                            ui.horizontal(|ui| {
                                ui.label("Cost:");
                                // コストは対数スケール（log(CPUサイクル数)）で表示
                                let cost_str = format!("{:.2}", snapshot.cost);
                                ui.label(cost_str);

                                ui.separator();

                                ui.label("Candidates:");
                                if let Some(num_candidates) = snapshot.num_candidates {
                                    ui.label(format!("{}", num_candidates));
                                } else {
                                    ui.label("-");
                                }
                            });
                        }
                    }

                    ui.separator();
                }

                // コスト遷移グラフを表示（折りたたみ可能）
                if let Some(ref history) = self.optimization_history {
                    egui::CollapsingHeader::new("Cost & Candidates Transition")
                        .default_open(true)
                        .show(ui, |ui| {
                            egui::Resize::default()
                                .default_height(200.0)
                                .min_height(100.0)
                                .max_height(600.0)
                                .resizable(true)
                                .show(ui, |ui| {
                                    // コストデータを収集
                                    let cost_points: Vec<[f64; 2]> = history
                                        .cost_transition()
                                        .iter()
                                        .map(|(step, cost)| [*step as f64, *cost as f64])
                                        .collect();

                                    // 候補数データを収集
                                    let candidate_points: Vec<[f64; 2]> = history
                                        .candidate_transition()
                                        .iter()
                                        .map(|(step, count)| [*step as f64, *count as f64])
                                        .collect();

                                    // 候補数の最大値を取得してスケーリング係数を計算
                                    let max_candidates = candidate_points
                                        .iter()
                                        .map(|p| p[1])
                                        .fold(0.0_f64, |a, b| a.max(b));
                                    let max_cost = cost_points
                                        .iter()
                                        .map(|p| p[1])
                                        .fold(0.0_f64, |a, b| a.max(b));

                                    // 候補数をコストスケールに正規化（同じグラフに表示するため）
                                    let scale = if max_candidates > 0.0 && max_cost > 0.0 {
                                        max_cost / max_candidates
                                    } else {
                                        1.0
                                    };
                                    let scaled_candidate_points: Vec<[f64; 2]> = candidate_points
                                        .iter()
                                        .map(|p| [p[0], p[1] * scale])
                                        .collect();

                                    // プロットを表示
                                    egui_plot::Plot::new("cost_plot")
                                        .view_aspect(2.0)
                                        .height(ui.available_height())
                                        .legend(egui_plot::Legend::default())
                                        .show(ui, |plot_ui| {
                                            // コストライン（青）
                                            plot_ui.line(
                                                egui_plot::Line::new(cost_points)
                                                    .color(egui::Color32::from_rgb(100, 150, 250))
                                                    .name("Cost"),
                                            );

                                            // 候補数ライン（緑、スケール済み）
                                            if !scaled_candidate_points.is_empty() {
                                                plot_ui.line(
                                                    egui_plot::Line::new(scaled_candidate_points)
                                                        .color(egui::Color32::from_rgb(
                                                            100, 200, 150,
                                                        ))
                                                        .name(format!(
                                                            "Candidates (×{:.1})",
                                                            scale
                                                        )),
                                                );
                                            }

                                            // 現在のステップを縦線で表示
                                            let current_step = self.current_step as f64;
                                            plot_ui.vline(
                                                egui_plot::VLine::new(current_step)
                                                    .color(egui::Color32::from_rgb(255, 100, 100))
                                                    .name("Current Step"),
                                            );
                                        });
                                });
                        });

                    ui.separator();
                }

                // ログを表示（最適化履歴がある場合）- グラフビューの前に配置
                if let Some(ref history) = self.optimization_history {
                    if let Some(snapshot) = history.get(self.current_step) {
                        // 折りたたみ可能なセクションとして表示
                        egui::CollapsingHeader::new(format!(
                            "Debug Logs ({} entries)",
                            snapshot.logs.len()
                        ))
                        .default_open(false) // デフォルトで閉じた状態にして、画面を広く使う
                        .show(ui, |ui| {
                            egui::Resize::default()
                                .default_height(200.0)
                                .min_height(100.0)
                                .max_height(800.0)
                                .resizable(true)
                                .show(ui, |ui| {
                                    if !snapshot.logs.is_empty() {
                                        egui::ScrollArea::both() // 長いログ行にも対応
                                    .id_salt("graph_logs_scroll")
                                    .max_height(ui.available_height())
                                    .auto_shrink([false, false])
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

                                            ui.colored_label(
                                                color,
                                                egui::RichText::new(log_line).monospace(),
                                            );
                                        }
                                    });
                                    } else {
                                        ui.label("No logs captured for this step.");
                                    }
                                });
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
                        // クリック検出用の一時変数
                        let mut clicked_node: Option<NodeId> = None;
                        let selected = self.selected_node;

                        self.snarl.show(
                            &mut GraphNodeViewStyle::new(&mut clicked_node, selected),
                            &egui_snarl::ui::SnarlStyle::default(),
                            egui::Id::new("graph_viewer_snarl"),
                            ui,
                        );

                        // クリックされたノードがあれば選択を更新
                        if let Some(node_id) = clicked_node {
                            self.selected_node = Some(node_id);
                            // サイドパネルを自動的に開く
                            if !self.show_side_panel {
                                self.show_side_panel = true;
                            }
                            log::debug!("Node clicked: {:?}", node_id);
                        }
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
                                let current_dot = graph.to_dot();
                                let prev_dot =
                                    self.optimization_history.as_ref().and_then(|history| {
                                        history
                                            .get(self.current_step - 1)
                                            .map(|prev_snapshot| prev_snapshot.graph.to_dot())
                                    });

                                if let Some(prev_text) = prev_dot {
                                    crate::diff_viewer::show_collapsible_diff(
                                        ui,
                                        &prev_text,
                                        &current_dot,
                                        "Show Diff (Previous -> Current)",
                                        "graph_dot_diff",
                                        true, // デフォルトで開く
                                        None,
                                    );
                                }

                                ui.add_space(5.0);
                            }

                            // DOTテキスト本文
                            let current_dot = graph.to_dot();
                            egui::ScrollArea::both() // 縦横両方にスクロール可能
                                .max_width(ui.available_width())
                                .max_height(ui.available_height())
                                .auto_shrink([false, false]) // 自動縮小を無効化して全幅を使う
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
            }); // ScrollArea::vertical() を閉じる
    }
}

/// egui-snarlのノードスタイル（クリックイベントを検出）
struct GraphNodeViewStyle<'a> {
    /// クリックされたノードID（クリック時に設定される）
    clicked_node: &'a mut Option<NodeId>,
    /// 現在選択されているノードID（ハイライト表示用）
    selected_node: Option<NodeId>,
}

impl<'a> GraphNodeViewStyle<'a> {
    fn new(clicked_node: &'a mut Option<NodeId>, selected_node: Option<NodeId>) -> Self {
        Self {
            clicked_node,
            selected_node,
        }
    }
}

impl egui_snarl::ui::SnarlViewer<GraphNodeView> for GraphNodeViewStyle<'_> {
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
            let is_selected = self.selected_node == Some(node);
            let has_code = node_data.details.ast.is_some();

            // ノードヘッダー
            ui.horizontal(|ui| {
                // 選択状態やCustomノードを示すインジケータ
                if is_selected {
                    ui.label("▶");
                } else if has_code {
                    ui.label("🔧");
                }

                // ノード名（選択時は強調）
                if is_selected {
                    ui.strong(&node_data.name);
                } else {
                    ui.label(&node_data.name);
                }

                // 選択ボタン（Customノードの場合のみ表示）
                if has_code {
                    let btn = ui.small_button("📝").on_hover_text("View code in sidebar");
                    if btn.clicked() {
                        *self.clicked_node = Some(node);
                    }
                }
            });

            // 詳細情報はサイドパネルに表示するため、ここでは省略
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
