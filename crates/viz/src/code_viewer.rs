//! 最終的に生成されたコードを表示するビューア
//!
//! グラフ最適化の結果として生成されたKernel(Program)ノードのコードを表示します。
//! AST最適化の各ステップを可視化する機能も提供します。

use crate::renderer_selector::{render_with_type, renderer_selector_ui, RendererType};
use harp::graph::{Graph, GraphNode, GraphOp};
use harp::opt::ast::OptimizationHistory as AstOptimizationHistory;
use harp::opt::graph::OptimizationHistory;
use std::collections::HashSet;

/// コードビューアアプリケーション
///
/// 最適化後のグラフから生成されたコードを表示します。
/// AST最適化の各ステップを可視化することもできます。
pub struct CodeViewerApp {
    /// グラフ最適化履歴
    optimization_history: Option<OptimizationHistory>,
    /// AST最適化履歴
    ast_history: Option<AstOptimizationHistory>,
    /// AST最適化の現在のステップ
    ast_current_step: usize,
    /// AST最適化ビューモード（trueの場合はAST履歴を表示）
    show_ast_history: bool,
    /// 現在のレンダラータイプ
    renderer_type: RendererType,
    /// キャッシュされた最終コード
    cached_code: Option<String>,
    /// キャッシュされたAST（レンダラー変更時に再レンダリング用）
    cached_ast: Option<harp::ast::AstNode>,
    /// 現在のステップのレンダリング済みコード（AST履歴表示用）
    current_step_code: Option<String>,
    /// 表示中の候補インデックス（0=選択された候補、1+=代替候補）
    viewed_candidate_index: usize,
    /// サイドパネルを表示するか
    show_side_panel: bool,
    /// サイドパネルの幅
    side_panel_width: f32,
}

impl Default for CodeViewerApp {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeViewerApp {
    /// 新しいCodeViewerAppを作成（デフォルトでCRendererを使用）
    pub fn new() -> Self {
        Self::with_renderer_type(RendererType::default())
    }

    /// 指定されたレンダラータイプでCodeViewerAppを作成
    pub fn with_renderer_type(renderer_type: RendererType) -> Self {
        Self {
            optimization_history: None,
            ast_history: None,
            ast_current_step: 0,
            show_ast_history: false,
            renderer_type,
            cached_code: None,
            cached_ast: None,
            current_step_code: None,
            viewed_candidate_index: 0,
            show_side_panel: true,
            side_panel_width: 350.0,
        }
    }

    /// レンダラータイプを設定
    pub fn set_renderer_type(&mut self, renderer_type: RendererType) {
        if self.renderer_type != renderer_type {
            self.renderer_type = renderer_type;
            // キャッシュされたASTがあれば再レンダリング
            if let Some(ref ast) = self.cached_ast {
                self.cached_code = Some(render_with_type(ast, renderer_type));
            }
        }
    }

    /// 現在のレンダラータイプを取得
    pub fn renderer_type(&self) -> RendererType {
        self.renderer_type
    }

    /// 最適化履歴を読み込む
    pub fn load_history(&mut self, history: OptimizationHistory) {
        if history.is_empty() {
            log::warn!("Attempted to load empty optimization history");
            return;
        }

        // 最終ステップのグラフからASTを抽出
        if let Some(last_snapshot) = history.snapshots().last() {
            self.cached_ast = self.extract_ast_from_graph(&last_snapshot.graph);
            // キャッシュされたASTからコードを生成
            if let Some(ref ast) = self.cached_ast {
                self.cached_code = Some(render_with_type(ast, self.renderer_type));
            } else {
                self.cached_code = None;
            }
        }

        self.optimization_history = Some(history);

        log::info!("Optimization history loaded for code viewer");
    }

    /// グラフを直接読み込む
    pub fn load_graph(&mut self, graph: Graph) {
        self.cached_ast = self.extract_ast_from_graph(&graph);
        // キャッシュされたASTからコードを生成
        if let Some(ref ast) = self.cached_ast {
            self.cached_code = Some(render_with_type(ast, self.renderer_type));
        } else {
            self.cached_code = None;
        }
        self.optimization_history = None;

        log::info!("Graph loaded for code viewer");
    }

    /// 最適化済みASTを直接読み込む
    ///
    /// グラフ履歴からの抽出をバイパスして、AST最適化後のProgramを直接設定します。
    /// これにより、AST最適化の結果がCode Viewerに正しく反映されます。
    pub fn load_optimized_ast(&mut self, ast: harp::ast::AstNode) {
        self.cached_ast = Some(ast.clone());
        self.cached_code = Some(render_with_type(&ast, self.renderer_type));
        log::info!("Optimized AST loaded for code viewer");
    }

    /// AST最適化履歴を読み込む
    ///
    /// AST最適化の各ステップを可視化するために使用します。
    pub fn load_ast_history(&mut self, history: AstOptimizationHistory) {
        if history.is_empty() {
            log::warn!("Attempted to load empty AST optimization history");
            return;
        }

        self.ast_history = Some(history);
        self.ast_current_step = 0;
        self.show_ast_history = true;
        self.viewed_candidate_index = 0;

        // 最初のステップのコードを生成
        self.update_ast_step_code();

        log::info!("AST optimization history loaded for code viewer");
    }

    /// AST最適化履歴の表示モードを切り替え
    pub fn toggle_ast_history_view(&mut self) {
        self.show_ast_history = !self.show_ast_history;
    }

    /// AST最適化の次のステップに進む
    pub fn next_ast_step(&mut self) {
        if let Some(ref history) = self.ast_history {
            if self.ast_current_step + 1 < history.len() {
                self.ast_current_step += 1;
                self.viewed_candidate_index = 0; // ステップ変更時はリセット
                self.update_ast_step_code();
            }
        }
    }

    /// AST最適化の前のステップに戻る
    pub fn prev_ast_step(&mut self) {
        if self.ast_current_step > 0 {
            self.ast_current_step -= 1;
            self.viewed_candidate_index = 0; // ステップ変更時はリセット
            self.update_ast_step_code();
        }
    }

    /// AST最適化の特定のステップにジャンプ
    pub fn goto_ast_step(&mut self, step: usize) {
        if let Some(ref history) = self.ast_history {
            if step < history.len() {
                self.ast_current_step = step;
                self.viewed_candidate_index = 0; // ステップ変更時はリセット
                self.update_ast_step_code();
            }
        }
    }

    /// 現在のステップの候補総数を取得（選択された候補 + 代替候補）
    fn get_candidate_count(&self) -> usize {
        if let Some(ref history) = self.ast_history {
            if let Some(snapshot) = history.get(self.ast_current_step) {
                return 1 + snapshot.alternatives.len(); // 選択された候補 + 代替候補
            }
        }
        1
    }

    /// 次の候補に切り替え
    pub fn next_candidate(&mut self) {
        let count = self.get_candidate_count();
        if self.viewed_candidate_index + 1 < count {
            self.viewed_candidate_index += 1;
            self.update_ast_step_code();
        }
    }

    /// 前の候補に切り替え
    pub fn prev_candidate(&mut self) {
        if self.viewed_candidate_index > 0 {
            self.viewed_candidate_index -= 1;
            self.update_ast_step_code();
        }
    }

    /// 現在のAST最適化ステップのコードを更新
    fn update_ast_step_code(&mut self) {
        if let Some(ref history) = self.ast_history {
            if let Some(snapshot) = history.get(self.ast_current_step) {
                let ast = if self.viewed_candidate_index == 0 {
                    // 選択された候補
                    &snapshot.ast
                } else {
                    // 代替候補
                    let alt_idx = self.viewed_candidate_index - 1;
                    if let Some(alt) = snapshot.alternatives.get(alt_idx) {
                        &alt.ast
                    } else {
                        &snapshot.ast
                    }
                };
                self.current_step_code = Some(render_with_type(ast, self.renderer_type));
            }
        }
    }

    /// グラフからKernel(Program)またはKernel(Function)のASTを抽出
    fn extract_ast_from_graph(&self, graph: &Graph) -> Option<harp::ast::AstNode> {
        // Kernel(Program/Function)を走査してASTを収集
        let mut visited = HashSet::new();
        let mut program_ast = None;
        let mut function_asts = Vec::new();

        // outputsから走査
        for output in graph.outputs().values() {
            Self::collect_custom_nodes(output, &mut visited, &mut program_ast, &mut function_asts);
        }

        // Kernel(Program)があればそれを優先
        if let Some(ast) = program_ast {
            return Some(ast);
        }

        // Kernel(Function)が複数ある場合は全て連結してProgram化
        if !function_asts.is_empty() {
            // 最初のASTを返す（複数ある場合は後で処理）
            // TODO: 複数のFunctionをProgramにまとめる
            return Some(function_asts.remove(0));
        }

        None
    }

    /// Kernelノードを再帰的に収集
    fn collect_custom_nodes(
        node: &GraphNode,
        visited: &mut HashSet<*const harp::graph::GraphNodeData>,
        program_ast: &mut Option<harp::ast::AstNode>,
        function_asts: &mut Vec<harp::ast::AstNode>,
    ) {
        let ptr = node.as_ptr();
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        // 入力ノードを先に処理
        for src in &node.src {
            Self::collect_custom_nodes(src, visited, program_ast, function_asts);
        }

        // Kernelノードの場合はASTを収集
        if let GraphOp::Kernel { ast, .. } = &node.op {
            // Programかどうかを判定
            if Self::is_program(ast) {
                *program_ast = Some(ast.clone());
            } else {
                function_asts.push(ast.clone());
            }
        }
    }

    /// ASTがProgramかどうかを判定
    fn is_program(ast: &harp::ast::AstNode) -> bool {
        use harp::ast::AstNode;
        match ast {
            AstNode::Program { .. } => true,
            AstNode::Block { statements, .. } => {
                // Block内にKernelが複数あるか確認
                statements
                    .iter()
                    .filter(|node| matches!(node, AstNode::Kernel { .. }))
                    .count()
                    > 1
            }
            _ => false,
        }
    }

    /// UIを描画
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // AST最適化履歴があるか確認
        let has_ast_history = self.ast_history.is_some();

        // キーボード入力処理（左右=ステップ、上下=候補）- AST履歴表示中のみ
        if has_ast_history && self.show_ast_history {
            ui.input(|i| {
                if i.key_pressed(egui::Key::ArrowLeft) {
                    self.prev_ast_step();
                } else if i.key_pressed(egui::Key::ArrowRight) {
                    self.next_ast_step();
                } else if i.key_pressed(egui::Key::ArrowUp) {
                    self.prev_candidate();
                } else if i.key_pressed(egui::Key::ArrowDown) {
                    self.next_candidate();
                }
            });
        }

        // ヘッダーとサイドパネルトグルボタン
        ui.horizontal(|ui| {
            ui.heading("Code Viewer");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // AST履歴表示中のみサイドパネルトグルを表示
                if has_ast_history && self.show_ast_history {
                    let toggle_text = if self.show_side_panel {
                        "Hide Details ▶"
                    } else {
                        "◀ Show Details"
                    };
                    if ui.button(toggle_text).clicked() {
                        self.show_side_panel = !self.show_side_panel;
                    }
                }
            });
        });
        ui.separator();

        // レンダラー選択とビュー切り替え
        ui.horizontal(|ui| {
            // レンダラー選択
            if renderer_selector_ui(ui, &mut self.renderer_type) {
                // レンダラーが変更されたら再レンダリング
                if let Some(ref ast) = self.cached_ast {
                    self.cached_code = Some(render_with_type(ast, self.renderer_type));
                }
                // AST履歴のコードも再レンダリング
                self.update_ast_step_code();
            }

            ui.separator();

            // AST最適化履歴がある場合はビュー切り替えボタンを表示
            if has_ast_history {
                let toggle_text = if self.show_ast_history {
                    "Show Final Code"
                } else {
                    "Show AST History"
                };
                if ui.button(toggle_text).clicked() {
                    self.toggle_ast_history_view();
                }
                ui.separator();
            }

            if let Some(ref history) = self.optimization_history {
                ui.label("Graph Steps:");
                ui.label(format!("{}", history.len()));

                ui.separator();

                // 最終コストを表示
                if let Some(last) = history.snapshots().last() {
                    ui.label("Final Cost:");
                    ui.label(format!("{}", last.cost));
                }
            }
        });
        ui.separator();

        // AST最適化履歴表示モード
        if has_ast_history && self.show_ast_history {
            // サイドパネルを先に表示（右側）
            if self.show_side_panel {
                egui::SidePanel::right("ast_details_panel")
                    .default_width(self.side_panel_width)
                    .min_width(250.0)
                    .max_width(600.0)
                    .resizable(true)
                    .show_inside(ui, |ui| {
                        self.show_side_panel_content(ui);
                    });
            }
            self.ui_ast_history(ui);
        } else {
            // 通常モード（最終コード表示）
            self.ui_final_code(ui);
        }
    }

    /// サイドパネルの内容を表示（候補セレクタ）
    fn show_side_panel_content(&mut self, ui: &mut egui::Ui) {
        // 候補セレクタを表示（代替候補がある場合のみ）
        // 先に必要なデータを抽出してborrow conflictを回避
        let candidate_data = self.ast_history.as_ref().and_then(|history| {
            history.get(self.ast_current_step).and_then(|snapshot| {
                if snapshot.alternatives.is_empty() {
                    None
                } else {
                    Some((
                        snapshot.cost,
                        snapshot.suggester_name.clone(),
                        snapshot
                            .alternatives
                            .iter()
                            .map(|alt| {
                                (
                                    alt.rank,
                                    alt.cost,
                                    alt.suggester_name.clone(),
                                    alt.description.clone(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ))
                }
            })
        });

        let mut new_candidate_index: Option<usize> = None;

        if let Some((snapshot_cost, suggester_name, alternatives)) = candidate_data {
            ui.heading("🔀 Candidate Selector");
            ui.separator();

            let candidate_count = 1 + alternatives.len();
            let viewed_idx = self.viewed_candidate_index;

            ui.horizontal(|ui| {
                // 前の候補
                if ui
                    .add_enabled(viewed_idx > 0, egui::Button::new("▲"))
                    .clicked()
                {
                    new_candidate_index = Some(viewed_idx.saturating_sub(1));
                }

                ui.label(format!("Candidate {}/{}", viewed_idx + 1, candidate_count));

                // 次の候補
                if ui
                    .add_enabled(viewed_idx + 1 < candidate_count, egui::Button::new("▼"))
                    .clicked()
                {
                    new_candidate_index = Some(viewed_idx + 1);
                }
            });

            // 現在の候補情報を表示
            if viewed_idx == 0 {
                // 選択された候補
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("★ Selected")
                            .color(egui::Color32::from_rgb(100, 200, 100))
                            .strong(),
                    );
                    ui.label(format!("cost={:.2}", snapshot_cost));
                });
                if let Some(ref name) = suggester_name {
                    ui.label(
                        egui::RichText::new(name).color(egui::Color32::from_rgb(100, 200, 150)),
                    );
                }
            } else {
                // 代替候補
                let alt_idx = viewed_idx - 1;
                if let Some((rank, cost, ref name, ref desc)) = alternatives.get(alt_idx) {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!("Rank {}", rank))
                                .color(egui::Color32::from_rgb(200, 150, 100)),
                        );
                        ui.label(format!("cost={:.2}", cost));
                    });
                    if let Some(ref name) = name {
                        ui.label(
                            egui::RichText::new(name).color(egui::Color32::from_rgb(150, 150, 200)),
                        );
                    }
                    if !desc.is_empty() {
                        ui.label(
                            egui::RichText::new(desc)
                                .color(egui::Color32::from_rgb(180, 180, 180))
                                .italics(),
                        );
                    }
                }
            }

            // 全候補のリスト（スクロール可能）
            ui.separator();
            egui::ScrollArea::vertical()
                .max_height(200.0)
                .show(ui, |ui| {
                    // 選択された候補（rank 0）
                    let is_current = viewed_idx == 0;
                    let text = format!("★ Selected: cost={:.2}", snapshot_cost);
                    let btn = if is_current {
                        egui::Button::new(egui::RichText::new(&text).color(egui::Color32::YELLOW))
                    } else {
                        egui::Button::new(&text)
                    };
                    if ui.add(btn).clicked() {
                        new_candidate_index = Some(0);
                    }

                    // 代替候補
                    for (idx, (rank, cost, ref name, ref desc)) in alternatives.iter().enumerate() {
                        let is_current = viewed_idx == idx + 1;
                        let name_str = name.as_deref().unwrap_or("unknown");
                        let text = if desc.is_empty() {
                            format!("Rank {}: cost={:.2} [{}]", rank, cost, name_str)
                        } else {
                            format!("Rank {}: cost={:.2} [{}] - {}", rank, cost, name_str, desc)
                        };
                        let btn = if is_current {
                            egui::Button::new(
                                egui::RichText::new(&text).color(egui::Color32::YELLOW),
                            )
                        } else {
                            egui::Button::new(&text)
                        };
                        if ui.add(btn).clicked() {
                            new_candidate_index = Some(idx + 1);
                        }
                    }
                });

            ui.separator();
        }

        // クロージャの外で候補インデックスを更新
        if let Some(idx) = new_candidate_index {
            self.viewed_candidate_index = idx;
            self.update_ast_step_code();
        }

        // ステップ情報を表示
        // 先に必要なデータを抽出
        let step_info = self.ast_history.as_ref().and_then(|history| {
            let history_len = history.len();
            history.get(self.ast_current_step).map(|snapshot| {
                (
                    self.ast_current_step,
                    history_len,
                    snapshot.cost,
                    snapshot.applied_rule.clone(),
                    snapshot.suggester_name.clone(),
                    snapshot.num_candidates,
                    snapshot.description.clone(),
                )
            })
        });

        ui.heading("📝 Step Info");
        ui.separator();

        if let Some((
            current_step,
            history_len,
            cost,
            applied_rule,
            suggester,
            num_candidates,
            description,
        )) = step_info
        {
            egui::Grid::new("step_info_grid")
                .num_columns(2)
                .spacing([10.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Step:");
                    ui.strong(format!("{}/{}", current_step, history_len - 1));
                    ui.end_row();

                    ui.label("Cost:");
                    ui.label(format!("{:.2}", cost));
                    ui.end_row();

                    if let Some(ref rule) = applied_rule {
                        ui.label("Rule:");
                        ui.label(
                            egui::RichText::new(rule).color(egui::Color32::from_rgb(100, 200, 150)),
                        );
                        ui.end_row();
                    }

                    if let Some(ref suggester_name) = suggester {
                        ui.label("Suggester:");
                        ui.label(
                            egui::RichText::new(suggester_name)
                                .color(egui::Color32::from_rgb(150, 150, 250)),
                        );
                        ui.end_row();
                    }

                    if let Some(num) = num_candidates {
                        ui.label("Candidates:");
                        ui.label(format!("{}", num));
                        ui.end_row();
                    }
                });

            ui.separator();
            ui.label("Description:");
            ui.label(&description);
        }
    }

    /// 最終コード表示UI
    fn ui_final_code(&mut self, ui: &mut egui::Ui) {
        // コード表示
        if let Some(ref code) = self.cached_code {
            ui.horizontal(|ui| {
                // クリップボードにコピーボタン
                if ui.button("Copy to Clipboard").clicked() {
                    ui.output_mut(|o| o.copied_text = code.clone());
                    log::info!("Code copied to clipboard");
                }

                ui.separator();

                // 行数を表示
                let line_count = code.lines().count();
                ui.label(format!("{} lines", line_count));
            });

            ui.separator();

            // シンタックスハイライト付きでコードを表示
            egui::ScrollArea::both()
                .id_salt("final_code_scroll")
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
                        code,
                        "c", // C言語風のシンタックスハイライト
                    );

                    ui.add(egui::Label::new(highlighted_code).selectable(true));
                });
        } else {
            ui.label("No code available.");
            ui.label("Load an optimized graph to view the generated code.");
            ui.add_space(20.0);

            ui.label("Tips:");
            ui.label("  - The graph must be fully lowered (contain Kernel nodes)");
            ui.label("  - Use single-stage or unified optimization for best results");
        }
    }

    /// AST最適化履歴表示UI
    fn ui_ast_history(&mut self, ui: &mut egui::Ui) {
        let history = match &self.ast_history {
            Some(h) => h.clone(),
            None => return,
        };

        let history_len = history.len();
        let current_step = self.ast_current_step;

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
                self.prev_ast_step();
            } else if next_clicked {
                self.next_ast_step();
            }

            ui.separator();

            // 最初と最後にジャンプ
            if ui
                .add_enabled(current_step > 0, egui::Button::new("⏮ First"))
                .clicked()
            {
                self.goto_ast_step(0);
            }
            if ui
                .add_enabled(current_step + 1 < history_len, egui::Button::new("Last ⏭"))
                .clicked()
            {
                self.goto_ast_step(history_len - 1);
            }
        });

        // 現在のステップの情報を表示
        if let Some(snapshot) = history.get(current_step) {
            // 適用されたルールを表示
            if let Some(ref rule_name) = snapshot.applied_rule {
                ui.horizontal(|ui| {
                    ui.label("Applied Rule:");
                    ui.label(
                        egui::RichText::new(rule_name)
                            .color(egui::Color32::from_rgb(100, 200, 150))
                            .strong(),
                    );
                });
            }

            // 提案したSuggester名を表示
            if let Some(ref suggester_name) = snapshot.suggester_name {
                ui.horizontal(|ui| {
                    ui.label("Suggester:");
                    ui.label(
                        egui::RichText::new(suggester_name)
                            .color(egui::Color32::from_rgb(150, 150, 250))
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
                let cost_str = format!("{}", snapshot.cost);
                ui.label(cost_str);

                ui.separator();

                ui.label("Candidates:");
                if let Some(num_candidates) = snapshot.num_candidates {
                    ui.label(format!("{}", num_candidates));
                } else {
                    ui.label("-");
                }

                // 代替候補がある場合は表示中の候補を表示
                if !snapshot.alternatives.is_empty() {
                    ui.separator();
                    let candidate_count = 1 + snapshot.alternatives.len();
                    if self.viewed_candidate_index == 0 {
                        ui.label(
                            egui::RichText::new(format!(
                                "Viewing: ★ Selected (1/{})",
                                candidate_count
                            ))
                            .color(egui::Color32::from_rgb(100, 200, 100)),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(format!(
                                "Viewing: Rank {} ({}/{})",
                                self.viewed_candidate_index,
                                self.viewed_candidate_index + 1,
                                candidate_count
                            ))
                            .color(egui::Color32::from_rgb(200, 150, 100)),
                        );
                    }
                }
            });
        }

        ui.separator();

        // コスト遷移グラフを表示（折りたたみ可能）
        egui::CollapsingHeader::new("Cost & Candidates Transition")
            .default_open(true)
            .show(ui, |ui| {
                egui::Resize::default()
                    .default_height(150.0)
                    .min_height(80.0)
                    .max_height(400.0)
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

                        // 候補数をコストスケールに正規化
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
                        egui_plot::Plot::new("ast_cost_plot")
                            .view_aspect(2.5)
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
                                            .color(egui::Color32::from_rgb(100, 200, 150))
                                            .name(format!("Candidates (×{:.1})", scale)),
                                    );
                                }

                                // 現在のステップを縦線で表示
                                let current_step = self.ast_current_step as f64;
                                plot_ui.vline(
                                    egui_plot::VLine::new(current_step)
                                        .color(egui::Color32::from_rgb(255, 100, 100))
                                        .name("Current Step"),
                                );
                            });
                    });
            });

        ui.separator();

        // ログを表示（折りたたみ可能）
        if let Some(snapshot) = history.get(self.ast_current_step) {
            egui::CollapsingHeader::new(format!("Debug Logs ({} entries)", snapshot.logs.len()))
                .default_open(false)
                .show(ui, |ui| {
                    egui::Resize::default()
                        .default_height(150.0)
                        .min_height(80.0)
                        .max_height(400.0)
                        .resizable(true)
                        .show(ui, |ui| {
                            if !snapshot.logs.is_empty() {
                                egui::ScrollArea::both()
                                    .id_salt("ast_logs_scroll")
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

        // コード差分表示（前のステップとの比較）- 折りたたみ可能
        if self.ast_current_step > 0 {
            if let Some(prev_snapshot) = history.get(self.ast_current_step - 1) {
                let prev_code = render_with_type(&prev_snapshot.ast, self.renderer_type);
                if let Some(ref current_code) = self.current_step_code {
                    crate::diff_viewer::show_collapsible_diff(
                        ui,
                        &prev_code,
                        current_code,
                        "Code Diff (Previous -> Current)",
                        "ast_code_diff",
                        true, // デフォルトで開く
                        None,
                    );
                    ui.separator();
                }
            }
        }

        // 現在のステップのコードを表示
        egui::CollapsingHeader::new("Generated Code")
            .default_open(true)
            .show(ui, |ui| {
                if let Some(ref code) = self.current_step_code {
                    ui.horizontal(|ui| {
                        // クリップボードにコピーボタン
                        if ui.button("Copy to Clipboard").clicked() {
                            ui.output_mut(|o| o.copied_text = code.clone());
                            log::info!("Code copied to clipboard");
                        }

                        ui.separator();

                        // 行数を表示
                        let line_count = code.lines().count();
                        ui.label(format!("{} lines", line_count));
                    });

                    ui.separator();

                    // シンタックスハイライト付きでコードを表示
                    egui::ScrollArea::both()
                        .id_salt("ast_step_code_scroll")
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
                                code,
                                "c",
                            );

                            ui.add(egui::Label::new(highlighted_code).selectable(true));
                        });
                } else {
                    ui.label("No code available for this step.");
                }
            });
    }
}
