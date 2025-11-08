//! AST最適化を可視化するビューア

use harp::ast::renderer::render_ast;
use harp::opt::ast::OptimizationHistory;
use similar::{ChangeTag, TextDiff};

/// ASTビューアアプリケーション
pub struct AstViewerApp {
    /// 最適化履歴
    optimization_history: Option<OptimizationHistory>,
    /// 現在表示中のステップインデックス
    current_step_index: usize,
    /// 選択中のビーム内の候補（rank）
    selected_rank: usize,
    /// コスト遷移グラフを表示するかどうか
    show_cost_graph: bool,
    /// Diffを表示するかどうか
    show_diff: bool,
}

impl Default for AstViewerApp {
    fn default() -> Self {
        Self::new()
    }
}

impl AstViewerApp {
    /// 新しいAstViewerAppを作成
    pub fn new() -> Self {
        Self {
            optimization_history: None,
            current_step_index: 0,
            selected_rank: 0,
            show_cost_graph: false,
            show_diff: false,
        }
    }

    /// 最適化履歴を読み込む
    pub fn load_history(&mut self, history: OptimizationHistory) {
        if history.is_empty() {
            log::warn!("Attempted to load empty AST optimization history");
            return;
        }

        self.optimization_history = Some(history);
        self.current_step_index = 0;
        self.selected_rank = 0;

        log::info!("AST optimization history loaded");
    }

    /// 次のステップに進む
    pub fn next_step(&mut self) {
        if let Some(ref history) = self.optimization_history {
            let current_step = self.get_current_step_number();
            let max_step = self.get_max_step_number();

            if current_step < max_step {
                // 次のステップを探す
                for i in (self.current_step_index + 1)..history.len() {
                    if let Some(snapshot) = history.get(i) {
                        if snapshot.step > current_step {
                            self.current_step_index = i;
                            self.selected_rank = 0;
                            return;
                        }
                    }
                }
            }
        }
    }

    /// 前のステップに戻る
    pub fn prev_step(&mut self) {
        if let Some(ref history) = self.optimization_history {
            let current_step = self.get_current_step_number();

            if current_step > 0 {
                // 前のステップを探す（逆順で検索）
                for i in (0..self.current_step_index).rev() {
                    if let Some(snapshot) = history.get(i) {
                        if snapshot.step < current_step {
                            self.current_step_index = i;
                            self.selected_rank = 0;
                            return;
                        }
                    }
                }
            }
        }
    }

    /// 現在のステップ番号を取得
    fn get_current_step_number(&self) -> usize {
        if let Some(ref history) = self.optimization_history {
            if let Some(snapshot) = history.get(self.current_step_index) {
                return snapshot.step;
            }
        }
        0
    }

    /// 最大ステップ番号を取得
    fn get_max_step_number(&self) -> usize {
        if let Some(ref history) = self.optimization_history {
            history
                .snapshots()
                .iter()
                .map(|s| s.step)
                .max()
                .unwrap_or(0)
        } else {
            0
        }
    }

    /// 現在のステップのすべての候補を取得
    fn get_current_step_candidates(&self) -> Vec<&harp::opt::ast::OptimizationSnapshot> {
        if let Some(ref history) = self.optimization_history {
            let current_step = self.get_current_step_number();
            history.get_step(current_step)
        } else {
            Vec::new()
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

        ui.horizontal(|ui| {
            ui.heading("AST Optimizer Viewer");
            ui.add_space(20.0);

            // コスト遷移グラフ表示トグルボタン（最適化履歴がある場合のみ）
            if self.optimization_history.is_some() {
                let cost_button_text = if self.show_cost_graph {
                    "Hide Cost Graph"
                } else {
                    "Show Cost Graph"
                };
                if ui.button(cost_button_text).clicked() {
                    self.show_cost_graph = !self.show_cost_graph;
                }

                ui.add_space(10.0);

                // Diff表示トグルボタン（ステップ0でない場合のみ）
                if self.get_current_step_number() > 0 {
                    let diff_button_text = if self.show_diff {
                        "Hide Diff"
                    } else {
                        "Show Diff"
                    };
                    if ui.button(diff_button_text).clicked() {
                        self.show_diff = !self.show_diff;
                    }
                }
            }
        });
        ui.separator();

        if self.optimization_history.is_none() {
            ui.label("No AST optimization history loaded.");
            ui.label("Load an optimization history to visualize it here.");
            return;
        }

        let current_step = self.get_current_step_number();
        let max_step = self.get_max_step_number();

        // ナビゲーションボタン
        let (prev_clicked, next_clicked) = ui
            .horizontal(|ui| {
                // 前のステップボタン
                let prev_clicked = ui
                    .add_enabled(current_step > 0, egui::Button::new("◀ Prev"))
                    .clicked();

                // ステップ情報表示
                ui.label(format!("Step: {} / {}", current_step, max_step));

                // 次のステップボタン
                let next_clicked = ui
                    .add_enabled(current_step < max_step, egui::Button::new("Next ▶"))
                    .clicked();

                (prev_clicked, next_clicked)
            })
            .inner;

        if prev_clicked {
            self.prev_step();
        } else if next_clicked {
            self.next_step();
        }

        // ステップスライダー
        let new_step = ui
            .horizontal(|ui| {
                ui.label("Step:");
                let mut step = current_step;
                if ui
                    .add(egui::Slider::new(&mut step, 0..=max_step).show_value(false))
                    .changed()
                {
                    Some(step)
                } else {
                    None
                }
            })
            .inner;

        if let Some(step) = new_step {
            let history = self.optimization_history.as_ref().unwrap();
            // スライダーで選択されたステップに移動
            for i in 0..history.len() {
                if let Some(snapshot) = history.get(i) {
                    if snapshot.step == step {
                        self.current_step_index = i;
                        self.selected_rank = 0;
                        break;
                    }
                }
            }
        }

        // 現在のステップの情報を表示
        let candidates = self.get_current_step_candidates();
        if let Some(current_snapshot) = candidates.get(self.selected_rank) {
            ui.horizontal(|ui| {
                ui.label("Cost:");
                ui.label(format!("{:.2}", current_snapshot.cost));
            });
            ui.horizontal(|ui| {
                ui.label("Description:");
                ui.label(&current_snapshot.description);
            });
        }

        ui.separator();

        // コスト遷移グラフを表示
        if self.show_cost_graph {
            ui.heading("Cost Transition");

            // コストデータを収集
            let cost_points: Vec<[f64; 2]> = self
                .optimization_history
                .as_ref()
                .unwrap()
                .cost_transition()
                .iter()
                .map(|(step, cost)| [*step as f64, *cost as f64])
                .collect();

            // プロットを表示
            egui_plot::Plot::new("ast_cost_plot")
                .view_aspect(2.0)
                .height(200.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        egui_plot::Line::new(cost_points)
                            .color(egui::Color32::from_rgb(100, 200, 150))
                            .name("Cost"),
                    );

                    // 現在のステップを縦線で表示
                    let current_step_f64 = current_step as f64;
                    plot_ui.vline(
                        egui_plot::VLine::new(current_step_f64)
                            .color(egui::Color32::from_rgb(255, 100, 100))
                            .name("Current Step"),
                    );
                });

            ui.separator();
        }

        // 候補リストの情報を先に収集
        let candidate_info: Vec<(usize, f32)> =
            candidates.iter().map(|c| (c.rank, c.cost)).collect();
        let selected_rank = self.selected_rank;
        let selected_code = candidates.get(selected_rank).map(|s| render_ast(&s.ast));

        // 前のステップのコードを取得（Diff表示用）
        let prev_code = if self.show_diff && current_step > 0 {
            // 前のステップの最良候補を取得
            let prev_step_candidates = self
                .optimization_history
                .as_ref()
                .unwrap()
                .get_step(current_step - 1);
            prev_step_candidates
                .iter()
                .find(|c| c.rank == 0)
                .map(|s| render_ast(&s.ast))
        } else {
            None
        };

        // コード表示用のクローンを作成
        let code_for_display = selected_code.clone();

        // 左右分割: 左側に候補リスト、右側にコード表示
        ui.columns(2, |columns| {
            // 左側: ビーム内の候補リスト
            columns[0].vertical(|ui| {
                ui.heading("Beam Candidates");
                ui.separator();

                egui::ScrollArea::vertical()
                    .id_salt("beam_candidates_scroll")
                    .max_height(ui.available_height())
                    .show(ui, |ui| {
                        for (i, (rank, cost)) in candidate_info.iter().enumerate() {
                            let is_selected = i == selected_rank;
                            let button_text = format!("Rank {}: Cost {:.2}", rank, cost);

                            if ui.selectable_label(is_selected, button_text).clicked() {
                                self.selected_rank = i;
                            }
                        }
                    });
            });

            // 右側: 選択したASTのコード表示
            columns[1].vertical(|ui| {
                ui.heading("AST Code");
                ui.separator();

                if let Some(rendered_code) = code_for_display {
                    egui::ScrollArea::vertical()
                        .id_salt("ast_code_scroll")
                        .max_height(ui.available_height())
                        .show(ui, |ui| {
                            // シンタックスハイライト付きでコードを表示
                            let theme = egui_extras::syntax_highlighting::CodeTheme::from_memory(
                                ui.ctx(),
                                ui.style(),
                            );

                            let code = egui_extras::syntax_highlighting::highlight(
                                ui.ctx(),
                                ui.style(),
                                &theme,
                                &rendered_code,
                                "rs",
                            );

                            ui.add(egui::Label::new(code).selectable(true));
                        });
                } else {
                    ui.label("No candidate selected");
                }
            });
        });

        // Diffを表示
        if self.show_diff {
            if let (Some(ref prev), Some(ref current)) = (&prev_code, &selected_code) {
                ui.separator();
                ui.heading("Code Diff (Previous → Current)");
                ui.separator();

                egui::ScrollArea::vertical()
                    .id_salt("diff_scroll")
                    .max_height(300.0)
                    .show(ui, |ui| {
                        // テキストdiffを計算
                        let diff = TextDiff::from_lines(prev, current);

                        // diffの各行を表示
                        for change in diff.iter_all_changes() {
                            let (prefix, color) = match change.tag() {
                                ChangeTag::Delete => ("- ", egui::Color32::from_rgb(255, 100, 100)),
                                ChangeTag::Insert => ("+ ", egui::Color32::from_rgb(100, 255, 100)),
                                ChangeTag::Equal => ("  ", egui::Color32::GRAY),
                            };

                            let line = format!("{}{}", prefix, change.value());
                            ui.colored_label(color, egui::RichText::new(line).monospace());
                        }
                    });
            } else if prev_code.is_none() {
                ui.separator();
                ui.label("No previous step available for diff.");
            }
        }
    }
}
