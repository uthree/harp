//! AST最適化を可視化するビューア

use harp::ast::renderer::render_ast_with;
use harp::backend::c_like::CLikeRenderer;
use harp::backend::openmp::CRenderer;
use harp::opt::ast::OptimizationHistory;
use similar::{ChangeTag, TextDiff};
use std::collections::HashMap;

/// ASTビューアアプリケーション（ジェネリックレンダラー対応）
///
/// # 型パラメータ
/// * `R` - ASTをレンダリングするレンダラー（`CLikeRenderer`を実装している必要があります）
///
/// # 複数のFunction対応
/// このビューアは、Program内の複数のFunctionの最適化履歴を保持できます。
/// Function名で切り替えて、それぞれの最適化過程を確認できます。
pub struct AstViewerApp<R = CRenderer>
where
    R: CLikeRenderer + Clone,
{
    /// Function名ごとの最適化履歴
    function_histories: HashMap<String, OptimizationHistory>,
    /// 現在選択されているFunction名
    selected_function: Option<String>,
    /// 現在表示中のステップインデックス
    current_step_index: usize,
    /// ASTレンダラー
    renderer: R,
}

impl Default for AstViewerApp<CRenderer> {
    fn default() -> Self {
        Self::new()
    }
}

impl AstViewerApp<CRenderer> {
    /// 新しいAstViewerAppを作成（デフォルトでCRendererを使用）
    pub fn new() -> Self {
        Self::with_renderer(CRenderer::new())
    }
}

impl<R> AstViewerApp<R>
where
    R: CLikeRenderer + Clone,
{
    /// カスタムレンダラーを使用してAstViewerAppを作成
    pub fn with_renderer(renderer: R) -> Self {
        Self {
            function_histories: HashMap::new(),
            selected_function: None,
            current_step_index: 0,
            renderer,
        }
    }

    /// 単一のFunction用の最適化履歴を読み込む（後方互換性のため）
    pub fn load_history(&mut self, history: OptimizationHistory) {
        self.load_function_history("main".to_string(), history);
    }

    /// 指定されたFunction名で最適化履歴を読み込む
    pub fn load_function_history(&mut self, function_name: String, history: OptimizationHistory) {
        if history.is_empty() {
            log::warn!(
                "Attempted to load empty AST optimization history for function '{}'",
                function_name
            );
            return;
        }

        self.function_histories
            .insert(function_name.clone(), history);

        // 最初のFunctionを自動選択
        if self.selected_function.is_none() {
            self.selected_function = Some(function_name.clone());
        }

        self.current_step_index = 0;

        log::info!(
            "AST optimization history loaded for function '{}'",
            function_name
        );
    }

    /// 複数のFunctionの最適化履歴を一括で読み込む
    pub fn load_multiple_histories(&mut self, histories: HashMap<String, OptimizationHistory>) {
        for (name, history) in histories {
            self.load_function_history(name, history);
        }
    }

    /// 現在選択されているFunctionの最適化履歴を取得
    fn current_history(&self) -> Option<&OptimizationHistory> {
        self.selected_function
            .as_ref()
            .and_then(|name| self.function_histories.get(name))
    }

    /// 次のステップに進む
    pub fn next_step(&mut self) {
        if let Some(history) = self.current_history() {
            let current_step = self.get_current_step_number();
            let max_step = self.get_max_step_number();

            if current_step < max_step {
                // 次のステップを探す
                for i in (self.current_step_index + 1)..history.len() {
                    if let Some(snapshot) = history.get(i) {
                        if snapshot.step > current_step {
                            self.current_step_index = i;
                            return;
                        }
                    }
                }
            }
        }
    }

    /// 前のステップに戻る
    pub fn prev_step(&mut self) {
        if let Some(history) = self.current_history() {
            let current_step = self.get_current_step_number();

            if current_step > 0 {
                // 前のステップを探す（逆順で検索）
                for i in (0..self.current_step_index).rev() {
                    if let Some(snapshot) = history.get(i) {
                        if snapshot.step < current_step {
                            self.current_step_index = i;
                            return;
                        }
                    }
                }
            }
        }
    }

    /// 現在のステップ番号を取得
    fn get_current_step_number(&self) -> usize {
        if let Some(history) = self.current_history() {
            if let Some(snapshot) = history.get(self.current_step_index) {
                return snapshot.step;
            }
        }
        0
    }

    /// 最大ステップ番号を取得
    fn get_max_step_number(&self) -> usize {
        if let Some(history) = self.current_history() {
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
        if let Some(history) = self.current_history() {
            let current_step = self.get_current_step_number();
            history.get_step(current_step)
        } else {
            Vec::new()
        }
    }

    /// UIを描画
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        // キーボード入力処理（左右矢印キーでステップ間を移動）
        if self.current_history().is_some() {
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

            // Function選択ドロップダウン（複数のFunctionがある場合のみ）
            if self.function_histories.len() > 1 {
                ui.label("Function:");
                let mut function_names: Vec<String> =
                    self.function_histories.keys().cloned().collect();
                function_names.sort();

                let selected = self
                    .selected_function
                    .clone()
                    .unwrap_or_else(|| "None".to_string());

                egui::ComboBox::from_id_salt("function_selector")
                    .selected_text(&selected)
                    .show_ui(ui, |ui| {
                        for name in &function_names {
                            if ui
                                .selectable_value(
                                    &mut self.selected_function,
                                    Some(name.clone()),
                                    name,
                                )
                                .clicked()
                            {
                                // Function切り替え時にステップをリセット
                                self.current_step_index = 0;
                            }
                        }
                    });

                ui.add_space(10.0);
            }
        });
        ui.separator();

        if self.current_history().is_none() {
            ui.label("No AST optimization history loaded.");
            if self.function_histories.is_empty() {
                ui.label("Load an optimization history to visualize it here.");
            } else {
                ui.label("Select a function from the dropdown above.");
            }
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
            if let Some(history) = self.current_history() {
                // スライダーで選択されたステップに移動
                for i in 0..history.len() {
                    if let Some(snapshot) = history.get(i) {
                        if snapshot.step == step {
                            self.current_step_index = i;
                            break;
                        }
                    }
                }
            }
        }

        // メインコンテンツ全体をスクロール可能に
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
        // 現在のステップの情報を表示（常にrank 0の候補を表示）
        let candidates = self.get_current_step_candidates();
        if let Some(current_snapshot) = candidates.first() {
            ui.horizontal(|ui| {
                ui.label("Cost:");
                // ASTのコストは非常に小さい（1e-9スケール）ので常に科学記数法で表示
                let cost_str = format!("{:.6e}", current_snapshot.cost);
                ui.label(cost_str);
            });
            ui.horizontal(|ui| {
                ui.label("Description:");
                ui.label(&current_snapshot.description);
            });
        }

        ui.separator();

        // コスト遷移グラフを表示（折りたたみ可能、高さリサイズ可能）
        egui::CollapsingHeader::new("Cost Transition")
            .default_open(true)
            .show(ui, |ui| {
                egui::Resize::default()
                    .default_height(200.0)
                    .min_height(100.0)
                    .max_height(600.0)
                    .resizable(true)
                    .show(ui, |ui| {
                        // コストデータを収集
                        let cost_points: Vec<[f64; 2]> = self
                            .current_history()
                            .map(|h| {
                                h.cost_transition()
                                    .iter()
                                    .map(|(step, cost)| [*step as f64, *cost as f64])
                                    .collect()
                            })
                            .unwrap_or_default();

                        // プロットを表示
                        egui_plot::Plot::new("ast_cost_plot")
                            .view_aspect(2.0)
                            .height(ui.available_height())
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
                    });
            });

        ui.separator();

        // 採用されたASTのコード（rank 0）を取得
        let selected_code = candidates.first()
            .map(|s| render_ast_with(&s.ast, &self.renderer));

        // ログを先に取得（借用エラー回避のため）
        let logs = candidates.first()
            .map(|s| s.logs.clone())
            .unwrap_or_default();

        // 前のステップのコードを取得（Diff表示用）
        let prev_code = if current_step > 0 {
            // 前のステップの最良候補（rank 0）を取得
            self.current_history().and_then(|h| {
                let prev_step_candidates = h.get_step(current_step - 1);
                prev_step_candidates
                    .iter()
                    .find(|c| c.rank == 0)
                    .map(|s| render_ast_with(&s.ast, &self.renderer))
            })
        } else {
            None
        };

        // コード表示用のクローンを作成
        let code_for_display = selected_code.clone();

        // ASTのコード表示
        ui.vertical(|ui| {
                egui::CollapsingHeader::new("AST Code")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // クリップボードにコピーボタン
                            if let Some(ref code) = code_for_display {
                                if ui.button("📋 Copy").clicked() {
                                    ui.output_mut(|o| o.copied_text = code.clone());
                                }
                            }
                        });

                        if let Some(rendered_code) = code_for_display {
                            // 高さをリサイズ可能に
                            egui::Resize::default()
                                .default_height(400.0)
                                .min_height(200.0)
                                .max_height(1000.0)
                                .resizable(true)
                                .show(ui, |ui| {
                                    egui::ScrollArea::both() // 縦横両方にスクロール可能
                                        .id_salt("ast_code_scroll")
                                        .max_height(ui.available_height())
                                        .auto_shrink([false, false]) // 自動縮小を無効化して全幅を使う
                                        .show(ui, |ui| {
                                            // シンタックスハイライト付きでコードを表示
                                            let theme =
                                                egui_extras::syntax_highlighting::CodeTheme::from_memory(
                                                    ui.ctx(),
                                                    ui.style(),
                                                );

                                            let code = egui_extras::syntax_highlighting::highlight(
                                                ui.ctx(),
                                                ui.style(),
                                                &theme,
                                                &rendered_code,
                                                "c", // C言語風のシンタックスハイライト
                                            );

                                            ui.add(egui::Label::new(code).selectable(true)); // 折り返しなし、ScrollArea::both()で横スクロール対応
                                        });
                                });
                        } else {
                            ui.label("No candidate selected");
                        }
                    });
            });

        // Diffを表示（折りたたみ可能、高さリサイズ可能）
        if let (Some(ref prev), Some(ref current)) = (&prev_code, &selected_code) {
            ui.separator();

            egui::CollapsingHeader::new("Code Diff (Previous -> Current)")
                .default_open(false)
                .show(ui, |ui| {
                    egui::Resize::default()
                        .default_height(300.0)
                        .min_height(100.0)
                        .max_height(800.0)
                        .resizable(true)
                        .show(ui, |ui| {
                            egui::ScrollArea::both() // 縦横両方にスクロール可能
                                .id_salt("diff_scroll")
                                .max_height(ui.available_height())
                                .auto_shrink([false, false]) // 自動縮小を無効化して全幅を使う
                                .show(ui, |ui| {
                                    // テキストdiffを計算
                                    let diff = TextDiff::from_lines(prev, current);

                                    // diffの各行を表示
                                    for change in diff.iter_all_changes() {
                                        let (prefix, color) = match change.tag() {
                                            ChangeTag::Delete => {
                                                ("- ", egui::Color32::from_rgb(255, 100, 100))
                                            }
                                            ChangeTag::Insert => {
                                                ("+ ", egui::Color32::from_rgb(100, 255, 100))
                                            }
                                            ChangeTag::Equal => ("  ", egui::Color32::GRAY),
                                        };

                                        let line = format!("{}{}", prefix, change.value());
                                        ui.colored_label(
                                            color,
                                            egui::RichText::new(line).monospace(),
                                        );
                                    }
                                });
                        });
                    });
            }

        // ログを表示（折りたたみ可能、高さリサイズ可能）
        ui.separator();

        egui::CollapsingHeader::new(format!("Debug Logs ({} entries)", logs.len()))
            .default_open(false)
            .show(ui, |ui| {
                egui::Resize::default()
                    .default_height(300.0)
                    .min_height(100.0)
                    .max_height(800.0)
                    .resizable(true)
                    .show(ui, |ui| {
                        if !logs.is_empty() {
                            egui::ScrollArea::both() // 長いログ行にも対応
                                .id_salt("logs_scroll")
                                .max_height(ui.available_height())
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    for log_line in &logs {
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
        }); // ScrollArea::vertical() を閉じる
    }
}
