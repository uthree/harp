//! パフォーマンス統計を可視化するビューア

/// パフォーマンスビューアアプリケーション
pub struct PerfViewerApp {
    /// サンプルデータ
    sample_data: Vec<PerfSample>,
}

/// パフォーマンスサンプル
#[derive(Clone, Debug)]
pub struct PerfSample {
    /// サンプルの名前
    pub name: String,
    /// 実行時間（ミリ秒）
    pub time_ms: f32,
    /// メモリ使用量（MB）
    pub memory_mb: f32,
}

impl Default for PerfViewerApp {
    fn default() -> Self {
        Self::new()
    }
}

impl PerfViewerApp {
    /// 新しいPerfViewerAppを作成
    pub fn new() -> Self {
        Self {
            sample_data: vec![
                PerfSample {
                    name: "Sample 1".to_string(),
                    time_ms: 10.5,
                    memory_mb: 128.0,
                },
                PerfSample {
                    name: "Sample 2".to_string(),
                    time_ms: 15.2,
                    memory_mb: 256.0,
                },
                PerfSample {
                    name: "Sample 3".to_string(),
                    time_ms: 8.3,
                    memory_mb: 192.0,
                },
            ],
        }
    }

    /// パフォーマンスサンプルを追加
    pub fn add_sample(&mut self, sample: PerfSample) {
        self.sample_data.push(sample);
    }

    /// UIを描画
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Performance Viewer");
        ui.separator();

        // サマリー統計
        if !self.sample_data.is_empty() {
            let total_time: f32 = self.sample_data.iter().map(|s| s.time_ms).sum();
            let avg_time = total_time / self.sample_data.len() as f32;
            let max_time = self
                .sample_data
                .iter()
                .map(|s| s.time_ms)
                .fold(0.0f32, f32::max);

            ui.horizontal(|ui| {
                ui.label("Total samples:");
                ui.label(self.sample_data.len().to_string());
            });

            ui.horizontal(|ui| {
                ui.label("Total time:");
                ui.label(format!("{:.2} ms", total_time));
            });

            ui.horizontal(|ui| {
                ui.label("Average time:");
                ui.label(format!("{:.2} ms", avg_time));
            });

            ui.horizontal(|ui| {
                ui.label("Max time:");
                ui.label(format!("{:.2} ms", max_time));
            });

            ui.separator();
        }

        // サンプルデータをテーブルで表示
        ui.label("Samples:");

        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("perf_samples_grid")
                .striped(true)
                .show(ui, |ui| {
                    // ヘッダー
                    ui.label("Name");
                    ui.label("Time (ms)");
                    ui.label("Memory (MB)");
                    ui.end_row();

                    // データ行
                    for sample in &self.sample_data {
                        ui.label(&sample.name);
                        ui.label(format!("{:.2}", sample.time_ms));
                        ui.label(format!("{:.1}", sample.memory_mb));
                        ui.end_row();
                    }
                });
        });

        ui.separator();

        // 簡易的な棒グラフ
        if !self.sample_data.is_empty() {
            ui.label("Time Distribution:");
            let max_time = self
                .sample_data
                .iter()
                .map(|s| s.time_ms)
                .fold(0.0f32, f32::max);

            for sample in &self.sample_data {
                let ratio = sample.time_ms / max_time;
                ui.horizontal(|ui| {
                    ui.label(&sample.name);
                    let bar_width = 200.0 * ratio;
                    let (rect, _response) =
                        ui.allocate_exact_size(egui::vec2(bar_width, 20.0), egui::Sense::hover());
                    ui.painter()
                        .rect_filled(rect, 2.0, egui::Color32::from_rgb(100, 150, 200));
                    ui.label(format!("{:.2} ms", sample.time_ms));
                });
            }
        }
    }
}
