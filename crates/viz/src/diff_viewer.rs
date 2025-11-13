//! テキストdiff表示の共通UIコンポーネント

use similar::{ChangeTag, TextDiff};

/// Diff表示の設定
#[derive(Clone)]
pub struct DiffViewerConfig {
    /// 削除行の色
    pub delete_color: egui::Color32,
    /// 追加行の色
    pub insert_color: egui::Color32,
    /// 変更なし行の色
    pub equal_color: egui::Color32,
    /// モノスペースフォントを使用するか
    pub monospace: bool,
}

impl Default for DiffViewerConfig {
    fn default() -> Self {
        Self {
            delete_color: egui::Color32::from_rgb(255, 150, 150),
            insert_color: egui::Color32::from_rgb(150, 255, 150),
            equal_color: egui::Color32::GRAY,
            monospace: true,
        }
    }
}

/// 2つのテキスト間のdiffを表示
///
/// # 引数
/// * `ui` - egui UI
/// * `prev_text` - 前のテキスト
/// * `current_text` - 現在のテキスト
/// * `config` - Diff表示の設定（Noneの場合はデフォルト）
pub fn show_text_diff(
    ui: &mut egui::Ui,
    prev_text: &str,
    current_text: &str,
    config: Option<&DiffViewerConfig>,
) {
    let config = config.cloned().unwrap_or_default();

    let diff = TextDiff::from_lines(prev_text, current_text);

    for change in diff.iter_all_changes() {
        let (prefix, color) = match change.tag() {
            ChangeTag::Delete => ("-", config.delete_color),
            ChangeTag::Insert => ("+", config.insert_color),
            ChangeTag::Equal => (" ", config.equal_color),
        };

        // changeには既に改行が含まれているので、末尾の改行を削除
        let change_str = change.to_string();
        let change_trimmed = change_str.trim_end_matches('\n').trim_end_matches('\r');
        let line = format!("{} {}", prefix, change_trimmed);
        let text = if config.monospace {
            egui::RichText::new(line).monospace()
        } else {
            egui::RichText::new(line)
        };

        ui.colored_label(color, text);
    }
}

/// スクロール可能でリサイズ可能なdiff表示
///
/// # 引数
/// * `ui` - egui UI
/// * `prev_text` - 前のテキスト
/// * `current_text` - 現在のテキスト
/// * `id_salt` - ScrollAreaのID（同じ画面に複数ある場合に区別するため）
/// * `config` - Diff表示の設定（Noneの場合はデフォルト）
pub fn show_resizable_diff(
    ui: &mut egui::Ui,
    prev_text: &str,
    current_text: &str,
    id_salt: &str,
    config: Option<&DiffViewerConfig>,
) {
    egui::Resize::default()
        .default_height(300.0)
        .min_height(100.0)
        .max_height(800.0)
        .resizable(true)
        .show(ui, |ui| {
            egui::ScrollArea::both()
                .id_salt(id_salt)
                .max_height(ui.available_height())
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    show_text_diff(ui, prev_text, current_text, config);
                });
        });
}

/// 折りたたみ可能なdiff表示
///
/// # 引数
/// * `ui` - egui UI
/// * `prev_text` - 前のテキスト
/// * `current_text` - 現在のテキスト
/// * `header` - ヘッダーテキスト
/// * `id_salt` - ScrollAreaのID
/// * `default_open` - デフォルトで開いているか
/// * `config` - Diff表示の設定（Noneの場合はデフォルト）
pub fn show_collapsible_diff(
    ui: &mut egui::Ui,
    prev_text: &str,
    current_text: &str,
    header: &str,
    id_salt: &str,
    default_open: bool,
    config: Option<&DiffViewerConfig>,
) {
    egui::CollapsingHeader::new(header)
        .default_open(default_open)
        .show(ui, |ui| {
            show_resizable_diff(ui, prev_text, current_text, id_salt, config);
        });
}
