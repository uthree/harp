//! シンプルなHarp Visualizerの実行例

use harp_viz::HarpVizApp;

fn main() -> eframe::Result {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_title("Harp Visualizer"),
        ..Default::default()
    };

    eframe::run_native(
        "Harp Visualizer",
        options,
        Box::new(|_cc| Ok(Box::new(HarpVizApp::new()))),
    )
}
