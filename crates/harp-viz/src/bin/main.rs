use harp_viz::GraphVisualizerApp;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Harp Computational Graph Visualizer",
        options,
        Box::new(|_cc| Ok(Box::new(GraphVisualizerApp::default()))),
    )
}
