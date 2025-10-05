pub mod ast;
pub mod backend;
pub mod graph;
pub mod lowerer;
pub mod opt;

/// VIZ=1のときにメッセージを表示するマクロ
/// main関数の最後に配置して使用する
/// 実際のビジュアライザー起動はharp_viz::launch_with_global_snapshots()を直接呼ぶ
#[macro_export]
macro_rules! launch_viz_if_enabled {
    () => {
        if $crate::opt::graph::is_viz_enabled() {
            let snapshots = $crate::opt::graph::take_global_snapshots();
            if snapshots.is_empty() {
                eprintln!("VIZ=1 is set but no optimization snapshots were recorded.");
            } else {
                eprintln!("VIZ=1 is set. {} snapshots recorded.", snapshots.len());
                eprintln!("To visualize, use: harp_viz::launch_with_global_snapshots()");
                eprintln!("Note: This requires adding harp-viz as a dependency.");
            }
        }
    };
}
