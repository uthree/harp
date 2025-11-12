use harp::ast::{AstNode, Literal, Scope};
use harp::opt::ast::transforms::tile_loop;

fn main() {
    // 定数stopを持つループ
    let loop_with_const = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(AstNode::Const(Literal::Int(0))),
        step: Box::new(AstNode::Const(Literal::Int(1))),
        stop: Box::new(AstNode::Const(Literal::Int(100))), // 定数
        body: Box::new(AstNode::Var("body".to_string())),
    };

    println!("=== テスト1: 定数stopのループ ===");
    match tile_loop(&loop_with_const, 32) {
        Some(tiled) => println!("タイル化成功:\n{:#?}", tiled),
        None => println!("タイル化失敗"),
    }

    // 変数stopを持つループ（lowererが生成する形式）
    let loop_with_var = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(AstNode::Const(Literal::Int(0))),
        step: Box::new(AstNode::Const(Literal::Int(1))),
        stop: Box::new(AstNode::Var("M".to_string())), // 変数
        body: Box::new(AstNode::Var("body".to_string())),
    };

    println!("\n=== テスト2: 変数stopのループ ===");
    match tile_loop(&loop_with_var, 32) {
        Some(tiled) => println!("タイル化成功:\n{:#?}", tiled),
        None => println!("タイル化失敗（これがpipeline_demoで起きている問題）"),
    }

    // 計算の検証
    println!("\n=== 計算の検証 ===");
    let stop_val = 100;
    let tile_size = 32;
    let aligned_stop = (stop_val / tile_size) * tile_size;
    println!("stop_val: {}", stop_val);
    println!("tile_size: {}", tile_size);
    println!("aligned_stop: {} (正しい計算: {} / {} * {} = {})",
        aligned_stop, stop_val, tile_size, tile_size, aligned_stop);
    println!("remainder: {} から {} まで", aligned_stop, stop_val);
}
