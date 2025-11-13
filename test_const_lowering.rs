use harp::backend::openmp::CRenderer;
use harp::graph::{DType, Graph, GraphNode};
use harp::lowerer::lower;
use harp::ast::renderer::render_ast_with;

fn main() {
    // a + 5.0 のようなグラフを作成（定数をブロードキャスト）
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![10])
        .build();

    // 定数ノードを作成してブロードキャスト
    let const_val = GraphNode::constant(5.0);
    // スカラーを[1]にreshapeしてから[10]にexpand
    let const_reshaped = const_val.view(const_val.view.clone().unsqueeze(0));
    let const_broadcast = const_reshaped.view(const_reshaped.view.clone().expand(vec![10.into()]));

    // a + const
    let result = a + const_broadcast;
    graph.output("result", result);

    // Lower to AST
    let program = lower(graph);

    // Render to code
    let renderer = CRenderer::new();
    let code = render_ast_with(&program, &renderer);

    println!("{}", code);
}
