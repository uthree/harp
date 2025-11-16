use harp::backend::openmp::CRenderer;
use harp::graph::{DType, Graph};
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

    // スカラー定数は自動的にブロードキャストされる
    // a + const
    let result = a + 5.0f32;
    graph.output("result", result);

    // Lower to AST
    let program = lower(graph);

    // Render to code
    let renderer = CRenderer::new();
    let code = render_ast_with(&program, &renderer);

    println!("{}", code);
}
