use harp::ast::DType;
use harp::backend::generic::CBackend;
use harp::backend::Backend;
use harp::graph::Graph;
use harp::s;

fn main() {
    let _ = env_logger::try_init();

    let mut backend = CBackend::new();
    let mut graph = Graph::new();

    let input_node = graph.input(DType::F32, s![2, 3]);
    graph.output(input_node);

    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_buffer = harp::backend::c::CBuffer::from_slice(&input_data, &[2, 3], DType::F32);

    let outputs = backend.execute(&graph, vec![input_buffer]);

    eprintln!("Output buffer dtype: {:?}", outputs[0].dtype);
    eprintln!("Output buffer shape: {:?}", outputs[0].shape);

    let output_data: Vec<f32> = outputs[0].to_vec();
    println!("Input:  {:?}", input_data);
    println!("Output: {:?}", output_data);
}
