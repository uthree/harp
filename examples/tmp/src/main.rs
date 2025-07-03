use harp::prelude::*;

fn main() {
    let mut graph = Graph::new();
    let a = graph.input(s![1], DataType::Float32);
    let b = graph.input(s![1], DataType::Float32);
    let c = a.clone() + b.clone() + a.clone();
    println!("{:?}", c)
}
