use harp::prelude::*;

fn main() {
    let tracker = ShapeTracker::full(shape![2, 3]);
    println!("{:?}", tracker);
}
