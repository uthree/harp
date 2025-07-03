use harp::prelude::*;

fn main() {
    let tracker = ShapeTracker::full(s![2, "a", 4]);
    println!("{:?}", tracker);
}
