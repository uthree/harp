use harp::shape::symbolic::Expr;
use harp::shape::tracker::ShapeTracker;

fn main() {
    let tracker = ShapeTracker::full(vec![2, 3]);
    println!("{:?}", tracker);
}
