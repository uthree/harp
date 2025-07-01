use harp::shape::symbolic::Expr;
use harp::shape::tracker::ShapeTracker;

fn main() {
    let tracker = ShapeTracker::from_shape(vec![2, 3]);
    println!("{:?}", tracker);
}
