use harp::shape::symbolic::Expr;
use harp::shape::tracker::ShapeTracker;

fn main() {
    let expr = Expr::Int(1) + Expr::Int(2) * Expr::Var("a".to_string());
    let expr = expr.simplify();
    println!("{:}", expr);
    let tracker = ShapeTracker::from_shape(vec![2, 3, 4]);
    println!("{:?}", tracker);
}
