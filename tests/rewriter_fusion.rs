use harp::node::{capture, constant};
use harp::pattern::{RewriteRule, Rewriter};
use harp::rewrite_rule;

#[test]
fn test_rewriter_fusion() {
    let rule1 = rewrite_rule!(let x = capture("x"); x.clone() + constant(0.0f32) => x);
    let rewriter1 = Rewriter::new("rewriter1", vec![rule1]);

    let rule2 = rewrite_rule!(let x = capture("x"); x.clone() * constant(1.0f32) => x);
    let rewriter2 = Rewriter::new("rewriter2", vec![rule2]);

    let fused = rewriter1 + rewriter2;

    assert_eq!(fused.name, "fused(rewriter1, rewriter2)");
    assert!(fused.rules.is_empty());
    assert_eq!(fused.sub_rewriters.len(), 2);
    assert_eq!(fused.sub_rewriters[0].name, "rewriter1");
    assert_eq!(fused.sub_rewriters[1].name, "rewriter2");

    let mut all_rules = Vec::new();
    fused.get_all_rules(&mut all_rules);
    assert_eq!(all_rules.len(), 2);
}