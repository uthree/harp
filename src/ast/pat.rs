use crate::ast::AstNode;

type RewriterFn = Box<dyn Fn(&[AstNode]) -> AstNode>;
type ConditionFn = Box<dyn Fn(&[AstNode]) -> bool>;

pub struct AstRewriteRule {
    pattern: AstNode,
    rewriter: RewriterFn,
    condition: ConditionFn,
}
