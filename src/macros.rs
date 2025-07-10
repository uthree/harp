/// Creates a `Rewriter` from a list of declarative rules.
///
/// # Example
///
/// ```
/// use harp::node::{self, Node};
/// use harp::pattern::Rewriter;
/// use harp::rewriter;
///
/// let simple_rewriter = rewriter!([
///     (
///         let x = capture("x")
///         => x + Node::from(0.0f32)
///         => |x| Some(x)
///     )
/// ]);
/// ```
#[macro_export]
macro_rules! rewriter {
    ([$(
        (
            let $($var:ident = capture($name:literal)),*
            => $searcher:expr
            => |$($arg:ident),*| $rewriter_body:expr
        )
    ),* $(,)?]) => {
        {
            let rules = vec![
                $(
                    $crate::pattern::RewriteRule::new_fn(
                        {
                            // This creates the searcher pattern.
                            // The variables ($var) are bound here and used in the $searcher expression.
                            $(let $var = $crate::node::capture($name);)*
                            $searcher
                        },
                        |captures| {
                            // This creates the rewriter function.
                            // It extracts the captured nodes from the `captures` map
                            // and binds them to variables ($arg) for the user's rewriter body.
                            $(
                                let $arg = captures.get(stringify!($arg))?.clone();
                            )*
                            $rewriter_body
                        }
                    )
                ),*
            ];
            $crate::pattern::Rewriter::new(rules)
        }
    };
}
