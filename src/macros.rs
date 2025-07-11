/// Creates a `Rewriter` from a list of declarative rules.
///
/// # Example
///
/// ```
/// use harp::node::{self, Node};
/// use harp::pattern::Rewriter;
/// use harp::rewriter;
///
/// let simple_rewriter = rewriter!("simple", [
///     (
///         let x = capture("x")
///         => x + Node::from(0.0f32)
///         => |node, x| Some(x)
///     )
/// ]);
/// ```
#[macro_export]
macro_rules! rewriter {
    // New rule syntax: allows capturing the matched node itself.
    // Example: `(let x = capture("x") => |node, x| { ... })`
    ($name:expr, [$(
        (
            let $($var:ident = capture($name_str:literal)),*
            => $searcher:expr
            => |$node:ident, $($arg:ident),*| $rewriter_body:expr
        )
    ),* $(,)?]) => {
        {
            let rules = vec![
                $(
                    $crate::pattern::RewriteRule::new_fn(
                        {
                            $(let $var = $crate::node::capture($name_str);)*
                            $searcher
                        },
                        |$node, captures| {
                            $(
                                let $arg = captures.get(stringify!($arg))?.clone();
                            )*
                            $rewriter_body
                        }
                    )
                ),*
            ];
            $crate::pattern::Rewriter::new($name, rules)
        }
    };

    // Original rule syntax: for backward compatibility.
    // Example: `(let x = capture("x") => |x| { ... })`
    ($name:expr, [$(
        (
            let $($var:ident = capture($name_str:literal)),*
            => $searcher:expr
            => |$($arg:ident),*| $rewriter_body:expr
        )
    ),* $(,)?]) => {
        {
            let rules = vec![
                $(
                    $crate::pattern::RewriteRule::new_fn(
                        {
                            $(let $var = $crate::node::capture($name_str);)*
                            $searcher
                        },
                        |_node, captures| {
                            $(
                                let $arg = captures.get(stringify!($arg))?.clone();
                            )*
                            $rewriter_body
                        }
                    )
                ),*
            ];
            $crate::pattern::Rewriter::new($name, rules)
        }
    };
}

