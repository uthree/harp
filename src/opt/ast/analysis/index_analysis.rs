use crate::ast::{AstNode, ConstLiteral};

/// Pattern of how an index depends on a loop variable
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexPattern {
    /// Index is exactly the loop variable: i
    Identity,
    /// Index is loop variable plus constant: i + c
    Offset(isize),
    /// Index is loop variable times constant: i * c
    Scaled(isize),
    /// Index is scaled and offset: i * a + b
    ScaledOffset { scale: isize, offset: isize },
    /// Index is a constant value (independent of loop variable)
    Constant(isize),
    /// Complex expression that doesn't match simple patterns
    Complex,
}

/// Analyze how an index expression depends on a loop variable
///
/// This function determines whether an index expression follows a simple
/// pattern with respect to a loop variable. This is useful for determining
/// whether different iterations of a loop access different memory locations.
///
/// # Arguments
/// * `index` - The index expression to analyze
/// * `loop_var` - The loop variable name
///
/// # Returns
/// The pattern of dependency, or `IndexPattern::Complex` if the expression
/// is too complex to analyze
///
/// # Examples
/// ```
/// // i -> Identity
/// // i + 5 -> Offset(5)
/// // i * 2 -> Scaled(2)
/// // i * 2 + 3 -> ScaledOffset { scale: 2, offset: 3 }
/// // 42 -> Constant(42)
/// // i + j -> Complex
/// ```
pub fn analyze_index_pattern(index: &AstNode, loop_var: &str) -> IndexPattern {
    match index {
        // Simple variable reference
        AstNode::Var(name) => {
            if name == loop_var {
                IndexPattern::Identity
            } else {
                // Variable other than loop variable - treat as complex
                IndexPattern::Complex
            }
        }

        // Constant value
        AstNode::Const(lit) => {
            if let Some(val) = extract_integer_literal(lit) {
                IndexPattern::Constant(val)
            } else {
                IndexPattern::Complex
            }
        }

        // Addition: could be i + c or more complex
        AstNode::Add(left, right) => {
            let left_pattern = analyze_index_pattern(left, loop_var);
            let right_pattern = analyze_index_pattern(right, loop_var);

            match (left_pattern, right_pattern) {
                // i + c
                (IndexPattern::Identity, IndexPattern::Constant(c))
                | (IndexPattern::Constant(c), IndexPattern::Identity) => IndexPattern::Offset(c),

                // (i * a) + b
                (IndexPattern::Scaled(a), IndexPattern::Constant(b))
                | (IndexPattern::Constant(b), IndexPattern::Scaled(a)) => {
                    IndexPattern::ScaledOffset {
                        scale: a,
                        offset: b,
                    }
                }

                // (i + a) + b = i + (a + b)
                (IndexPattern::Offset(a), IndexPattern::Constant(b))
                | (IndexPattern::Constant(b), IndexPattern::Offset(a)) => {
                    IndexPattern::Offset(a + b)
                }

                // (i * a + b) + c = i * a + (b + c)
                (
                    IndexPattern::ScaledOffset {
                        scale: a,
                        offset: b,
                    },
                    IndexPattern::Constant(c),
                )
                | (
                    IndexPattern::Constant(c),
                    IndexPattern::ScaledOffset {
                        scale: a,
                        offset: b,
                    },
                ) => IndexPattern::ScaledOffset {
                    scale: a,
                    offset: b + c,
                },

                // c1 + c2 (both constants)
                (IndexPattern::Constant(c1), IndexPattern::Constant(c2)) => {
                    IndexPattern::Constant(c1 + c2)
                }

                // Anything else is too complex
                _ => IndexPattern::Complex,
            }
        }

        // Multiplication: could be i * c or more complex
        AstNode::Mul(left, right) => {
            let left_pattern = analyze_index_pattern(left, loop_var);
            let right_pattern = analyze_index_pattern(right, loop_var);

            match (left_pattern, right_pattern) {
                // i * c
                (IndexPattern::Identity, IndexPattern::Constant(c))
                | (IndexPattern::Constant(c), IndexPattern::Identity) => IndexPattern::Scaled(c),

                // (i + a) * b = i * b + a * b
                (IndexPattern::Offset(a), IndexPattern::Constant(b))
                | (IndexPattern::Constant(b), IndexPattern::Offset(a)) => {
                    IndexPattern::ScaledOffset {
                        scale: b,
                        offset: a * b,
                    }
                }

                // c1 * c2 (both constants)
                (IndexPattern::Constant(c1), IndexPattern::Constant(c2)) => {
                    IndexPattern::Constant(c1 * c2)
                }

                // Anything else is too complex
                _ => IndexPattern::Complex,
            }
        }

        // Cast: analyze the inner expression
        AstNode::Cast { dtype: _, expr } => analyze_index_pattern(expr, loop_var),

        // Anything else is complex
        _ => IndexPattern::Complex,
    }
}

/// Extract an integer value from a constant literal
fn extract_integer_literal(lit: &ConstLiteral) -> Option<isize> {
    match lit {
        ConstLiteral::Isize(v) => Some(*v),
        ConstLiteral::Usize(v) => Some(*v as isize),
        _ => None,
    }
}

/// Check if two index patterns could access the same memory location
/// for different values of the loop variable **within the same iteration**.
///
/// Returns `true` if the patterns are guaranteed to access different locations
/// for the same loop iteration value, `false` if they might access the same location.
///
/// **Note**: This is NOT sufficient for parallelization safety! Use
/// `are_patterns_disjoint_for_parallelization` for that purpose.
///
/// # Examples
/// ```
/// // i and i -> same location (not disjoint)
/// // i and i+1 -> different locations (disjoint)
/// // i and j -> unknown (complex, assume not disjoint)
/// ```
pub fn are_patterns_disjoint(p1: &IndexPattern, p2: &IndexPattern) -> bool {
    match (p1, p2) {
        // Same pattern with same parameters -> not disjoint (might overlap)
        (IndexPattern::Identity, IndexPattern::Identity) => false,
        (IndexPattern::Offset(o1), IndexPattern::Offset(o2)) => o1 != o2,
        (IndexPattern::Scaled(s1), IndexPattern::Scaled(s2)) => s1 != s2 || *s1 == 0,
        (
            IndexPattern::ScaledOffset {
                scale: s1,
                offset: o1,
            },
            IndexPattern::ScaledOffset {
                scale: s2,
                offset: o2,
            },
        ) => s1 != s2 || o1 != o2,

        // Constant patterns: disjoint if different values
        (IndexPattern::Constant(c1), IndexPattern::Constant(c2)) => c1 != c2,

        // Identity vs Offset: always disjoint if offset != 0
        (IndexPattern::Identity, IndexPattern::Offset(o))
        | (IndexPattern::Offset(o), IndexPattern::Identity) => *o != 0,

        // Complex patterns: conservatively assume they might overlap
        (IndexPattern::Complex, _) | (_, IndexPattern::Complex) => false,

        // Different pattern types: conservatively assume might overlap
        _ => false,
    }
}

/// Check if two index patterns will NEVER access the same memory location
/// across **any** two different iterations (for parallelization safety).
///
/// This is more strict than `are_patterns_disjoint`. For parallelization, we need
/// to ensure that no two iterations (with different loop variable values) can
/// access the same memory location.
///
/// Returns `true` only if we can prove that iterations will never conflict.
///
/// # Examples
/// ```
/// // i and i -> SAFE (each iteration has its own i value)
/// // i and i-1 -> UNSAFE (iteration i+1's i-1 equals iteration i's i)
/// // i*2 and i*2+1 -> SAFE (always access different parities)
/// // i and constant -> UNSAFE (all iterations might access the constant location)
/// ```
pub fn are_patterns_disjoint_for_parallelization(p1: &IndexPattern, p2: &IndexPattern) -> bool {
    match (p1, p2) {
        // If both patterns are identical, each iteration accesses its own unique location - SAFE
        _ if p1 == p2 => true,

        // Scaled patterns with different scales that don't divide each other
        // e.g., i*2 and i*3 never overlap
        (IndexPattern::Scaled(s1), IndexPattern::Scaled(s2)) => {
            // If scales are coprime (gcd == 1), they never overlap
            s1 != s2 && gcd(s1.abs(), s2.abs()) == 1
        }

        // Scaled offset patterns - need to check if they can ever be equal
        // i*a+b == i*c+d for different i values?
        // This is complex, so conservatively return false
        (IndexPattern::ScaledOffset { .. }, IndexPattern::ScaledOffset { .. }) => false,

        // i*a vs i*a+b: can overlap if b is a multiple of a
        (IndexPattern::Scaled(a), IndexPattern::ScaledOffset { scale, offset })
        | (IndexPattern::ScaledOffset { scale, offset }, IndexPattern::Scaled(a)) => {
            a != scale || offset % a != 0
        }

        // Identity vs Offset: iteration i+offset writes what iteration i reads - UNSAFE
        (IndexPattern::Identity, IndexPattern::Offset(_))
        | (IndexPattern::Offset(_), IndexPattern::Identity) => false,

        // Two different offsets: i+a vs i+b - each iteration has unique value - SAFE if a==b
        (IndexPattern::Offset(o1), IndexPattern::Offset(o2)) => o1 == o2,

        // Constant vs anything with loop variable dependency: UNSAFE
        // All iterations might access the constant location
        (IndexPattern::Constant(_), IndexPattern::Identity)
        | (IndexPattern::Identity, IndexPattern::Constant(_))
        | (IndexPattern::Constant(_), IndexPattern::Offset(_))
        | (IndexPattern::Offset(_), IndexPattern::Constant(_))
        | (IndexPattern::Constant(_), IndexPattern::Scaled(_))
        | (IndexPattern::Scaled(_), IndexPattern::Constant(_))
        | (IndexPattern::Constant(_), IndexPattern::ScaledOffset { .. })
        | (IndexPattern::ScaledOffset { .. }, IndexPattern::Constant(_)) => false,

        // Two constants: safe if they're the same (no conflict) or different (disjoint)
        (IndexPattern::Constant(c1), IndexPattern::Constant(c2)) => c1 == c2,

        // Complex patterns: conservatively assume unsafe
        (IndexPattern::Complex, _) | (_, IndexPattern::Complex) => false,

        // Any other combination: conservatively assume unsafe
        _ => false,
    }
}

/// Greatest common divisor (for checking if scaled patterns can overlap)
fn gcd(a: isize, b: isize) -> isize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;

    #[test]
    fn test_identity_pattern() {
        let index = var("i");
        let pattern = analyze_index_pattern(&index, "i");
        assert_eq!(pattern, IndexPattern::Identity);
    }

    #[test]
    fn test_constant_pattern() {
        let index = const_val(ConstLiteral::Usize(42));
        let pattern = analyze_index_pattern(&index, "i");
        assert_eq!(pattern, IndexPattern::Constant(42));
    }

    #[test]
    fn test_offset_pattern() {
        // i + 5
        let index = add(var("i"), const_val(ConstLiteral::Isize(5)));
        let pattern = analyze_index_pattern(&index, "i");
        assert_eq!(pattern, IndexPattern::Offset(5));

        // 5 + i (commutative)
        let index = add(const_val(ConstLiteral::Isize(5)), var("i"));
        let pattern = analyze_index_pattern(&index, "i");
        assert_eq!(pattern, IndexPattern::Offset(5));
    }

    #[test]
    fn test_scaled_pattern() {
        // i * 2
        let index = mul(var("i"), const_val(ConstLiteral::Isize(2)));
        let pattern = analyze_index_pattern(&index, "i");
        assert_eq!(pattern, IndexPattern::Scaled(2));

        // 2 * i (commutative)
        let index = mul(const_val(ConstLiteral::Isize(2)), var("i"));
        let pattern = analyze_index_pattern(&index, "i");
        assert_eq!(pattern, IndexPattern::Scaled(2));
    }

    #[test]
    fn test_scaled_offset_pattern() {
        // i * 2 + 3
        let index = add(
            mul(var("i"), const_val(ConstLiteral::Isize(2))),
            const_val(ConstLiteral::Isize(3)),
        );
        let pattern = analyze_index_pattern(&index, "i");
        assert_eq!(
            pattern,
            IndexPattern::ScaledOffset {
                scale: 2,
                offset: 3
            }
        );
    }

    #[test]
    fn test_complex_pattern() {
        // i + j (two variables)
        let index = add(var("i"), var("j"));
        let pattern = analyze_index_pattern(&index, "i");
        assert_eq!(pattern, IndexPattern::Complex);
    }

    #[test]
    fn test_disjoint_offsets() {
        let p1 = IndexPattern::Offset(0);
        let p2 = IndexPattern::Offset(1);
        assert!(are_patterns_disjoint(&p1, &p2));

        let p3 = IndexPattern::Offset(0);
        let p4 = IndexPattern::Offset(0);
        assert!(!are_patterns_disjoint(&p3, &p4));
    }

    #[test]
    fn test_disjoint_identity() {
        let p1 = IndexPattern::Identity;
        let p2 = IndexPattern::Identity;
        // Same index pattern -> not disjoint
        assert!(!are_patterns_disjoint(&p1, &p2));

        let p3 = IndexPattern::Identity;
        let p4 = IndexPattern::Offset(1);
        // i vs i+1 -> disjoint
        assert!(are_patterns_disjoint(&p3, &p4));
    }

    #[test]
    fn test_disjoint_constants() {
        let p1 = IndexPattern::Constant(5);
        let p2 = IndexPattern::Constant(10);
        assert!(are_patterns_disjoint(&p1, &p2));

        let p3 = IndexPattern::Constant(5);
        let p4 = IndexPattern::Constant(5);
        assert!(!are_patterns_disjoint(&p3, &p4));
    }

    #[test]
    fn test_complex_not_disjoint() {
        let p1 = IndexPattern::Complex;
        let p2 = IndexPattern::Identity;
        // Complex patterns: conservatively assume overlap
        assert!(!are_patterns_disjoint(&p1, &p2));
    }
}
