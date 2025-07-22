pub use crate::dtype::{DType, Number};
use crate::dot::ToDot;
use std::collections::HashSet;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::rc::Rc;

// operator types
#[derive(Clone, PartialEq, Debug)]
pub enum Op {
    Add,
    Mul,
    Recip,
    Rem,
    Load,
    Store,
    Cast(DType),
    Const(Number),
    Var(String),
    Exp2,
    Log2,
    Sin,
    Sqrt,
    Capture(usize), // Marker for pattern matching

    // Controll flow
    Loop,
    Block,
    If,
}

// internal data of UOp
#[derive(Clone, PartialEq, Debug)]
pub struct UOp_ {
    pub op: Op,
    pub dtype: DType,
    pub src: Vec<UOp>,
}

// micro operator
#[derive(Clone, PartialEq, Debug)]
pub struct UOp(pub Rc<UOp_>);

impl UOp {
    pub fn new(op: Op, dtype: DType, src: Vec<UOp>) -> Self {
        UOp(Rc::new(UOp_ { op, dtype, src }))
    }

    pub fn var(name: &str, dtype: DType) -> Self {
        UOp::new(Op::Var(name.to_string()), dtype, vec![])
    }

    // --- Unary Operations ---
    pub fn recip(self) -> Self {
        let dtype = self.0.dtype.clone();
        UOp::new(Op::Recip, dtype, vec![self])
    }

    pub fn cast(self, dtype: DType) -> Self {
        UOp::new(Op::Cast(dtype.clone()), dtype, vec![self])
    }

    pub fn exp2(self) -> Self {
        let dtype = self.0.dtype.clone();
        UOp::new(Op::Exp2, dtype, vec![self])
    }

    pub fn log2(self) -> Self {
        let dtype = self.0.dtype.clone();
        UOp::new(Op::Log2, dtype, vec![self])
    }

    pub fn sin(self) -> Self {
        let dtype = self.0.dtype.clone();
        UOp::new(Op::Sin, dtype, vec![self])
    }

    pub fn sqrt(self) -> Self {
        let dtype = self.0.dtype.clone();
        UOp::new(Op::Sqrt, dtype, vec![self])
    }
}

macro_rules! impl_from_for_uop {
    ($($t:ty => ($variant:ident, $dtype:ident)),*) => {
        $(
            impl From<$t> for UOp {
                fn from(n: $t) -> Self {
                    UOp::new(
                        Op::Const(Number::$variant(n)),
                        DType::$dtype,
                        vec![],
                    )
                }
            }
        )*
    };
}

impl_from_for_uop! {
    u8 => (U8, U8), u16 => (U16, U16), u32 => (U32, U32), u64 => (U64, U64),
    i8 => (I8, I8), i16 => (I16, I16), i32 => (I32, I32), i64 => (I64, I64),
    f32 => (F32, F32), f64 => (F64, F64)
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:ident) => {
        // --- UOp op UOp ---
        impl $trait<UOp> for UOp {
            type Output = UOp;
            fn $method(self, rhs: UOp) -> Self::Output {
                (&self).$method(&rhs)
            }
        }

        // --- UOp op &UOp ---
        impl $trait<&UOp> for UOp {
            type Output = UOp;
            fn $method(self, rhs: &UOp) -> Self::Output {
                (&self).$method(rhs)
            }
        }

        // --- &UOp op UOp ---
        impl $trait<UOp> for &UOp {
            type Output = UOp;
            fn $method(self, rhs: UOp) -> Self::Output {
                self.$method(&rhs)
            }
        }

        // --- &UOp op &UOp ---
        impl $trait<&UOp> for &UOp {
            type Output = UOp;
            fn $method(self, rhs: &UOp) -> Self::Output {
                // TODO: Implement proper dtype promotion
                let dtype = self.0.dtype.clone();
                UOp::new(Op::$op, dtype, vec![self.clone(), rhs.clone()])
            }
        }
    };
}

impl_binary_op!(Add, add, Add);
impl_binary_op!(Mul, mul, Mul);
impl_binary_op!(Rem, rem, Rem);

impl Sub for UOp {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        &self - &rhs
    }
}

impl Sub<&UOp> for &UOp {
    type Output = UOp;
    fn sub(self, rhs: &UOp) -> UOp {
        let neg_one: UOp = match &self.0.dtype {
            DType::I8 => (-1i8).into(),
            DType::I16 => (-1i16).into(),
            DType::I32 => (-1i32).into(),
            DType::I64 => (-1i64).into(),
            DType::F32 => (-1.0f32).into(),
            DType::F64 => (-1.0f64).into(),
            dtype => unimplemented!("Subtraction is not implemented for dtype {:?}", dtype),
        };
        self + &(rhs * neg_one)
    }
}

impl Div for UOp {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        &self / &rhs
    }
}

impl Div<&UOp> for &UOp {
    type Output = UOp;
    fn div(self, rhs: &UOp) -> UOp {
        self * &rhs.clone().recip()
    }
}

impl Neg for UOp {
    type Output = Self;
    fn neg(self) -> Self::Output {
        &self * &(-1i32).into()
    }
}

impl Neg for &UOp {
    type Output = UOp;
    fn neg(self) -> Self::Output {
        let neg_one: UOp = match &self.0.dtype {
            DType::I8 => (-1i8).into(),
            DType::I16 => (-1i16).into(),
            DType::I32 => (-1i32).into(),
            DType::I64 => (-1i64).into(),
            DType::F32 => (-1.0f32).into(),
            DType::F64 => (-1.0f64).into(),
            dtype => unimplemented!("Negation is not implemented for dtype {:?}", dtype),
        };
        self * &neg_one
    }
}

impl ToDot for UOp {
    fn to_dot(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph G {\n");
        dot.push_str("  node [shape=box];\n");
        let mut visited = HashSet::new();
        build_dot_uop(self, &mut dot, &mut visited);
        dot.push_str("}\n");
        dot
    }
}

fn build_dot_uop(uop: &UOp, dot: &mut String, visited: &mut HashSet<*const UOp_>) {
    let ptr = Rc::as_ptr(&uop.0);
    if visited.contains(&ptr) {
        return;
    }
    visited.insert(ptr);

    let label = match &uop.0.op {
        Op::Const(n) => format!("const {}\n{:?}", n, uop.0.dtype),
        Op::Var(name) => format!("var {}\n{:?}", name, uop.0.dtype),
        op => format!("{:?}\n{:?}", op, uop.0.dtype),
    }
    .replace('\n', "\\n");
    dot.push_str(&format!("  \"{ptr:p}\" [label=\"{label}\"];\n"));

    for src in &uop.0.src {
        let src_ptr = Rc::as_ptr(&src.0);
        dot.push_str(&format!("  \"{src_ptr:p}\" -> \"{ptr:p}\";\n"));
        build_dot_uop(src, dot, visited);
    }
}

macro_rules! impl_assign_op {
    ($trait:ident, $method:ident, $op_trait:ident, $op_method:ident) => {
        impl $trait<UOp> for UOp {
            fn $method(&mut self, rhs: UOp) {
                *self = (&*self).$op_method(&rhs);
            }
        }

        impl $trait<&UOp> for UOp {
            fn $method(&mut self, rhs: &UOp) {
                *self = (&*self).$op_method(rhs);
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign, Add, add);
impl_assign_op!(MulAssign, mul_assign, Mul, mul);
impl_assign_op!(SubAssign, sub_assign, Sub, sub);
impl_assign_op!(DivAssign, div_assign, Div, div);
impl_assign_op!(RemAssign, rem_assign, Rem, rem);

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(DType::U8, DType::U16)]
    #[case(DType::I32, DType::F32)]
    fn test_dtype_comparison(#[case] a: DType, #[case] b: DType) {
        assert_eq!(a, a.clone());
        assert_ne!(a, b);
    }

    #[rstest]
    #[case(5i32.into(), Op::Const(Number::I32(5)), DType::I32)]
    #[case(3.14f64.into(), Op::Const(Number::F64(3.14)), DType::F64)]
    #[case(255u8.into(), Op::Const(Number::U8(255)), DType::U8)]
    fn test_uop_from_numeric(#[case] uop: UOp, #[case] expected_op: Op, #[case] expected_dtype: DType) {
        assert_eq!(uop.0.op, expected_op);
        assert_eq!(uop.0.dtype, expected_dtype);
        assert!(uop.0.src.is_empty());
    }

    #[test]
    fn test_variable_creation() {
        let var_n = UOp::var("N", DType::U64);
        assert_eq!(var_n.0.op, Op::Var("N".to_string()));
        assert_eq!(var_n.0.dtype, DType::U64);
        assert!(var_n.0.src.is_empty());
    }

    #[test]
    fn test_binary_operations() {
        let a: UOp = 5i32.into();
        let b: UOp = 10i32.into();

        let c = &a + &b;
        assert_eq!(c.0.op, Op::Add);
        assert_eq!(c.0.src[0], a);
        assert_eq!(c.0.src[1], b);

        let d = &a * &b;
        assert_eq!(d.0.op, Op::Mul);
    }

    #[test]
    fn test_assign_operations() {
        let b: UOp = 10i32.into();
        let mut a: UOp = 5i32.into();
        let expected = a.clone() + &b;
        a += &b;
        assert_eq!(a, expected);
    }

    #[test]
    fn test_cast_operation() {
        let a: UOp = 5i32.into();
        let b = a.clone().cast(DType::F64);
        assert_eq!(b.0.op, Op::Cast(DType::F64));
        assert_eq!(b.0.dtype, DType::F64);
        assert_eq!(b.0.src[0], a);
    }

    #[rstest]
    #[case(UOp::from(5i32).exp2(), Op::Exp2)]
    #[case(UOp::from(5i32).log2(), Op::Log2)]
    #[case(UOp::from(5i32).sin(), Op::Sin)]
    #[case(UOp::from(5i32).sqrt(), Op::Sqrt)]
    #[case(UOp::from(5i32).recip(), Op::Recip)]
    fn test_unary_operations(#[case] uop: UOp, #[case] expected_op: Op) {
        assert_eq!(uop.0.op, expected_op);
        assert_eq!(uop.0.src.len(), 1);
    }

    #[test]
    fn test_to_dot() {
        let a = UOp::var("a", DType::F32);
        let b = UOp::from(1.0f32);
        let c = &a + &b;
        let dot_string = c.to_dot();
        println!("\n--- UOp DOT --- \n{}", dot_string);
        assert!(dot_string.starts_with("digraph G"));
        assert!(dot_string.contains("[label=\"var a\\nF32\"]"));
        assert!(dot_string.contains("[label=\"const 1f\\nF32\"]"));
        assert!(dot_string.contains("[label=\"Add\\nF32\"]"));
        assert!(dot_string.contains("->"));
    }
}
