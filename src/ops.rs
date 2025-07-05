pub trait Operator {}

pub trait ElementwiseUnaryOperator {}
pub trait ElementwiseBinaryOperator {}

pub struct Input {}
impl Operator for Input {}

pub struct Add {}
impl Operator for Add {}
impl ElementwiseBinaryOperator for Add {}

pub struct Mul {}
impl Operator for Mul {}
impl ElementwiseBinaryOperator for Mul {}

pub struct Recip {}
impl Operator for Recip {}
impl ElementwiseUnaryOperator for Recip {}

pub struct Rem {}
impl Operator for Rem {}
impl ElementwiseBinaryOperator for Rem {}

pub struct Sin {}
impl Operator for Sin {}
impl ElementwiseUnaryOperator for Sin {}

pub struct Exp2 {}
impl Operator for Exp2 {}
impl ElementwiseUnaryOperator for Exp2 {}

pub struct Log2 {}
impl Operator for Log2 {}
impl ElementwiseUnaryOperator for Log2 {}

pub struct Sqrt {}
impl Operator for Sqrt {}
impl ElementwiseUnaryOperator for Sqrt {}
