pub trait Operator {}

pub struct Input {}
impl Operator for Input {}

pub struct Add {}
impl Operator for Add {}

pub struct Mul {}
impl Operator for Mul {}

pub struct Recip {}
impl Operator for Recip {}

pub struct Rem {}
impl Operator for Rem {}

pub struct Sin {}
impl Operator for Sin {}

pub struct Exp2 {}
impl Operator for Exp2 {}

pub struct Log2 {}
impl Operator for Log2 {}

pub struct Sqrt {}
impl Operator for Sqrt {}
