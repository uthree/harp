use super::symbolic::Expr;

pub trait ToShape {
    fn to_shape(self) -> Vec<Expr>;
}

macro_rules! impl_to_shape_for_tuple {
    ($($t:ident),*) => {
        impl<$($t: Into<Expr>),*> ToShape for ($($t,)*) {
            #[inline]
            fn to_shape(self) -> Vec<Expr> {
                #[allow(non_snake_case)]
                let ($($t,)*) = self;
                vec![$($t.into()),*]
            }
        }
    };
}

impl_to_shape_for_tuple!();
impl_to_shape_for_tuple!(T1);
impl_to_shape_for_tuple!(T1, T2);
impl_to_shape_for_tuple!(T1, T2, T3);
impl_to_shape_for_tuple!(T1, T2, T3, T4);
impl_to_shape_for_tuple!(T1, T2, T3, T4, T5);
impl_to_shape_for_tuple!(T1, T2, T3, T4, T5, T6);
impl_to_shape_for_tuple!(T1, T2, T3, T4, T5, T6, T7);
impl_to_shape_for_tuple!(T1, T2, T3, T4, T5, T6, T7, T8);
