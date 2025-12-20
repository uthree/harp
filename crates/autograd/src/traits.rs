use num_traits::One;

use crate::variable::Variable;

/// 逆伝播の中間点として使用可能な型
/// Clone を実装する全ての型に自動実装される
pub trait GradNode: Clone {}

/// 逆伝播の起点として使用可能な型
/// 初期勾配を生成するメソッドを持つ
pub trait GradRoot: GradNode {
    /// 初期勾配を生成（スカラーの場合は1.0）
    fn unit_grad() -> Self;
}

/// 勾配関数を表すトレイト (計算グラフのエッジ)
/// 逆伝播時に勾配を入力変数に伝播する
pub trait GradFn<GradType> {
    /// 出力側から受け取った勾配を入力側に伝播する
    fn backward(&mut self, grad_y: GradType);
}

/// 勾配変換トレイト
/// 出力の勾配から入力の勾配への変換を抽象化
pub trait GradInto<T> {
    fn gradient_into(self) -> T;
}

// ============================================================================
// ブランケット実装
// ============================================================================

/// Clone を実装する全ての型は GradNode を自動実装
impl<T: Clone> GradNode for T {}

/// Clone + One を実装する全ての型は GradRoot を自動実装
impl<T: Clone + One> GradRoot for T {
    fn unit_grad() -> Self {
        T::one()
    }
}

/// From を実装していれば自動で GradientInto も使える
impl<T, U> GradInto<Variable<U>> for Variable<T>
where
    T: Clone + 'static,
    U: From<T> + 'static,
{
    fn gradient_into(self) -> Variable<U> {
        Variable::new(U::from(self.value()))
    }
}
