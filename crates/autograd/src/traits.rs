/// 逆伝播の中間点として使用可能な型
/// スカラー型とテンソル型の両方に実装する
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

// ============================================================================
// 組み込み型に対するトレイト実装
// ============================================================================

impl GradNode for f32 {}
impl GradNode for f64 {}

impl GradRoot for f32 {
    fn unit_grad() -> Self {
        1.0
    }
}

impl GradRoot for f64 {
    fn unit_grad() -> Self {
        1.0
    }
}
