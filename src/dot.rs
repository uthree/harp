//! DOT形式でグラフを表現するためのトレイト。
pub trait ToDot {
    /// DOT形式のグラフ表現を文字列として返す。
    fn to_dot(&self) -> String;
}
