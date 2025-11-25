use harp::DType;
use std::rc::Rc;

/// 統一中間表現（Unified Operation）
/// 高レベル演算から低レベル演算までを単一のグラフ構造で表現
#[derive(Debug, Clone, PartialEq)]
pub enum UOp {
    // ========== 高レベル演算（テンソルレベル） ==========
    /// テンソル入力
    Input {
        dtype: DType,
        name: String,
        shape: Vec<usize>,
    },

    /// 定数値
    Const { dtype: DType, value: f64 },

    /// Element-wise演算
    Elementwise {
        dtype: DType,
        op: ElementwiseOp,
        inputs: Vec<Rc<UOp>>,
    },

    /// Reduce演算
    Reduce {
        dtype: DType,
        op: ReduceOp,
        input: Rc<UOp>,
        axis: usize,
        input_shape: Vec<usize>,
    },

    // ========== 中レベル演算（ループレベル） ==========
    /// ループ構造
    Loop {
        dtype: DType,
        var: String,
        start: usize,
        end: usize,
        parallel: bool,
        body: Rc<UOp>,
    },

    /// メモリロード
    Load {
        dtype: DType,
        buffer: String,
        index: Option<Rc<UOp>>,
    },

    /// メモリストア
    Store {
        dtype: DType,
        buffer: String,
        index: Option<Rc<UOp>>,
        value: Rc<UOp>,
    },

    /// バリア同期
    Barrier { dtype: DType },

    /// シーケンス（複数の操作を順次実行）
    Sequence { dtype: DType, ops: Vec<Rc<UOp>> },

    // ========== 低レベル演算（スカラーレベル） ==========
    /// スレッドインデックス（GPUの場合）
    ThreadIdx { dtype: DType, dim: usize },

    /// グループインデックス（GPUの場合）
    GroupIdx { dtype: DType, dim: usize },

    /// 変数参照
    Var { dtype: DType, name: String },

    /// スカラー加算
    Add {
        dtype: DType,
        lhs: Rc<UOp>,
        rhs: Rc<UOp>,
    },

    /// スカラー乗算
    Mul {
        dtype: DType,
        lhs: Rc<UOp>,
        rhs: Rc<UOp>,
    },

    /// スカラー最大値
    Max {
        dtype: DType,
        lhs: Rc<UOp>,
        rhs: Rc<UOp>,
    },

    /// スカラー剰余
    Rem {
        dtype: DType,
        lhs: Rc<UOp>,
        rhs: Rc<UOp>,
    },

    /// スカラー整数除算
    Idiv {
        dtype: DType,
        lhs: Rc<UOp>,
        rhs: Rc<UOp>,
    },

    /// スカラー逆数
    Recip { dtype: DType, arg: Rc<UOp> },

    /// スカラー平方根
    Sqrt { dtype: DType, arg: Rc<UOp> },

    /// スカラー比較（小なり）
    LessThan {
        dtype: DType,
        lhs: Rc<UOp>,
        rhs: Rc<UOp>,
    },

    /// 三項演算子 (cond ? then : else_)
    Select {
        dtype: DType,
        cond: Rc<UOp>,
        then_: Rc<UOp>,
        else_: Rc<UOp>,
    },

    // ========== パターンマッチング用 ==========
    /// ワイルドカード（パターンマッチング用）
    Wildcard { dtype: DType, id: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementwiseOp {
    // Unary
    Neg,
    Recip,
    Sqrt,
    Exp,
    Log,

    // Binary
    Add,
    Mul,
    Max,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
}

impl UOp {
    /// dtypeを取得
    pub fn dtype(&self) -> &DType {
        match self {
            UOp::Input { dtype, .. }
            | UOp::Const { dtype, .. }
            | UOp::Elementwise { dtype, .. }
            | UOp::Reduce { dtype, .. }
            | UOp::Loop { dtype, .. }
            | UOp::Load { dtype, .. }
            | UOp::Store { dtype, .. }
            | UOp::Barrier { dtype }
            | UOp::Sequence { dtype, .. }
            | UOp::ThreadIdx { dtype, .. }
            | UOp::GroupIdx { dtype, .. }
            | UOp::Var { dtype, .. }
            | UOp::Add { dtype, .. }
            | UOp::Mul { dtype, .. }
            | UOp::Max { dtype, .. }
            | UOp::Rem { dtype, .. }
            | UOp::Idiv { dtype, .. }
            | UOp::Recip { dtype, .. }
            | UOp::Sqrt { dtype, .. }
            | UOp::LessThan { dtype, .. }
            | UOp::Select { dtype, .. }
            | UOp::Wildcard { dtype, .. } => dtype,
        }
    }

    /// 子ノードを取得（汎用的な走査用）
    pub fn children(&self) -> Vec<&Rc<UOp>> {
        match self {
            UOp::Input { .. }
            | UOp::Const { .. }
            | UOp::ThreadIdx { .. }
            | UOp::GroupIdx { .. }
            | UOp::Var { .. }
            | UOp::Barrier { .. }
            | UOp::Wildcard { .. } => vec![],

            UOp::Elementwise { inputs, .. } => inputs.iter().collect(),
            UOp::Reduce { input, .. } => vec![input],
            UOp::Loop { body, .. } => vec![body],
            UOp::Load { index, .. } => index.iter().collect(),
            UOp::Store { index, value, .. } => {
                let mut v: Vec<&Rc<UOp>> = index.iter().collect();
                v.push(value);
                v
            }
            UOp::Sequence { ops, .. } => ops.iter().collect(),
            UOp::Add { lhs, rhs, .. }
            | UOp::Mul { lhs, rhs, .. }
            | UOp::Max { lhs, rhs, .. }
            | UOp::Rem { lhs, rhs, .. }
            | UOp::Idiv { lhs, rhs, .. }
            | UOp::LessThan { lhs, rhs, .. } => vec![lhs, rhs],
            UOp::Recip { arg, .. } | UOp::Sqrt { arg, .. } => vec![arg],
            UOp::Select {
                cond, then_, else_, ..
            } => vec![cond, then_, else_],
        }
    }

    /// UOpのグラフを可視化用に文字列化
    pub fn to_debug_string(&self, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        let dtype_str = format!("{:?}", self.dtype());

        match self {
            UOp::Input { name, shape, .. } => {
                format!("{}Input({:?}, {:?}):{}", prefix, name, shape, dtype_str)
            }
            UOp::Const { value, .. } => {
                format!("{}Const({}):{}", prefix, value, dtype_str)
            }
            UOp::Var { name, .. } => {
                format!("{}Var({:?}):{}", prefix, name, dtype_str)
            }
            UOp::ThreadIdx { dim, .. } => {
                format!("{}ThreadIdx({}):{}", prefix, dim, dtype_str)
            }
            UOp::GroupIdx { dim, .. } => {
                format!("{}GroupIdx({}):{}", prefix, dim, dtype_str)
            }
            UOp::Wildcard { id, .. } => {
                format!("{}Wildcard({}):{}", prefix, id, dtype_str)
            }
            UOp::Barrier { .. } => {
                format!("{}Barrier:{}", prefix, dtype_str)
            }
            _ => {
                let op_name = match self {
                    UOp::Elementwise { op, .. } => format!("Elementwise({:?})", op),
                    UOp::Reduce { op, axis, .. } => format!("Reduce({:?}, axis={})", op, axis),
                    UOp::Loop {
                        var,
                        start,
                        end,
                        parallel,
                        ..
                    } => format!("Loop({}, {}..{}, parallel={})", var, start, end, parallel),
                    UOp::Load { buffer, .. } => format!("Load({})", buffer),
                    UOp::Store { buffer, .. } => format!("Store({})", buffer),
                    UOp::Sequence { .. } => "Sequence".to_string(),
                    UOp::Add { .. } => "Add".to_string(),
                    UOp::Mul { .. } => "Mul".to_string(),
                    UOp::Max { .. } => "Max".to_string(),
                    UOp::Rem { .. } => "Rem".to_string(),
                    UOp::Idiv { .. } => "Idiv".to_string(),
                    UOp::Recip { .. } => "Recip".to_string(),
                    UOp::Sqrt { .. } => "Sqrt".to_string(),
                    UOp::LessThan { .. } => "LessThan".to_string(),
                    UOp::Select { .. } => "Select".to_string(),
                    _ => "Unknown".to_string(),
                };

                let children = self.children();
                if children.is_empty() {
                    format!("{}{}:{}", prefix, op_name, dtype_str)
                } else {
                    let mut result = format!("{}{}:{}(\n", prefix, op_name, dtype_str);
                    for child in children {
                        result.push_str(&child.to_debug_string(indent + 1));
                        result.push('\n');
                    }
                    result.push_str(&format!("{})", prefix));
                    result
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helper::*;

    #[test]
    fn test_basic_uop() {
        let a = input("a", vec![10], DType::F32);
        let b = input("b", vec![10], DType::F32);
        let c = add(a, b);

        println!("{}", c.to_debug_string(0));

        match &*c {
            UOp::Add { .. } => {}
            _ => panic!("Expected Add"),
        }
        assert_eq!(c.children().len(), 2);
    }

    #[test]
    fn test_reduce() {
        let a = input("a", vec![10, 20], DType::F32);
        let sum = reduce(ReduceOp::Sum, a, 1, vec![10, 20]);

        println!("{}", sum.to_debug_string(0));

        match &*sum {
            UOp::Reduce { op, axis, .. } => {
                assert_eq!(*op, ReduceOp::Sum);
                assert_eq!(*axis, 1);
            }
            _ => panic!("Expected Reduce"),
        }
    }

    #[test]
    fn test_dtype_accessor() {
        let a = input("a", vec![10], DType::F32);
        assert_eq!(*a.dtype(), DType::F32);

        let b = const_val(42.0, DType::Unknown);
        assert_eq!(*b.dtype(), DType::Unknown);
    }

    #[test]
    fn test_children() {
        let a = const_val(1.0, DType::F32);
        let b = const_val(2.0, DType::F32);
        let c = add(a, b);

        assert_eq!(c.children().len(), 2);

        let inp = input("x", vec![10], DType::F32);
        assert_eq!(inp.children().len(), 0);
    }
}
