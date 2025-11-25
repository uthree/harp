use harp::DType;
use std::rc::Rc;

/// 統一中間表現（Unified Operation）
/// 高レベル演算から低レベル演算までを単一のグラフ構造で表現
///
/// dtypeは葉ノード（Input, Const, Var, ThreadIdx, GroupIdx, Load）のみが保持し、
/// その他のノードは子ノードから型推論される。
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
        op: ElementwiseOp,
        inputs: Vec<Rc<UOp>>,
    },

    /// Reduce演算
    Reduce {
        op: ReduceOp,
        input: Rc<UOp>,
        axis: usize,
        input_shape: Vec<usize>,
    },

    // ========== 中レベル演算（ループレベル） ==========
    /// ループ構造
    Loop {
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
        buffer: String,
        index: Option<Rc<UOp>>,
        value: Rc<UOp>,
    },

    /// バリア同期
    Barrier,

    /// シーケンス（複数の操作を順次実行）
    Sequence(Vec<Rc<UOp>>),

    // ========== 低レベル演算（スカラーレベル） ==========
    /// スレッドインデックス（GPUの場合）
    ThreadIdx { dtype: DType, dim: usize },

    /// グループインデックス（GPUの場合）
    GroupIdx { dtype: DType, dim: usize },

    /// 変数参照
    Var { dtype: DType, name: String },

    /// スカラー加算
    Add(Rc<UOp>, Rc<UOp>),

    /// スカラー乗算
    Mul(Rc<UOp>, Rc<UOp>),

    /// スカラー最大値
    Max(Rc<UOp>, Rc<UOp>),

    /// スカラー剰余
    Rem(Rc<UOp>, Rc<UOp>),

    /// スカラー整数除算
    Idiv(Rc<UOp>, Rc<UOp>),

    /// スカラー逆数
    Recip(Rc<UOp>),

    /// スカラー平方根
    Sqrt(Rc<UOp>),

    /// スカラー比較（小なり）
    LessThan(Rc<UOp>, Rc<UOp>),

    /// 三項演算子 (cond ? then : else_)
    Select(Rc<UOp>, Rc<UOp>, Rc<UOp>),

    // ========== パターンマッチング用 ==========
    /// ワイルドカード（パターンマッチング用）
    Wildcard(usize),
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
    /// dtypeを型推論で取得
    pub fn dtype(&self) -> DType {
        match self {
            // 葉ノード: dtypeを保持
            UOp::Input { dtype, .. } => dtype.clone(),
            UOp::Const { dtype, .. } => dtype.clone(),
            UOp::Var { dtype, .. } => dtype.clone(),
            UOp::ThreadIdx { dtype, .. } => dtype.clone(),
            UOp::GroupIdx { dtype, .. } => dtype.clone(),
            UOp::Load { dtype, .. } => dtype.clone(),

            // 二項演算: 左辺の型を継承
            UOp::Add(lhs, _)
            | UOp::Mul(lhs, _)
            | UOp::Max(lhs, _)
            | UOp::Rem(lhs, _)
            | UOp::Idiv(lhs, _) => lhs.dtype(),

            // 比較演算: bool (Unknown で代用)
            UOp::LessThan(..) => DType::Unknown,

            // 単項演算: 引数の型を継承
            UOp::Recip(arg) | UOp::Sqrt(arg) => arg.dtype(),

            // Select: then_の型を継承
            UOp::Select(_, then_, _) => then_.dtype(),

            // Elementwise/Reduce: 入力の型を継承
            UOp::Elementwise { inputs, .. } => {
                inputs.first().map(|i| i.dtype()).unwrap_or(DType::Unknown)
            }
            UOp::Reduce { input, .. } => input.dtype(),

            // Loop: bodyの型を継承
            UOp::Loop { body, .. } => body.dtype(),

            // Store: valueの型を継承
            UOp::Store { value, .. } => value.dtype(),

            // Sequence: 最後の要素の型（空ならUnknown）
            UOp::Sequence(ops) => ops.last().map(|o| o.dtype()).unwrap_or(DType::Unknown),

            // Barrier: void相当
            UOp::Barrier => DType::Unknown,

            // Wildcard: パターンマッチング用、任意の型にマッチ
            UOp::Wildcard(_) => DType::Unknown,
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
            | UOp::Barrier
            | UOp::Wildcard(_) => vec![],

            UOp::Elementwise { inputs, .. } => inputs.iter().collect(),
            UOp::Reduce { input, .. } => vec![input],
            UOp::Loop { body, .. } => vec![body],
            UOp::Load { index, .. } => index.iter().collect(),
            UOp::Store { index, value, .. } => {
                let mut v: Vec<&Rc<UOp>> = index.iter().collect();
                v.push(value);
                v
            }
            UOp::Sequence(ops) => ops.iter().collect(),
            UOp::Add(lhs, rhs)
            | UOp::Mul(lhs, rhs)
            | UOp::Max(lhs, rhs)
            | UOp::Rem(lhs, rhs)
            | UOp::Idiv(lhs, rhs)
            | UOp::LessThan(lhs, rhs) => vec![lhs, rhs],
            UOp::Recip(arg) | UOp::Sqrt(arg) => vec![arg],
            UOp::Select(cond, then_, else_) => vec![cond, then_, else_],
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
            UOp::Wildcard(id) => {
                format!("{}Wildcard({}):{}", prefix, id, dtype_str)
            }
            UOp::Barrier => {
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
                    UOp::Sequence(_) => "Sequence".to_string(),
                    UOp::Add(..) => "Add".to_string(),
                    UOp::Mul(..) => "Mul".to_string(),
                    UOp::Max(..) => "Max".to_string(),
                    UOp::Rem(..) => "Rem".to_string(),
                    UOp::Idiv(..) => "Idiv".to_string(),
                    UOp::Recip(_) => "Recip".to_string(),
                    UOp::Sqrt(_) => "Sqrt".to_string(),
                    UOp::LessThan(..) => "LessThan".to_string(),
                    UOp::Select(..) => "Select".to_string(),
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
            UOp::Add(..) => {}
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
    fn test_dtype_inference() {
        // Inputからの型推論
        let a = input("a", vec![10], DType::F32);
        assert_eq!(a.dtype(), DType::F32);

        // Constからの型推論（異なる型でテスト）
        let b = const_val(42.0, DType::Unknown);
        assert_eq!(b.dtype(), DType::Unknown);

        // Addの型推論（左辺から継承）
        let c = add(a.clone(), input("b", vec![10], DType::F32));
        assert_eq!(c.dtype(), DType::F32);

        // ネストした演算の型推論
        let d = mul(c.clone(), const_val(2.0, DType::F32));
        assert_eq!(d.dtype(), DType::F32);
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
