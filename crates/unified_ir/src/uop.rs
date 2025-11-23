use harp::DType;
use std::collections::HashMap;
use std::rc::Rc;

/// 統一中間表現（Unified Operation）
/// 高レベル演算から低レベル演算までを単一のグラフ構造で表現
#[derive(Debug, Clone, PartialEq)]
pub struct UOp(pub Rc<UOpData>);

#[derive(Debug, Clone, PartialEq)]
pub struct UOpData {
    pub op: UOpKind,
    pub dtype: DType,
    pub src: Vec<UOp>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UOpKind {
    // ========== 高レベル演算（テンソルレベル） ==========
    /// テンソル入力
    Input { name: String, shape: Vec<usize> },

    /// 定数値
    Const { value: f64 },

    /// Element-wise演算
    /// src[0]: 入力テンソル（Binary演算の場合はsrc[1]も使用）
    Elementwise { op: ElementwiseOp },

    /// Reduce演算
    /// src[0]: 入力テンソル
    Reduce {
        op: ReduceOp,
        axis: usize,
        input_shape: Vec<usize>,
    },

    // ========== 中レベル演算（ループレベル） ==========
    /// ループ構造
    /// src[0]: ループ本体
    Loop {
        var: String,
        start: usize,
        end: usize,
        parallel: bool, // GPU並列化するか
    },

    /// メモリロード
    /// src[0]: アドレス計算
    Load {
        buffer: String,
        index: Option<Box<UOp>>, // Noneの場合は直接バッファー参照
    },

    /// メモリストア
    /// src[0]: アドレス計算
    /// src[1]: 格納する値
    Store {
        buffer: String,
        index: Option<Box<UOp>>,
    },

    /// バリア同期
    Barrier,

    /// シーケンス（複数の操作を順次実行）
    Sequence,

    // ========== 低レベル演算（スカラーレベル） ==========
    /// スレッドインデックス（GPUの場合）
    ThreadIdx { dim: usize },

    /// グループインデックス（GPUの場合）
    GroupIdx { dim: usize },

    /// 変数参照
    Var { name: String },

    /// スカラー加算
    Add,

    /// スカラー乗算
    Mul,

    /// スカラー最大値
    Max,

    /// スカラー剰余
    Rem,

    /// スカラー整数除算
    Idiv,

    /// スカラー逆数
    Recip,

    /// スカラー平方根
    Sqrt,

    /// スカラー比較（小なり）
    LessThan,

    /// 三項演算子 (src[0] ? src[1] : src[2])
    Select,

    // ========== パターンマッチング用 ==========
    /// ワイルドカード（パターンマッチング用）
    Wildcard { id: usize },
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
    /// 新しいUOpを作成
    pub fn new(op: UOpKind, dtype: DType, src: Vec<UOp>) -> Self {
        UOp(Rc::new(UOpData { op, dtype, src }))
    }

    /// Input演算を作成
    pub fn input(name: impl Into<String>, shape: Vec<usize>, dtype: DType) -> Self {
        Self::new(
            UOpKind::Input {
                name: name.into(),
                shape,
            },
            dtype,
            vec![],
        )
    }

    /// Const演算を作成
    pub fn const_val(value: f64, dtype: DType) -> Self {
        Self::new(UOpKind::Const { value }, dtype, vec![])
    }

    /// Elementwise演算を作成
    pub fn elementwise(op: ElementwiseOp, inputs: Vec<UOp>, dtype: DType) -> Self {
        Self::new(UOpKind::Elementwise { op }, dtype, inputs)
    }

    /// Reduce演算を作成
    pub fn reduce(op: ReduceOp, input: UOp, axis: usize, input_shape: Vec<usize>) -> Self {
        let dtype = input.0.dtype.clone();
        Self::new(
            UOpKind::Reduce {
                op,
                axis,
                input_shape,
            },
            dtype,
            vec![input],
        )
    }

    /// Add演算を作成
    pub fn add(lhs: UOp, rhs: UOp) -> Self {
        let dtype = lhs.0.dtype.clone();
        Self::new(UOpKind::Add, dtype, vec![lhs, rhs])
    }

    /// Mul演算を作成
    pub fn mul(lhs: UOp, rhs: UOp) -> Self {
        let dtype = lhs.0.dtype.clone();
        Self::new(UOpKind::Mul, dtype, vec![lhs, rhs])
    }

    /// Wildcard演算を作成（パターンマッチング用）
    pub fn wildcard(id: usize, dtype: DType) -> Self {
        Self::new(UOpKind::Wildcard { id }, dtype, vec![])
    }

    /// Var演算を作成
    pub fn var(name: impl Into<String>, dtype: DType) -> Self {
        Self::new(UOpKind::Var { name: name.into() }, dtype, vec![])
    }

    /// ThreadIdx演算を作成
    pub fn thread_idx(dim: usize, dtype: DType) -> Self {
        Self::new(UOpKind::ThreadIdx { dim }, dtype, vec![])
    }

    /// Loop演算を作成
    pub fn loop_op(var: String, start: usize, end: usize, body: UOp, parallel: bool) -> Self {
        let dtype = body.0.dtype.clone();
        Self::new(
            UOpKind::Loop {
                var,
                start,
                end,
                parallel,
            },
            dtype,
            vec![body],
        )
    }

    /// Load演算を作成
    pub fn load(buffer: String, index: Option<UOp>, dtype: DType) -> Self {
        Self::new(
            UOpKind::Load {
                buffer,
                index: index.map(Box::new),
            },
            dtype,
            vec![],
        )
    }

    /// Store演算を作成
    pub fn store(buffer: String, index: Option<UOp>, value: UOp) -> Self {
        let dtype = value.0.dtype.clone();
        Self::new(
            UOpKind::Store {
                buffer,
                index: index.map(Box::new),
            },
            dtype,
            vec![value],
        )
    }

    /// Sequence演算を作成
    pub fn sequence(ops: Vec<UOp>) -> Self {
        let dtype = if ops.is_empty() {
            DType::F32
        } else {
            ops.last().unwrap().0.dtype.clone()
        };
        Self::new(UOpKind::Sequence, dtype, ops)
    }

    /// パターンマッチングとワイルドカード置換
    pub fn substitute(&self, mapping: &HashMap<usize, UOp>) -> UOp {
        match &self.0.op {
            UOpKind::Wildcard { id } => mapping.get(id).cloned().unwrap_or_else(|| self.clone()),
            _ => {
                let new_src: Vec<UOp> = self.0.src.iter().map(|s| s.substitute(mapping)).collect();

                if new_src
                    .iter()
                    .zip(&self.0.src)
                    .all(|(a, b)| Rc::ptr_eq(&a.0, &b.0))
                {
                    self.clone()
                } else {
                    UOp::new(self.0.op.clone(), self.0.dtype.clone(), new_src)
                }
            }
        }
    }

    /// UOpのグラフを可視化用に文字列化
    pub fn to_debug_string(&self, indent: usize) -> String {
        let prefix = "  ".repeat(indent);
        let op_str = format!("{:?}", self.0.op);
        let dtype_str = format!("{:?}", self.0.dtype);

        if self.0.src.is_empty() {
            format!("{}{}:{}", prefix, op_str, dtype_str)
        } else {
            let mut result = format!("{}{}:{}(\n", prefix, op_str, dtype_str);
            for src in &self.0.src {
                result.push_str(&src.to_debug_string(indent + 1));
                result.push('\n');
            }
            result.push_str(&format!("{})", prefix));
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_uop() {
        let a = UOp::input("a", vec![10], DType::F32);
        let b = UOp::input("b", vec![10], DType::F32);
        let c = UOp::add(a, b);

        println!("{}", c.to_debug_string(0));

        match &c.0.op {
            UOpKind::Add => {}
            _ => panic!("Expected Add"),
        }
        assert_eq!(c.0.src.len(), 2);
    }

    #[test]
    fn test_reduce() {
        let a = UOp::input("a", vec![10, 20], DType::F32);
        let sum = UOp::reduce(ReduceOp::Sum, a, 1, vec![10, 20]);

        println!("{}", sum.to_debug_string(0));

        match &sum.0.op {
            UOpKind::Reduce { op, axis, .. } => {
                assert_eq!(*op, ReduceOp::Sum);
                assert_eq!(*axis, 1);
            }
            _ => panic!("Expected Reduce"),
        }
    }

    #[test]
    fn test_wildcard_substitution() {
        let w0 = UOp::wildcard(0, DType::F32);
        let w1 = UOp::wildcard(1, DType::F32);
        let pattern = UOp::add(w0.clone(), w1.clone());

        let a = UOp::const_val(1.0, DType::F32);
        let b = UOp::const_val(2.0, DType::F32);

        let mut mapping = HashMap::new();
        mapping.insert(0, a);
        mapping.insert(1, b);

        let result = pattern.substitute(&mapping);

        match &result.0.op {
            UOpKind::Add => {}
            _ => panic!("Expected Add"),
        }

        match &result.0.src[0].0.op {
            UOpKind::Const { value } => assert_eq!(*value, 1.0),
            _ => panic!("Expected Const"),
        }
    }
}
