use crate::uop::{ElementwiseOp, ReduceOp, UOp};
use harp::DType;
use std::rc::Rc;

// ========== マクロ定義 ==========

/// 二項演算のヘルパー関数を生成するマクロ
macro_rules! impl_binary_helper {
    ($fn_name:ident, $variant:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $fn_name(lhs: Rc<UOp>, rhs: Rc<UOp>) -> Rc<UOp> {
            let dtype = lhs.dtype().clone();
            Rc::new(UOp::$variant { dtype, lhs, rhs })
        }
    };
}

/// 単項演算のヘルパー関数を生成するマクロ
macro_rules! impl_unary_helper {
    ($fn_name:ident, $variant:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $fn_name(arg: Rc<UOp>) -> Rc<UOp> {
            let dtype = arg.dtype().clone();
            Rc::new(UOp::$variant { dtype, arg })
        }
    };
}

// ========== 二項演算ヘルパー ==========
impl_binary_helper!(add, Add, "Create an add node: lhs + rhs");
impl_binary_helper!(mul, Mul, "Create a mul node: lhs * rhs");
impl_binary_helper!(max, Max, "Create a max node: max(lhs, rhs)");
impl_binary_helper!(rem, Rem, "Create a remainder node: lhs % rhs");
impl_binary_helper!(idiv, Idiv, "Create an integer division node: lhs / rhs");
impl_binary_helper!(less_than, LessThan, "Create a less than node: lhs < rhs");

// ========== 単項演算ヘルパー ==========
impl_unary_helper!(recip, Recip, "Create a reciprocal node: 1 / arg");
impl_unary_helper!(sqrt, Sqrt, "Create a square root node: sqrt(arg)");

// ========== その他のヘルパー ==========

/// 入力ノードを作成
pub fn input(name: impl Into<String>, shape: Vec<usize>, dtype: DType) -> Rc<UOp> {
    Rc::new(UOp::Input {
        dtype,
        name: name.into(),
        shape,
    })
}

/// 定数ノードを作成
pub fn const_val(value: f64, dtype: DType) -> Rc<UOp> {
    Rc::new(UOp::Const { dtype, value })
}

/// 変数ノードを作成
pub fn var(name: impl Into<String>, dtype: DType) -> Rc<UOp> {
    Rc::new(UOp::Var {
        dtype,
        name: name.into(),
    })
}

/// ワイルドカードノードを作成（パターンマッチング用）
pub fn wildcard(id: usize, dtype: DType) -> Rc<UOp> {
    Rc::new(UOp::Wildcard { dtype, id })
}

/// スレッドインデックスノードを作成
pub fn thread_idx(dim: usize, dtype: DType) -> Rc<UOp> {
    Rc::new(UOp::ThreadIdx { dtype, dim })
}

/// グループインデックスノードを作成
pub fn group_idx(dim: usize, dtype: DType) -> Rc<UOp> {
    Rc::new(UOp::GroupIdx { dtype, dim })
}

/// バリアノードを作成
pub fn barrier(dtype: DType) -> Rc<UOp> {
    Rc::new(UOp::Barrier { dtype })
}

/// Element-wise演算ノードを作成
pub fn elementwise(op: ElementwiseOp, inputs: Vec<Rc<UOp>>, dtype: DType) -> Rc<UOp> {
    Rc::new(UOp::Elementwise { dtype, op, inputs })
}

/// Reduce演算ノードを作成
pub fn reduce(op: ReduceOp, input: Rc<UOp>, axis: usize, input_shape: Vec<usize>) -> Rc<UOp> {
    let dtype = input.dtype().clone();
    Rc::new(UOp::Reduce {
        dtype,
        op,
        input,
        axis,
        input_shape,
    })
}

/// ループノードを作成
pub fn loop_op(
    var: impl Into<String>,
    start: usize,
    end: usize,
    body: Rc<UOp>,
    parallel: bool,
) -> Rc<UOp> {
    let dtype = body.dtype().clone();
    Rc::new(UOp::Loop {
        dtype,
        var: var.into(),
        start,
        end,
        parallel,
        body,
    })
}

/// ロードノードを作成
pub fn load(buffer: impl Into<String>, index: Option<Rc<UOp>>, dtype: DType) -> Rc<UOp> {
    Rc::new(UOp::Load {
        dtype,
        buffer: buffer.into(),
        index,
    })
}

/// ストアノードを作成
pub fn store(buffer: impl Into<String>, index: Option<Rc<UOp>>, value: Rc<UOp>) -> Rc<UOp> {
    let dtype = value.dtype().clone();
    Rc::new(UOp::Store {
        dtype,
        buffer: buffer.into(),
        index,
        value,
    })
}

/// シーケンスノードを作成
pub fn sequence(ops: Vec<Rc<UOp>>) -> Rc<UOp> {
    let dtype = if ops.is_empty() {
        DType::F32
    } else {
        ops.last().unwrap().dtype().clone()
    };
    Rc::new(UOp::Sequence { dtype, ops })
}

/// Select（三項演算子）ノードを作成
pub fn select(cond: Rc<UOp>, then_: Rc<UOp>, else_: Rc<UOp>) -> Rc<UOp> {
    let dtype = then_.dtype().clone();
    Rc::new(UOp::Select {
        dtype,
        cond,
        then_,
        else_,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_ops() {
        let a = const_val(1.0, DType::F32);
        let b = const_val(2.0, DType::F32);

        let add_node = add(a.clone(), b.clone());
        match &*add_node {
            UOp::Add { .. } => {}
            _ => panic!("Expected Add node"),
        }

        let mul_node = mul(a.clone(), b.clone());
        match &*mul_node {
            UOp::Mul { .. } => {}
            _ => panic!("Expected Mul node"),
        }

        let max_node = max(a.clone(), b.clone());
        match &*max_node {
            UOp::Max { .. } => {}
            _ => panic!("Expected Max node"),
        }
    }

    #[test]
    fn test_unary_ops() {
        let a = const_val(4.0, DType::F32);

        let recip_node = recip(a.clone());
        match &*recip_node {
            UOp::Recip { .. } => {}
            _ => panic!("Expected Recip node"),
        }

        let sqrt_node = sqrt(a.clone());
        match &*sqrt_node {
            UOp::Sqrt { .. } => {}
            _ => panic!("Expected Sqrt node"),
        }
    }

    #[test]
    fn test_var_helper() {
        let var_node = var("x", DType::F32);
        match &*var_node {
            UOp::Var { name, .. } => assert_eq!(name, "x"),
            _ => panic!("Expected Var node"),
        }
    }

    #[test]
    fn test_load_store_helper() {
        let idx = var("i", DType::F32);
        let load_node = load("input", Some(idx.clone()), DType::F32);
        match &*load_node {
            UOp::Load { buffer, .. } => assert_eq!(buffer, "input"),
            _ => panic!("Expected Load node"),
        }

        let value = const_val(42.0, DType::F32);
        let store_node = store("output", Some(idx), value);
        match &*store_node {
            UOp::Store { buffer, .. } => assert_eq!(buffer, "output"),
            _ => panic!("Expected Store node"),
        }
    }

    #[test]
    fn test_loop_helper() {
        let body = const_val(0.0, DType::F32);
        let loop_node = loop_op("i", 0, 100, body, true);
        match &*loop_node {
            UOp::Loop {
                var,
                start,
                end,
                parallel,
                ..
            } => {
                assert_eq!(var, "i");
                assert_eq!(*start, 0);
                assert_eq!(*end, 100);
                assert!(*parallel);
            }
            _ => panic!("Expected Loop node"),
        }
    }

    #[test]
    fn test_sequence_helper() {
        let a = const_val(1.0, DType::F32);
        let b = const_val(2.0, DType::F32);
        let seq = sequence(vec![a, b]);
        match &*seq {
            UOp::Sequence { ops, .. } => assert_eq!(ops.len(), 2),
            _ => panic!("Expected Sequence node"),
        }
    }

    #[test]
    fn test_composite_expression() {
        // (a + b) * c
        let a = const_val(1.0, DType::F32);
        let b = const_val(2.0, DType::F32);
        let c = const_val(3.0, DType::F32);

        let sum = add(a, b);
        let product = mul(sum, c);

        match &*product {
            UOp::Mul { lhs, .. } => match &**lhs {
                UOp::Add { .. } => {}
                _ => panic!("Expected Add node inside Mul"),
            },
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_memory_operation() {
        // output[i] = input[i] * 2.0
        let i = var("i", DType::F32);
        let loaded = load("input", Some(i.clone()), DType::F32);
        let two = const_val(2.0, DType::F32);
        let doubled = mul(loaded, two);
        let stored = store("output", Some(i), doubled);

        match &*stored {
            UOp::Store { .. } => {}
            _ => panic!("Expected Store node"),
        }
    }
}
