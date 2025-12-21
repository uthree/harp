//! グラフシグネチャとカーネルキャッシュ
//!
//! 計算グラフの構造をハッシュ化し、同じ計算パターンを再利用するための機構を提供します。

use harp_core::graph::shape::Expr;
use harp_core::graph::{DType, GraphNode};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

// ============================================================================
// GraphSignature - 計算グラフの構造を表すシグネチャ
// ============================================================================

/// 形状シグネチャの要素
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShapeSig {
    /// 静的な形状（コンパイル時に確定）
    Static(usize),
    /// 動的な形状（shape変数名）
    Dynamic(String),
}

/// データ型のシグネチャ（ハッシュ可能な表現）
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DTypeSig {
    Unknown,
    Bool,
    I32,
    F32,
}

impl From<&DType> for DTypeSig {
    fn from(dtype: &DType) -> Self {
        match dtype {
            DType::Unknown => DTypeSig::Unknown,
            DType::Bool => DTypeSig::Bool,
            DType::I32 => DTypeSig::I32,
            DType::F32 => DTypeSig::F32,
        }
    }
}

/// グラフシグネチャ
///
/// 計算グラフの構造を一意に識別するためのシグネチャです。
/// 同じ構造のグラフは同じシグネチャを持ちます。
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphSignature {
    /// 構造ハッシュ（演算の種類とトポロジー）
    structure_hash: u64,
    /// 入力の形状シグネチャ
    input_shapes: Vec<Vec<ShapeSig>>,
    /// 出力の形状シグネチャ
    output_shapes: Vec<Vec<ShapeSig>>,
    /// 入力のデータ型
    input_dtypes: Vec<DTypeSig>,
    /// 出力のデータ型
    output_dtypes: Vec<DTypeSig>,
}

impl GraphSignature {
    /// GraphNodeからシグネチャを生成
    ///
    /// 出力ノードのリストからグラフ全体の構造をハッシュ化します。
    pub fn from_outputs(outputs: &[&GraphNode]) -> Self {
        let mut hasher = DefaultHasher::new();
        let mut visited = HashSet::new();

        // トポロジカル順序でノードを訪問してハッシュを計算
        for output in outputs {
            Self::hash_node(output, &mut hasher, &mut visited);
        }

        let structure_hash = hasher.finish();

        // 出力の形状とデータ型を収集
        let output_shapes: Vec<_> = outputs.iter().map(|n| Self::shape_to_sig(n)).collect();
        let output_dtypes: Vec<_> = outputs.iter().map(|n| DTypeSig::from(&n.dtype)).collect();

        // 入力ノードを収集
        let input_nodes = Self::collect_inputs(outputs);
        let input_shapes: Vec<_> = input_nodes.iter().map(Self::shape_to_sig).collect();
        let input_dtypes: Vec<_> = input_nodes
            .iter()
            .map(|n| DTypeSig::from(&n.dtype))
            .collect();

        Self {
            structure_hash,
            input_shapes,
            output_shapes,
            input_dtypes,
            output_dtypes,
        }
    }

    /// 単一の出力ノードからシグネチャを生成
    pub fn from_output(output: &GraphNode) -> Self {
        Self::from_outputs(&[output])
    }

    /// ノードを再帰的にハッシュ
    fn hash_node(node: &GraphNode, hasher: &mut impl Hasher, visited: &mut HashSet<usize>) {
        let ptr = node.as_ptr() as usize;
        if !visited.insert(ptr) {
            // 既に訪問済みのノードは参照としてハッシュ
            ptr.hash(hasher);
            return;
        }

        // 演算の種類をハッシュ（discriminantを使用）
        std::mem::discriminant(&node.op).hash(hasher);

        // データ型をハッシュ
        DTypeSig::from(&node.dtype).hash(hasher);

        // 子ノードを再帰的にハッシュ
        for child in &node.src {
            Self::hash_node(child, hasher, visited);
        }
    }

    /// 形状をシグネチャに変換
    fn shape_to_sig(node: &GraphNode) -> Vec<ShapeSig> {
        node.view.shape().iter().map(Self::expr_to_sig).collect()
    }

    /// Exprをシグネチャに変換
    fn expr_to_sig(expr: &Expr) -> ShapeSig {
        if let Some(val) = expr.as_const() {
            ShapeSig::Static(val as usize)
        } else {
            ShapeSig::Dynamic(format!("{:?}", expr))
        }
    }

    /// 入力ノード（Bufferノード）を収集
    fn collect_inputs(outputs: &[&GraphNode]) -> Vec<GraphNode> {
        use harp_core::graph::GraphOp;

        let mut inputs = Vec::new();
        let mut visited = HashSet::new();

        fn visit(node: &GraphNode, inputs: &mut Vec<GraphNode>, visited: &mut HashSet<usize>) {
            let ptr = node.as_ptr() as usize;
            if !visited.insert(ptr) {
                return;
            }

            // Bufferノードは入力
            if matches!(&node.op, GraphOp::Buffer { .. }) {
                inputs.push(node.clone());
            }

            for child in &node.src {
                visit(child, inputs, visited);
            }
        }

        for output in outputs {
            visit(output, &mut inputs, &mut visited);
        }

        inputs
    }

    /// 構造ハッシュを取得
    pub fn structure_hash(&self) -> u64 {
        self.structure_hash
    }
}

// ============================================================================
// KernelCache - コンパイル済みカーネルのキャッシュ
// ============================================================================

/// コンパイル済みカーネルのキャッシュ
///
/// スレッドセーフなキャッシュで、同じ計算グラフに対するカーネルを再利用します。
pub struct KernelCache<K> {
    cache: RwLock<HashMap<GraphSignature, Arc<K>>>,
}

impl<K> Default for KernelCache<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K> KernelCache<K> {
    /// 新しいキャッシュを作成
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// キャッシュからカーネルを取得
    pub fn get(&self, sig: &GraphSignature) -> Option<Arc<K>> {
        self.cache.read().unwrap().get(sig).cloned()
    }

    /// キャッシュにカーネルを挿入
    pub fn insert(&self, sig: GraphSignature, kernel: K) -> Arc<K> {
        let arc_kernel = Arc::new(kernel);
        self.cache.write().unwrap().insert(sig, arc_kernel.clone());
        arc_kernel
    }

    /// キャッシュから取得、なければ生成して挿入
    pub fn get_or_insert<F, E>(&self, sig: GraphSignature, f: F) -> Result<Arc<K>, E>
    where
        F: FnOnce() -> Result<K, E>,
    {
        // まず読み取りロックで確認
        if let Some(kernel) = self.get(&sig) {
            return Ok(kernel);
        }

        // なければ生成して挿入
        let kernel = f()?;
        Ok(self.insert(sig, kernel))
    }

    /// キャッシュをクリア
    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
    }

    /// キャッシュのエントリ数を取得
    pub fn len(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    /// キャッシュが空かどうか
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_cache_basic() {
        let cache: KernelCache<String> = KernelCache::new();
        assert!(cache.is_empty());

        // 手動でシグネチャを作成（テスト用）
        let sig = GraphSignature {
            structure_hash: 12345,
            input_shapes: vec![vec![ShapeSig::Static(100), ShapeSig::Static(100)]],
            output_shapes: vec![vec![ShapeSig::Static(100), ShapeSig::Static(100)]],
            input_dtypes: vec![DTypeSig::F32],
            output_dtypes: vec![DTypeSig::F32],
        };

        // 挿入
        let kernel = cache.insert(sig.clone(), "test_kernel".to_string());
        assert_eq!(cache.len(), 1);
        assert_eq!(*kernel, "test_kernel");

        // 取得
        let retrieved = cache.get(&sig);
        assert!(retrieved.is_some());
        assert_eq!(*retrieved.unwrap(), "test_kernel");
    }

    #[test]
    fn test_kernel_cache_get_or_insert() {
        let cache: KernelCache<i32> = KernelCache::new();

        let sig = GraphSignature {
            structure_hash: 99999,
            input_shapes: vec![],
            output_shapes: vec![vec![ShapeSig::Static(10)]],
            input_dtypes: vec![],
            output_dtypes: vec![DTypeSig::I32],
        };

        // 初回は生成される
        let result: Result<Arc<i32>, ()> = cache.get_or_insert(sig.clone(), || Ok(42));
        assert!(result.is_ok());
        assert_eq!(*result.unwrap(), 42);

        // 2回目はキャッシュから取得
        let mut called = false;
        let result: Result<Arc<i32>, ()> = cache.get_or_insert(sig, || {
            called = true;
            Ok(100)
        });
        assert!(result.is_ok());
        assert_eq!(*result.unwrap(), 42); // 元の値
        assert!(!called); // クロージャは呼ばれない
    }

    #[test]
    fn test_shape_sig() {
        let static_sig = ShapeSig::Static(100);
        let dynamic_sig = ShapeSig::Dynamic("batch_size".to_string());

        assert_ne!(static_sig, dynamic_sig);

        // ハッシュ可能
        let mut set = HashSet::new();
        set.insert(static_sig.clone());
        set.insert(dynamic_sig.clone());
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_dtype_sig() {
        assert_eq!(DTypeSig::from(&DType::F32), DTypeSig::F32);
        assert_eq!(DTypeSig::from(&DType::I32), DTypeSig::I32);
        assert_eq!(DTypeSig::from(&DType::Bool), DTypeSig::Bool);
        assert_eq!(DTypeSig::from(&DType::Unknown), DTypeSig::Unknown);
    }
}
