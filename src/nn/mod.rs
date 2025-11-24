//! ニューラルネットワークモジュール
//!
//! PyTorchのtorch.nnに相当する機能を提供します。
//!
//! # 設計思想
//!
//! - **軽量なModule trait**: 必要最小限の機能に絞った実装
//! - **New type pattern**: `Parameter(Tensor)`でパラメータを型レベルで区別
//! - **明示的な管理**: 階層構造は明示的にパラメータを収集
//! - **ゼロコスト抽象化**: ランタイムオーバーヘッドを最小化

pub mod init;
pub mod macros;
pub mod optim;

use crate::autograd::Tensor;
use std::collections::HashMap;

/// 学習可能なパラメータ
///
/// `Tensor`のnewtype wrapper。`requires_grad`が常に`true`であることを型で保証します。
///
/// # Examples
///
/// ```
/// use harp::nn::Parameter;
/// use harp::autograd::Tensor;
///
/// // ゼロで初期化
/// let param = Parameter::zeros(vec![10, 20]);
///
/// // 既存のTensorから作成
/// let tensor = Tensor::ones(vec![5, 5]);
/// let param = Parameter::new(tensor);
///
/// // Tensorのメソッドを透過的に使用可能（Derefトレイトによる）
/// let shape = param.data.view.shape();
/// ```
#[derive(Clone)]
pub struct Parameter(Tensor);

impl Parameter {
    /// 既存のTensorからParameterを作成
    ///
    /// `requires_grad`は自動的に`true`に設定されます。
    pub fn new(tensor: Tensor) -> Self {
        // requires_grad=trueで新しいTensorを作成
        let t = Tensor::from_graph_node(tensor.data.clone(), true);
        Parameter(t)
    }

    /// ゼロで初期化されたParameterを作成
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::new(Tensor::zeros(shape))
    }

    /// 1で初期化されたParameterを作成
    pub fn ones(shape: Vec<usize>) -> Self {
        Self::new(Tensor::ones(shape))
    }

    /// 内部のTensorへの不変参照を取得
    pub fn tensor(&self) -> &Tensor {
        &self.0
    }

    /// 内部のTensorへの可変参照を取得
    pub fn tensor_mut(&mut self) -> &mut Tensor {
        &mut self.0
    }

    /// Tensorに変換（所有権を移動）
    pub fn into_tensor(self) -> Tensor {
        self.0
    }
}

// DerefトレイトでTensorのメソッドを透過的に使えるようにする
impl std::ops::Deref for Parameter {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Parameter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// ニューラルネットワークのモジュール
///
/// PyTorchの`nn.Module`に相当するtraitです。
///
/// # 設計
///
/// このtraitは**パラメータ管理のみ**に特化しています。
/// `forward`メソッドは含まれていません。各モジュールが自由に定義できます。
///
/// - `named_parameters()`: 名前付きパラメータの辞書を返す（必須実装）
/// - `named_parameters_mut()`: 可変参照版（必須実装）
/// - `parameters()`: 名前なしリスト（デフォルト実装あり）
/// - `parameters_mut()`: 可変参照リスト（デフォルト実装あり）
/// - `zero_grad()`: デフォルト実装あり
/// - `num_parameters()`: デフォルト実装あり
///
/// # 名前付きパラメータ
///
/// パラメータには必ず名前を付ける必要があります。これにより：
/// - モデルの保存/読み込み（`state_dict()`）が可能
/// - デバッグが容易（どのパラメータが問題かわかる）
/// - 選択的な更新（特定の層のみ凍結など）が可能
/// - 既存フレームワークとの互換性が向上
///
/// # Forwardメソッドについて
///
/// `forward`メソッドは各モジュールで自由に定義してください。
/// 引数の数や型、返り値の型が異なる場合も対応できます。
///
/// # Examples
///
/// ```ignore
/// use harp::nn::{Module, Parameter};
/// use harp::autograd::Tensor;
/// use std::collections::HashMap;
///
/// struct Linear {
///     weight: Parameter,
///     bias: Parameter,
/// }
///
/// impl Linear {
///     // forwardは通常のメソッドとして定義
///     pub fn forward(&self, input: &Tensor) -> Tensor {
///         // input.matmul(&self.weight) + &self.bias
///         unimplemented!()
///     }
/// }
///
/// impl Module for Linear {
///     fn named_parameters(&self) -> HashMap<String, &Parameter> {
///         let mut params = HashMap::new();
///         params.insert("weight".to_string(), &self.weight);
///         params.insert("bias".to_string(), &self.bias);
///         params
///     }
///
///     fn named_parameters_mut(&mut self) -> HashMap<String, &mut Parameter> {
///         let mut params = HashMap::new();
///         params.insert("weight".to_string(), &mut self.weight);
///         params.insert("bias".to_string(), &mut self.bias);
///         params
///     }
/// }
///
/// // 階層的なモジュールの例
/// struct MLP {
///     layer1: Linear,
///     layer2: Linear,
/// }
///
/// impl Module for MLP {
///     fn named_parameters(&self) -> HashMap<String, &Parameter> {
///         let mut params = HashMap::new();
///         // layer1のパラメータにプレフィックスを付けて追加
///         for (name, param) in self.layer1.named_parameters() {
///             params.insert(format!("layer1.{}", name), param);
///         }
///         // layer2のパラメータにプレフィックスを付けて追加
///         for (name, param) in self.layer2.named_parameters() {
///             params.insert(format!("layer2.{}", name), param);
///         }
///         params
///     }
///
///     fn named_parameters_mut(&mut self) -> HashMap<String, &mut Parameter> {
///         let mut params = HashMap::new();
///         for (name, param) in self.layer1.named_parameters_mut() {
///             params.insert(format!("layer1.{}", name), param);
///         }
///         for (name, param) in self.layer2.named_parameters_mut() {
///             params.insert(format!("layer2.{}", name), param);
///         }
///         params
///     }
/// }
/// ```
pub trait Module {
    /// このモジュールの全パラメータを名前付きで返す
    ///
    /// サブモジュールがある場合は、階層的な名前（例: "layer1.weight"）を使用してください。
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let params = module.named_parameters();
    /// for (name, param) in params {
    ///     println!("{}: shape = {:?}", name, param.data.view.shape());
    /// }
    /// ```
    fn named_parameters(&self) -> HashMap<String, &Parameter>;

    /// パラメータの可変参照を名前付きで返す
    ///
    /// オプティマイザーがパラメータを更新する際に使用します。
    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Parameter>;

    /// パラメータのリストを返す（便利メソッド）
    ///
    /// 名前が不要な場合に使用します。オプティマイザーなど。
    fn parameters(&self) -> Vec<&Parameter> {
        self.named_parameters().values().copied().collect()
    }

    /// パラメータの可変参照リストを返す（便利メソッド）
    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        self.named_parameters_mut().into_values().collect()
    }

    /// 全パラメータの勾配をゼロクリア
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }

    /// パラメータの総数を返す
    fn num_parameters(&self) -> usize {
        self.parameters()
            .iter()
            .map(|p| {
                p.data
                    .view
                    .shape()
                    .iter()
                    .map(|s| match s {
                        crate::graph::shape::Expr::Const(v) => *v as usize,
                        _ => 0,
                    })
                    .product::<usize>()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_new() {
        let tensor = Tensor::zeros(vec![10, 20]);
        let param = Parameter::new(tensor);

        // requires_gradがtrueになっているか確認
        assert!(param.requires_grad());
    }

    #[test]
    fn test_parameter_zeros() {
        let param = Parameter::zeros(vec![5, 10]);

        // requires_gradがtrueになっているか確認
        assert!(param.requires_grad());

        // 形状の確認
        let shape = param.data.view.shape();
        assert_eq!(shape.len(), 2);
    }

    #[test]
    fn test_parameter_ones() {
        let param = Parameter::ones(vec![3, 4]);

        // requires_gradがtrueになっているか確認
        assert!(param.requires_grad());

        // 形状の確認
        let shape = param.data.view.shape();
        assert_eq!(shape.len(), 2);
    }

    #[test]
    fn test_parameter_deref() {
        let param = Parameter::zeros(vec![10, 20]);

        // Derefで透過的にTensorのメソッドが使える
        assert!(param.requires_grad());

        // grad()も使える（初期状態ではNone）
        assert!(param.grad().is_none());
    }

    // 簡単なModuleの実装例
    struct SimpleModule {
        weight: Parameter,
        bias: Parameter,
    }

    impl SimpleModule {
        // forwardは通常のメソッドとして定義
        pub fn forward(&self, input: &Tensor) -> Tensor {
            // 簡単な線形変換（matmul未実装なので加算のみ）
            let tmp = input + self.weight.tensor();
            &tmp + self.bias.tensor()
        }
    }

    impl Module for SimpleModule {
        fn named_parameters(&self) -> HashMap<String, &Parameter> {
            let mut params = HashMap::new();
            params.insert("weight".to_string(), &self.weight);
            params.insert("bias".to_string(), &self.bias);
            params
        }

        fn named_parameters_mut(&mut self) -> HashMap<String, &mut Parameter> {
            let mut params = HashMap::new();
            params.insert("weight".to_string(), &mut self.weight);
            params.insert("bias".to_string(), &mut self.bias);
            params
        }
    }

    #[test]
    fn test_module_parameters() {
        let module = SimpleModule {
            weight: Parameter::zeros(vec![10, 20]),
            bias: Parameter::zeros(vec![10, 20]),
        };

        let params = module.parameters();
        assert_eq!(params.len(), 2);

        // すべてrequires_grad=true
        for p in params {
            assert!(p.requires_grad());
        }
    }

    #[test]
    fn test_module_num_parameters() {
        let module = SimpleModule {
            weight: Parameter::zeros(vec![10, 20]),
            bias: Parameter::zeros(vec![10, 20]),
        };

        let num = module.num_parameters();
        assert_eq!(num, 10 * 20 + 10 * 20); // 400
    }

    #[test]
    fn test_module_zero_grad() {
        let module = SimpleModule {
            weight: Parameter::zeros(vec![2, 2]),
            bias: Parameter::zeros(vec![2, 2]),
        };

        // zero_gradが呼べることを確認
        module.zero_grad();

        // 勾配が初期化されていることを確認
        for p in module.parameters() {
            assert!(p.grad().is_none());
        }
    }

    #[test]
    fn test_module_forward() {
        let module = SimpleModule {
            weight: Parameter::zeros(vec![2, 2]),
            bias: Parameter::ones(vec![2, 2]),
        };

        let input = Tensor::ones(vec![2, 2]);

        // forwardを呼び出せることを確認
        let output = module.forward(&input);

        // 出力の形状確認
        let shape = output.data.view.shape();
        assert_eq!(shape.len(), 2);
    }

    #[test]
    fn test_module_named_parameters() {
        let module = SimpleModule {
            weight: Parameter::zeros(vec![10, 20]),
            bias: Parameter::zeros(vec![10, 20]),
        };

        let named_params = module.named_parameters();
        assert_eq!(named_params.len(), 2);

        // weightとbiasが含まれているか確認
        assert!(named_params.contains_key("weight"));
        assert!(named_params.contains_key("bias"));

        // すべてrequires_grad=true
        for (_name, param) in named_params {
            assert!(param.requires_grad());
        }
    }

    #[test]
    fn test_module_named_parameters_mut() {
        let mut module = SimpleModule {
            weight: Parameter::zeros(vec![5, 5]),
            bias: Parameter::zeros(vec![5, 5]),
        };

        let named_params_mut = module.named_parameters_mut();
        assert_eq!(named_params_mut.len(), 2);

        // weightとbiasが含まれているか確認
        assert!(named_params_mut.contains_key("weight"));
        assert!(named_params_mut.contains_key("bias"));
    }

    #[test]
    fn test_module_parameters_from_named() {
        let module = SimpleModule {
            weight: Parameter::zeros(vec![10, 20]),
            bias: Parameter::zeros(vec![10, 20]),
        };

        // parameters()はnamed_parameters()から自動生成される
        let params = module.parameters();
        assert_eq!(params.len(), 2);

        // すべてrequires_grad=true
        for param in params {
            assert!(param.requires_grad());
        }
    }
}
