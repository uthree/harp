//! Module実装を自動化するマクロ
//!
//! `impl_module!`マクロを使うことで、Moduleトレイトの実装を自動生成できます。
//!
//! # 使用例
//!
//! ## Parameterのみの場合
//!
//! ```ignore
//! use harp::nn::{Module, Parameter};
//! use harp::impl_module;
//!
//! struct Linear {
//!     weight: Parameter,
//!     bias: Parameter,
//! }
//!
//! impl_module! {
//!     for Linear {
//!         parameters: [weight, bias]
//!     }
//! }
//! ```
//!
//! ## 階層的なモジュール
//!
//! ```ignore
//! struct MLP {
//!     layer1: Linear,
//!     layer2: Linear,
//! }
//!
//! impl_module! {
//!     for MLP {
//!         modules: [layer1, layer2]
//!     }
//! }
//! ```
//!
//! ## 混在する場合
//!
//! ```ignore
//! struct CustomModule {
//!     weight: Parameter,
//!     sub_module: Linear,
//! }
//!
//! impl_module! {
//!     for CustomModule {
//!         parameters: [weight],
//!         modules: [sub_module]
//!     }
//! }
//! ```

/// Module traitの実装を自動生成するマクロ
///
/// # 構文
///
/// ```ignore
/// impl_module! {
///     for StructName {
///         parameters: [field1, field2, ...],  // Parameterフィールド（省略可）
///         modules: [module1, module2, ...]     // サブモジュール（省略可）
///     }
/// }
/// ```
///
/// # 生成されるコード
///
/// - `named_parameters(&self) -> HashMap<String, &Parameter>`
/// - `named_parameters_mut(&mut self) -> HashMap<String, &mut Parameter>`
///
/// Parameterフィールドはフィールド名がそのままパラメータ名になります。
/// サブモジュールのパラメータには `"module_name.param_name"` の形式で階層的な名前が付けられます。
#[macro_export]
macro_rules! impl_module {
    // パラメータのみ
    (for $struct_name:ident {
        parameters: [$($param:ident),* $(,)?]
    }) => {
        impl $crate::nn::Module for $struct_name {
            fn named_parameters(&self) -> std::collections::HashMap<String, &$crate::nn::Parameter> {
                let mut params = std::collections::HashMap::new();
                $(
                    params.insert(stringify!($param).to_string(), &self.$param);
                )*
                params
            }

            fn named_parameters_mut(&mut self) -> std::collections::HashMap<String, &mut $crate::nn::Parameter> {
                let mut params = std::collections::HashMap::new();
                $(
                    params.insert(stringify!($param).to_string(), &mut self.$param);
                )*
                params
            }
        }
    };

    // サブモジュールのみ
    (for $struct_name:ident {
        modules: [$($module:ident),* $(,)?]
    }) => {
        impl $crate::nn::Module for $struct_name {
            fn named_parameters(&self) -> std::collections::HashMap<String, &$crate::nn::Parameter> {
                let mut params = std::collections::HashMap::new();
                $(
                    for (name, param) in self.$module.named_parameters() {
                        params.insert(format!("{}.{}", stringify!($module), name), param);
                    }
                )*
                params
            }

            fn named_parameters_mut(&mut self) -> std::collections::HashMap<String, &mut $crate::nn::Parameter> {
                let mut params = std::collections::HashMap::new();
                $(
                    for (name, param) in self.$module.named_parameters_mut() {
                        params.insert(format!("{}.{}", stringify!($module), name), param);
                    }
                )*
                params
            }
        }
    };

    // パラメータとサブモジュールの両方
    (for $struct_name:ident {
        parameters: [$($param:ident),* $(,)?],
        modules: [$($module:ident),* $(,)?]
    }) => {
        impl $crate::nn::Module for $struct_name {
            fn named_parameters(&self) -> std::collections::HashMap<String, &$crate::nn::Parameter> {
                let mut params = std::collections::HashMap::new();

                // Parameterフィールドを追加
                $(
                    params.insert(stringify!($param).to_string(), &self.$param);
                )*

                // サブモジュールのパラメータを追加
                $(
                    for (name, param) in self.$module.named_parameters() {
                        params.insert(format!("{}.{}", stringify!($module), name), param);
                    }
                )*

                params
            }

            fn named_parameters_mut(&mut self) -> std::collections::HashMap<String, &mut $crate::nn::Parameter> {
                let mut params = std::collections::HashMap::new();

                // Parameterフィールドを追加
                $(
                    params.insert(stringify!($param).to_string(), &mut self.$param);
                )*

                // サブモジュールのパラメータを追加
                $(
                    for (name, param) in self.$module.named_parameters_mut() {
                        params.insert(format!("{}.{}", stringify!($module), name), param);
                    }
                )*

                params
            }
        }
    };

    // サブモジュールが先に書かれている場合もサポート
    (for $struct_name:ident {
        modules: [$($module:ident),* $(,)?],
        parameters: [$($param:ident),* $(,)?]
    }) => {
        impl_module! {
            for $struct_name {
                parameters: [$($param),*],
                modules: [$($module),*]
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::super::{Module, Parameter};

    // テスト用の簡単なモジュール
    struct SimpleModule {
        weight: Parameter,
        bias: Parameter,
    }

    impl_module! {
        for SimpleModule {
            parameters: [weight, bias]
        }
    }

    #[test]
    fn test_simple_module_parameters() {
        let module = SimpleModule {
            weight: Parameter::zeros(vec![10, 20]),
            bias: Parameter::zeros(vec![10]),
        };

        let params = module.named_parameters();
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    #[test]
    fn test_simple_module_parameters_mut() {
        let mut module = SimpleModule {
            weight: Parameter::zeros(vec![10, 20]),
            bias: Parameter::zeros(vec![10]),
        };

        let params = module.named_parameters_mut();
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    // 階層的なモジュールのテスト
    struct TwoLayerNet {
        layer1: SimpleModule,
        layer2: SimpleModule,
    }

    impl_module! {
        for TwoLayerNet {
            modules: [layer1, layer2]
        }
    }

    #[test]
    fn test_hierarchical_module() {
        let net = TwoLayerNet {
            layer1: SimpleModule {
                weight: Parameter::zeros(vec![10, 20]),
                bias: Parameter::zeros(vec![10]),
            },
            layer2: SimpleModule {
                weight: Parameter::zeros(vec![5, 10]),
                bias: Parameter::zeros(vec![5]),
            },
        };

        let params = net.named_parameters();
        assert_eq!(params.len(), 4);
        assert!(params.contains_key("layer1.weight"));
        assert!(params.contains_key("layer1.bias"));
        assert!(params.contains_key("layer2.weight"));
        assert!(params.contains_key("layer2.bias"));
    }

    // パラメータとモジュールの混在
    struct MixedModule {
        direct_param: Parameter,
        sub_module: SimpleModule,
    }

    impl_module! {
        for MixedModule {
            parameters: [direct_param],
            modules: [sub_module]
        }
    }

    #[test]
    fn test_mixed_module() {
        let module = MixedModule {
            direct_param: Parameter::zeros(vec![5]),
            sub_module: SimpleModule {
                weight: Parameter::zeros(vec![10, 20]),
                bias: Parameter::zeros(vec![10]),
            },
        };

        let params = module.named_parameters();
        assert_eq!(params.len(), 3);
        assert!(params.contains_key("direct_param"));
        assert!(params.contains_key("sub_module.weight"));
        assert!(params.contains_key("sub_module.bias"));
    }

    // 逆順（modules, parametersの順）でもOK
    struct ReversedModule {
        direct_param: Parameter,
        sub_module: SimpleModule,
    }

    impl_module! {
        for ReversedModule {
            modules: [sub_module],
            parameters: [direct_param]
        }
    }

    #[test]
    fn test_reversed_order() {
        let module = ReversedModule {
            direct_param: Parameter::zeros(vec![5]),
            sub_module: SimpleModule {
                weight: Parameter::zeros(vec![10, 20]),
                bias: Parameter::zeros(vec![10]),
            },
        };

        let params = module.named_parameters();
        assert_eq!(params.len(), 3);
        assert!(params.contains_key("direct_param"));
        assert!(params.contains_key("sub_module.weight"));
        assert!(params.contains_key("sub_module.bias"));
    }

    #[test]
    fn test_num_parameters_with_macro() {
        let module = SimpleModule {
            weight: Parameter::zeros(vec![10, 20]),
            bias: Parameter::zeros(vec![10]),
        };

        let num = module.num_parameters();
        assert_eq!(num, 10 * 20 + 10);
    }
}
