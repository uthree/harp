//! Derive macro integration tests

use harp_nn::{DeriveModule, Module, Parameter, impl_module};

// 簡単なモジュール（impl_module!マクロ使用）
struct SimpleModule {
    weight: Parameter,
    bias: Parameter,
}

impl_module! {
    for SimpleModule {
        parameters: [weight, bias]
    }
}

// Derive macroのテスト
#[derive(DeriveModule)]
struct DeriveSimpleModule {
    weight: Parameter,
    bias: Parameter,
}

#[test]
fn test_derive_simple_module() {
    let module = DeriveSimpleModule {
        weight: Parameter::zeros(vec![10, 20]),
        bias: Parameter::zeros(vec![10]),
    };

    let params = module.named_parameters();
    assert_eq!(params.len(), 2);
    assert!(params.contains_key("weight"));
    assert!(params.contains_key("bias"));
}

#[derive(DeriveModule)]
struct DeriveHierarchical {
    layer1: SimpleModule,
    layer2: SimpleModule,
}

#[test]
fn test_derive_hierarchical() {
    let module = DeriveHierarchical {
        layer1: SimpleModule {
            weight: Parameter::zeros(vec![10, 20]),
            bias: Parameter::zeros(vec![10]),
        },
        layer2: SimpleModule {
            weight: Parameter::zeros(vec![5, 10]),
            bias: Parameter::zeros(vec![5]),
        },
    };

    let params = module.named_parameters();
    assert_eq!(params.len(), 4);
    assert!(params.contains_key("layer1.weight"));
    assert!(params.contains_key("layer1.bias"));
    assert!(params.contains_key("layer2.weight"));
    assert!(params.contains_key("layer2.bias"));
}

#[derive(DeriveModule)]
struct DeriveMixed {
    direct_param: Parameter,
    sub_module: SimpleModule,
}

#[test]
fn test_derive_mixed() {
    let module = DeriveMixed {
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
fn test_derive_num_parameters() {
    let module = DeriveSimpleModule {
        weight: Parameter::zeros(vec![10, 20]),
        bias: Parameter::zeros(vec![10]),
    };

    let num = module.num_parameters();
    assert_eq!(num, 10 * 20 + 10);
}

#[test]
fn test_derive_parameters_mut() {
    let mut module = DeriveSimpleModule {
        weight: Parameter::zeros(vec![10, 20]),
        bias: Parameter::zeros(vec![10]),
    };

    let params_mut = module.named_parameters_mut();
    assert_eq!(params_mut.len(), 2);
    assert!(params_mut.contains_key("weight"));
    assert!(params_mut.contains_key("bias"));
}
