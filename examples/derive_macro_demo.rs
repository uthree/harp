//! Derive macro デモ
//!
//! `#[derive(Module)]`を使った自動実装の例

use harp::prelude::*;
use harp_nn::{Module, Parameter};

// 1. 基本的なLinear層（deriveマクロ使用）
#[derive(DeriveModule)]
struct Linear {
    weight: Parameter,
    bias: Parameter,
}

impl Linear {
    fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight: Parameter::zeros(vec![out_features, in_features]),
            bias: Parameter::zeros(vec![out_features]),
        }
    }
}

// 2. MLPもderiveで簡単に
#[derive(DeriveModule)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    layer1: Linear,
    layer2: Linear,
}

impl MLP {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            layer1: Linear::new(input_size, hidden_size),
            layer2: Linear::new(hidden_size, output_size),
        }
    }
}

// 3. 混在する場合もOK
#[derive(DeriveModule)]
struct CustomNet {
    scale: Parameter,
    offset: Parameter,
    mlp: MLP,
}

impl CustomNet {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            scale: Parameter::ones(vec![output_size]),
            offset: Parameter::zeros(vec![output_size]),
            mlp: MLP::new(input_size, hidden_size, output_size),
        }
    }
}

fn main() {
    println!("=== Derive Macro Demo ===\n");

    // Linear層の作成
    println!("1. Linear layer with #[derive(Module)]:");
    let linear = Linear::new(10, 5);
    println!("  Total parameters: {}", linear.num_parameters());

    println!("  Named parameters:");
    for (name, param) in linear.named_parameters() {
        let shape = param.data.view.shape();
        println!("    {}: {:?}", name, shape);
    }
    println!();

    // MLPの作成
    println!("2. MLP with automatic nested parameter detection:");
    let mlp = MLP::new(10, 20, 5);
    println!("  Total parameters: {}", mlp.num_parameters());

    println!("  Named parameters:");
    for (name, param) in mlp.named_parameters() {
        let shape = param.data.view.shape();
        println!("    {}: {:?}", name, shape);
    }
    println!();

    // CustomNetの作成
    println!("3. CustomNet with mixed parameters:");
    let custom_net = CustomNet::new(10, 20, 5);
    println!("  Total parameters: {}", custom_net.num_parameters());

    println!("  Named parameters:");
    for (name, param) in custom_net.named_parameters() {
        let shape = param.data.view.shape();
        println!("    {}: {:?}", name, shape);
    }
    println!();

    println!("=== Demo Complete ===");
    println!();
    println!("Benefits of #[derive(Module)]:");
    println!("  ✓ Just add #[derive(Module)] to your struct");
    println!("  ✓ Automatic parameter detection");
    println!("  ✓ Automatic nested module support");
    println!("  ✓ Zero boilerplate code");
    println!("  ✓ Type-safe and clean");
}
