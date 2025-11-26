//! Module自動実装マクロのデモ
//!
//! `impl_module!`マクロを使った簡単な例

use harp_autograd::Tensor;
use harp_nn::{Module, Parameter, impl_module};

// 1. 基本的なLinear層
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

    // forwardは通常のメソッドとして定義
    // （matmulが未実装なので、このデモでは定義のみ）
    #[allow(dead_code)]
    fn forward(&self, _input: &Tensor) -> Tensor {
        unimplemented!("matmul not yet implemented")
    }
}

// マクロでModule traitを自動実装
impl_module! {
    for Linear {
        parameters: [weight, bias]
    }
}

// 2. 2層のMLP
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

    #[allow(dead_code)]
    fn forward(&self, input: &Tensor) -> Tensor {
        // let h = self.layer1.forward(input);
        // let h = h.relu();
        // self.layer2.forward(&h)
        let _ = input;
        unimplemented!("matmul not yet implemented")
    }
}

// サブモジュールを含む場合もマクロで簡単に実装
impl_module! {
    for MLP {
        modules: [layer1, layer2]
    }
}

// 3. パラメータとサブモジュールの混在
struct CustomNet {
    // 直接のパラメータ
    scale: Parameter,
    offset: Parameter,
    // サブモジュール
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

    #[allow(dead_code)]
    fn forward(&self, input: &Tensor) -> Tensor {
        // let output = self.mlp.forward(input);
        // &output * self.scale.tensor() + self.offset.tensor()
        let _ = input;
        unimplemented!("matmul not yet implemented")
    }
}

// パラメータとモジュールの両方を指定
impl_module! {
    for CustomNet {
        parameters: [scale, offset],
        modules: [mlp]
    }
}

fn main() {
    println!("=== Module Macro Demo ===\n");

    // Linear層の作成
    println!("1. Basic Linear layer:");
    let linear = Linear::new(10, 5);
    println!("  Total parameters: {}", linear.num_parameters());
    println!("  Expected: {} (10*5 + 5 = 55)", 10 * 5 + 5);

    println!("  Named parameters:");
    for (name, param) in linear.named_parameters() {
        let shape = param.data.view.shape();
        println!("    {}: {:?}", name, shape);
    }
    println!();

    // MLPの作成
    println!("2. Two-layer MLP:");
    let mlp = MLP::new(10, 20, 5);
    println!("  Total parameters: {}", mlp.num_parameters());
    println!(
        "  Expected: {} (10*20 + 20 + 20*5 + 5 = 325)",
        10 * 20 + 20 + 20 * 5 + 5
    );

    println!("  Named parameters:");
    for (name, param) in mlp.named_parameters() {
        let shape = param.data.view.shape();
        println!("    {}: {:?}", name, shape);
    }
    println!();

    // CustomNetの作成
    println!("3. Custom network with mixed parameters:");
    let custom_net = CustomNet::new(10, 20, 5);
    println!("  Total parameters: {}", custom_net.num_parameters());
    println!(
        "  Expected: {} (5 + 5 + 325 = 335)",
        5 + 5 + 10 * 20 + 20 + 20 * 5 + 5
    );

    println!("  Named parameters:");
    for (name, param) in custom_net.named_parameters() {
        let shape = param.data.view.shape();
        println!("    {}: {:?}", name, shape);
    }
    println!();

    println!("=== Demo Complete ===");
    println!();
    println!("Benefits of impl_module! macro:");
    println!("  ✓ No boilerplate code for named_parameters()");
    println!("  ✓ No boilerplate code for named_parameters_mut()");
    println!("  ✓ Automatic hierarchical naming (e.g., mlp.layer1.weight)");
    println!("  ✓ Type-safe parameter collection");
    println!("  ✓ Easy to maintain and extend");
}
