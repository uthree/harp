# 中間表現 (IR) の設計: 実行計画 (Execution Plan)

このドキュメントでは、計算グラフをコードに変換する前段階として導入する中間表現 (IR) の設計について説明します。このIRは、より具体的な「実行計画 (Execution Plan)」として機能し、演算の厳密な実行順序やメモリ管理に関する情報を提供します。

## 導入の目的

現在の計算グラフ (`Graph`) は、演算とデータフローの依存関係を表現していますが、具体的なメモリの割り当て、解放、および演算の厳密な実行順序（トポロジカルソート以外）までは指定していません。このIRは、これらの詳細を明確にすることで、コード生成と最適化のプロセスをより効率的かつ柔軟にします。

-   **実行順序の明確化**: 演算が実行される厳密な順序を線形リストとして表現します。
-   **メモリ管理の抽象化**: 各演算がどのメモリ領域（バッファ）を読み書きするかを明示し、メモリの割り当てと解放の計画を可能にします。
-   **バックエンドへの変換容易性**: 特定のハードウェア（CPU, GPUなど）向けのコード生成において、必要な低レベルな情報を提供します。

## 設計要素

### 1. バッファ (Buffer)

計算中に割り当てられる具体的なメモリ領域を表します。

```rust
// src/ir/buffer.rs

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferId(usize); // 各バッファの一意なID

#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub id: BufferId,
    pub shape: ShapeTracker, // このバッファ内のデータの形状
    pub dtype: DType,        // データ型 (例: f32)
    // pub lifetime: BufferLifetime, // (オプション) このバッファが有効な期間
    // pub allocated_size: usize, // (オプション) 実際に割り当てられたバイト数
}

// データ型を表現するEnum
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    F32,
    I32,
    // ...
}
```

### 2. 命令 (Instruction)

各命令は、入力バッファを受け取り、出力バッファを生成する単一の具体的な操作を表します。

```rust
// src/ir/instruction.rs

use super::buffer::BufferId;
use crate::operator::Operator; // 既存のOperatorを再利用

#[derive(Debug)]
pub struct Instruction {
    pub op: Box<dyn Operator>, // 実際の演算 (Add, Mul, Exp2など)
    pub inputs: Vec<BufferId>, // 入力バッファのIDリスト
    pub outputs: Vec<BufferId>, // 出力バッファのIDリスト (通常は1つ)
    // pub debug_name: String, // (オプション) デバッグ/可視化のための名前
}
```

### 3. 実行計画 (ExecutionPlan)

主要なIR構造体であり、命令のシーケンスと、プラン内で使用される全てのバッファのレジストリを含みます。

```rust
// src/ir/execution_plan.rs

use super::instruction::Instruction;
use super::buffer::{BufferId, BufferInfo, DType};
use crate::{graph::Graph, operator::Operator, shape::tracker::ShapeTracker};
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

pub struct ExecutionPlan {
    pub instructions: Vec<Instruction>,
    pub buffers: HashMap<BufferId, BufferInfo>, // プランで使用される全てのバッファ
    pub input_buffers: Vec<BufferId>, // プラン全体の入力となるバッファ
    pub output_buffers: Vec<BufferId>, // プラン全体の出力となるバッファ
}

impl ExecutionPlan {
    /// 計算グラフ (Graph) から実行計画 (ExecutionPlan) を構築する
    pub fn from_graph(graph: &Graph) -> Self {
        let mut instructions = Vec::new();
        let mut buffers = HashMap::new();
        let mut input_buffers = Vec::new();
        let mut output_buffers = Vec::new();

        let mut node_to_buffer_map: HashMap<NodeIndex, BufferId> = HashMap::new();
        let mut next_buffer_id = 0;

        // 1. Graphをトポロジカルソートして実行順序を決定
        let sorted_nodes = toposort(&graph.graph, None)
            .expect("Graph should be a DAG for topological sort");

        for node_idx in sorted_nodes {
            let node = graph.graph.node_weight(node_idx).unwrap();

            let mut instruction_inputs = Vec::new();
            for (parent_idx, _edge_metadata) in graph.parents(node_idx) {
                if let Some(buffer_id) = node_to_buffer_map.get(&parent_idx) {
                    instruction_inputs.push(*buffer_id);
                } else {
                    // Graphの入力ノードの場合の処理
                }
            }

            let output_buffer_id = BufferId(next_buffer_id);
            next_buffer_id += 1;

            let output_buffer_info = BufferInfo {
                id: output_buffer_id,
                shape: node.shape.clone(),
                dtype: DType::F32, // 仮定
            };
            buffers.insert(output_buffer_id, output_buffer_info);
            node_to_buffer_map.insert(node_idx, output_buffer_id);

            let instruction = Instruction {
                op: node.op().clone_box(),
                inputs: instruction_inputs,
                outputs: vec![output_buffer_id],
            };
            instructions.push(instruction);

            if graph.inputs.contains(&node_idx) {
                input_buffers.push(output_buffer_id);
            }
            if graph.outputs.contains(&node_idx) {
                output_buffers.push(output_buffer_id);
            }
        }

        ExecutionPlan {
            instructions,
            buffers,
            input_buffers,
            output_buffers,
        }
    }
}
```

### `Operator` トレイトの変更

`Box<dyn Operator>` をクローンできるようにするため、`Operator` トレイトに `clone_box` メソッドを追加し、既存の全ての `Operator` 実装に `#[derive(Clone)]` と `clone_box` の実装が必要です。

```rust
// src/operator.rs (既存のファイルに追記)

pub trait Operator: fmt::Debug + Any + Send + Sync {
    fn name(&self) -> &str;
    fn as_any(&self) -> &dyn Any;
    fn clone_box(&self) -> Box<dyn Operator>; // この行を追加
}

// 各Operatorの実装例
// #[derive(Debug, Clone)]
// pub struct Add;
// impl Operator for Add {
//     // ... 既存のメソッド ...
//     fn clone_box(&self) -> Box<dyn Operator> {
//         Box::new(self.clone())
//     }
// }
```

## 実装の次のステップ

1.  `src/ir/` ディレクトリを作成し、`buffer.rs`, `instruction.rs`, `execution_plan.rs` を上記の定義で作成します。
2.  `src/operator.rs` を修正し、`Operator` トレイトに `clone_box` を追加し、既存の全ての `Operator` 構造体に `#[derive(Clone)]` を追加して `clone_box` を実装します。
3.  `ExecutionPlan::from_graph` メソッドを実装します。特に、`Graph` の入力ノードから `BufferId` を適切に割り当てるロジックを考慮します。

この設計により、計算グラフの抽象的な表現と、具体的な実行計画の間の明確な分離が実現され、将来的な最適化や異なるバックエンドへのコード生成が容易になります。
