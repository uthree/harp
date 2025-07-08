use crate::graph::graph::Graph;
use crate::ir::{Function, Instruction, Kernel};
use petgraph::{algo::toposort, graph::NodeIndex};
use std::collections::HashMap;

/// Compiles a computation graph into an executable IR function.
pub struct Compiler {
    vreg_map: HashMap<NodeIndex, usize>,
    vreg_counter: usize,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            vreg_map: HashMap::new(),
            vreg_counter: 0,
        }
    }

    fn new_vreg(&mut self) -> usize {
        let vreg = self.vreg_counter;
        self.vreg_counter += 1;
        vreg
    }

    /// Compiles the given graph into a `Function`.
    ///
    // # Arguments
    ///
    /// * `graph` - The computation graph to compile.
    /// * `name` - The name for the resulting function.
    pub fn compile(&mut self, graph: &Graph, name: &str) -> Function {
        // 1. Topologically sort the graph nodes.
        let sorted_nodes = toposort(&graph.graph, None).unwrap();
        let mut instructions = vec![];

        // 2. Iterate through sorted nodes and generate instructions.
        for node_idx in sorted_nodes {
            let node = &graph.graph[node_idx];
            let op = node.op();

            // This is a very simplified mapping.
            // We will need to handle different operators, buffers, etc.
            if let Some(const_op) = op.as_any().downcast_ref::<crate::graph::operator::Const>() {
                let out_vreg = self.new_vreg();
                self.vreg_map.insert(node_idx, out_vreg);
                instructions.push(Instruction::Const {
                    out: out_vreg,
                    val: const_op.scalar,
                });
            }
            // TODO: Add cases for other operators (Alu, Load, Store, etc.)
        }

        // 3. Manage virtual registers and memory buffers (partially done).
        // 4. Construct and return the final Function object.

        // For now, creating a single kernel with all instructions.
        let kernel = Kernel {
            name: "main_kernel".to_string(),
            instructions,
            vregs: vec![], // TODO: Populate this with actual DTypes
            launch_dims: [1, 1, 1],
            reads: vec![],
            writes: vec![],
        };

        Function {
            name: name.to_string(),
            kernels: vec![kernel],
            buffers: vec![],
            required_memory: 0,
            args: vec![],
            ret: 0,
        }
    }
}
