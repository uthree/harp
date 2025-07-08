use crate::{
    graph::{
        graph::Graph,
        operator::{self, Input, Operator},
    },
    ir::{AluOp, Buffer, BufferId, Function, Instruction, Kernel, MemorySpace},
};
use petgraph::{algo::toposort, graph::NodeIndex, Direction, visit::EdgeRef};
use std::collections::HashMap;

/// Compiles a computation graph into an executable IR function.
pub struct Compiler {
    vreg_map: HashMap<NodeIndex, usize>,
    vreg_counter: usize,
    buffers: Vec<Buffer>,
    buffer_counter: usize,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            vreg_map: HashMap::new(),
            vreg_counter: 0,
            buffers: vec![],
            buffer_counter: 0,
        }
    }

    fn new_vreg(&mut self) -> usize {
        let vreg = self.vreg_counter;
        self.vreg_counter += 1;
        vreg
    }

    fn new_buffer(&mut self) -> BufferId {
        let buffer_id = self.buffer_counter;
        self.buffer_counter += 1;
        buffer_id
    }

    /// Compiles the given graph into a `Function`.
    pub fn compile(&mut self, graph: &Graph, name: &str) -> Function {
        let mut instructions = vec![];
        let mut args = vec![];

        // 1. Handle input nodes and create input buffers.
        for &node_idx in &graph.inputs {
            let node = &graph.graph[node_idx];
            let input_op = node.op().as_any().downcast_ref::<Input>().unwrap();
            let buffer_id = self.new_buffer();
            args.push(buffer_id);
            self.buffers.push(Buffer {
                id: buffer_id,
                offset: 0,
                size: 0, // TODO: Calculate size
                dtype: input_op.dtype,
                memory_space: MemorySpace::Host,
            });
            let out_vreg = self.new_vreg();
            self.vreg_map.insert(node_idx, out_vreg);
            instructions.push(Instruction::Load {
                out: out_vreg,
                from: buffer_id,
                shape: node.shape.clone(),
            });
        }

        // 2. Topologically sort and iterate through nodes.
        let sorted_nodes = toposort(&graph.graph, None).unwrap();
        for node_idx in sorted_nodes {
            if graph.inputs.contains(&node_idx) {
                continue;
            }

            let node = &graph.graph[node_idx];
            let op = node.op();
            let out_vreg = self.new_vreg();
            self.vreg_map.insert(node_idx, out_vreg);

            let mut parents: Vec<_> = graph
                .graph
                .edges_directed(node_idx, Direction::Incoming)
                .collect();
            // Sort parents by argument index to ensure correct order for lhs and rhs.
            parents.sort_by_key(|edge| edge.weight().arg_index);

            if let Some(const_op) = op.as_any().downcast_ref::<operator::Const>() {
                instructions.push(Instruction::Const {
                    out: out_vreg,
                    val: const_op.scalar,
                });
            } else if let Some(alu_op) = self.map_to_alu_op(op) {
                if parents.len() == 1 { // UnaryOp
                    let lhs = self.vreg_map[&parents[0].source()];
                    instructions.push(Instruction::Alu {
                        op: alu_op,
                        out: out_vreg,
                        lhs,
                        rhs: None,
                    });
                } else if parents.len() == 2 { // BinaryOp
                    let lhs = self.vreg_map[&parents[0].source()];
                    let rhs = self.vreg_map[&parents[1].source()];
                    instructions.push(Instruction::Alu {
                        op: alu_op,
                        out: out_vreg,
                        lhs,
                        rhs: Some(rhs),
                    });
                }
            }
            // TODO: Handle other operator types (Rand, Store, etc.)
        }

        let kernel = Kernel {
            name: "main_kernel".to_string(),
            instructions,
            vregs: vec![], // TODO: Populate
            launch_dims: [1, 1, 1],
            reads: args.clone(),
            writes: vec![], // TODO: Populate
        };

        Function {
            name: name.to_string(),
            kernels: vec![kernel],
            buffers: self.buffers.clone(),
            required_memory: 0, // TODO: Calculate
            args,
            ret: 0, // TODO: Populate
        }
    }

    fn map_to_alu_op(&self, op: &dyn Operator) -> Option<AluOp> {
        let op_any = op.as_any();
        match op_any {
            _ if op_any.is::<operator::Add>() => Some(AluOp::Add),
            _ if op_any.is::<operator::Mul>() => Some(AluOp::Mul),
            _ if op_any.is::<operator::Exp2>() => Some(AluOp::Exp2),
            _ if op_any.is::<operator::Log2>() => Some(AluOp::Log2),
            _ if op_any.is::<operator::Sin>() => Some(AluOp::Sin),
            _ if op_any.is::<operator::Sqrt>() => Some(AluOp::Sqrt),
            _ if op_any.is::<operator::Recip>() => Some(AluOp::Recip),
            _ if op_any.is::<operator::LessThan>() => Some(AluOp::LessThan),
            _ => None,
        }
    }
}
