use crate::{
    ast::AstNode,
    backend::{Backend, Buffer, Compiler, Kernel, Renderer},
    graph::{Graph, GraphSignature},
    opt::{
        graph::GraphOptimizer,
        ast::{
            heuristic::{
                beam_search::BeamSearchAstOptimizer, handcode::HandcodedCostEstimator,
                rule_based_suggester::RuleBasedRewriteSuggester,
            },
            rule::{
                algebraic_simplification, associative_rules, commutative_rules, distributive_rules,
                factorization_rule,
            },
        },
        graph::fusion::ElementwiseFusion,
        CombinedGraphOptimizer,
    },
};
use std::collections::HashMap;
pub struct GenericBackend<C, R, B>
where
    R: Renderer,
    C: Compiler<B, CodeRepr = R::CodeRepr>,
    B: Buffer,
{
    compiler: C,
    renderer: R,
    cache: HashMap<Graph, C::KernelType>,
    graph_optimizer: CombinedGraphOptimizer,
    ast_optimizer: BeamSearchAstOptimizer<RuleBasedRewriteSuggester, HandcodedCostEstimator>,
}

impl<C, R, B> GenericBackend<C, R, B>
where
    R: Renderer,
    C: Compiler<B, CodeRepr = R::CodeRepr>,
    B: Buffer,
{
    pub fn with_options(&mut self, options: (R::Option, C::Option)) -> &mut Self {
        let (renderer_option, compiler_option) = options;
        self.compiler.with_option(compiler_option);
        self.renderer.with_option(renderer_option);
        self
    }

    pub fn compile(&mut self, graph: &Graph) -> C::KernelType {
        let optimized_graph = self.graph_optimizer.optimize(graph);
        let ast = crate::lowerer::lower_graph(&optimized_graph);
        let optimized_ast = self.ast_optimizer.optimize(&ast);
        let code = self.renderer.render(optimized_ast);
        self.compiler.compile(&code, graph.signature.clone())
    }
}

impl<C, R, B> Backend<B> for GenericBackend<C, R, B>
where
    R: Renderer,
    C: Compiler<B, CodeRepr = R::CodeRepr>,
    B: Buffer,
{
    type Option = (R::Option, C::Option);

    fn new() -> Self {
        Self {
            compiler: C::new(),
            renderer: R::new(),
            cache: HashMap::new(),
            graph_optimizer: CombinedGraphOptimizer::new(vec![Box::new(ElementwiseFusion)]),
            ast_optimizer: BeamSearchAstOptimizer::new(
                RuleBasedRewriteSuggester::new(
                    algebraic_simplification() // 不要なノードの除去と定数項の事前計算
                        + commutative_rules() // 交換法則
                        + distributive_rules() // 分配法則
                        + associative_rules() // 結合法則
                        + factorization_rule(), // 因数分解
                ),
                HandcodedCostEstimator,
            ),
        }
    }

    fn with_option(&mut self, option: Self::Option) {
        self.with_options(option);
    }

    fn is_available(&self) -> bool {
        self.compiler.is_available()
    }

    fn execute(&mut self, graph: &Graph, inputs: Vec<B>) -> Vec<B> {
        if !self.cache.contains_key(graph) {
            let kernel = self.compile(graph);
            self.cache.insert(graph.clone(), kernel);
        }
        let kernel = self.cache.get_mut(graph).unwrap();

        // Prepare output buffers
        let output_buffers: Vec<B> = graph
            .signature
            .outputs
            .iter()
            .map(|sig| {
                let shape_usize = sig
                    .shape
                    .iter()
                    .map(|e| e.evaluate(&HashMap::new()) as usize) // Assumes no shape vars for now
                    .collect();
                B::allocate(sig.dtype.clone(), shape_usize)
            })
            .collect();

        let mut all_buffers = output_buffers;
        all_buffers.extend(inputs);

        kernel.call(all_buffers, &[]) // Assumes no shape vars for now
    }
}

impl<C, R, B> Default for GenericBackend<C, R, B>
where
    R: Renderer,
    C: Compiler<B, CodeRepr = R::CodeRepr>,
    B: Buffer,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::DType,
        backend::{Backend, Buffer, Kernel, c::CBackend},
        graph::{Graph, TensorSignature, shape::expr::Expr as ShapeExpr},
    };

    #[test]
    fn test_generic_backend_with_c() {
        // 1. Build a graph
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(1)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);
        let c = a + b;
        graph.outputs.push(c);
        graph.signature.outputs.push(TensorSignature {
            dtype: dtype.clone(),
            shape: shape.clone(),
        });

        // 2. Use GenericBackend (via CBackend) to compile
        let mut backend = CBackend::new();
        let mut kernel = backend.compile(&graph);

        // 3. Prepare buffers and run the kernel
        let a_data = vec![1.0f32];
        let b_data = vec![2.0f32];
        let shape_usize = vec![1];

        let a_buffer = crate::backend::c::CBuffer::from_slice(&a_data, &shape_usize, dtype.clone());
        let b_buffer = crate::backend::c::CBuffer::from_slice(&b_data, &shape_usize, dtype.clone());
        let out_buffer = crate::backend::c::CBuffer::allocate(dtype, shape_usize);

        let buffers = vec![out_buffer, a_buffer, b_buffer];
        let result_buffers = kernel.call(buffers, &[]);

        // 4. Check the result
        let result_data = result_buffers[0].to_vec::<f32>();
        assert_eq!(result_data, vec![3.0f32]);
    }
}

pub use crate::backend::c::CBackend;