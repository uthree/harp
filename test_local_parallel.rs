use harp::ast::{AstNode, DType, Literal, Mutability, Scope, VarDecl, VarKind};
use harp::opt::ast::suggesters::{LocalParallelizationSuggester, GroupParallelizationSuggester};
use harp::opt::ast::AstSuggester;

fn const_int(v: isize) -> AstNode {
    AstNode::Const(Literal::Int(v))
}

fn var(name: &str) -> AstNode {
    AstNode::Var(name.to_string())
}

fn make_simple_function() -> AstNode {
    let body = AstNode::Store {
        ptr: Box::new(var("output")),
        offset: Box::new(var("i")),
        value: Box::new(AstNode::Load {
            ptr: Box::new(var("input")),
            offset: Box::new(var("i")),
            count: 1,
            dtype: DType::F32,
        }),
    };

    let range = AstNode::Range {
        var: "i".to_string(),
        start: Box::new(const_int(0)),
        step: Box::new(const_int(1)),
        stop: Box::new(var("N")),
        body: Box::new(body),
    };

    AstNode::Function {
        name: Some("kernel_0".to_string()),
        params: vec![
            VarDecl {
                name: "input".to_string(),
                dtype: DType::Ptr(Box::new(DType::F32)),
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            },
            VarDecl {
                name: "output".to_string(),
                dtype: DType::Ptr(Box::new(DType::F32)),
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            },
            VarDecl {
                name: "N".to_string(),
                dtype: DType::Int,
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            },
        ],
        return_type: DType::Tuple(vec![]),
        body: Box::new(range),
    }
}

fn main() {
    let func = make_simple_function();
    
    // LocalParallelizationSuggesterでの結果
    let local_suggester = LocalParallelizationSuggester::new();
    let local_results = local_suggester.suggest(&func);
    
    println!("=== LocalParallelizationSuggester Results ===");
    println!("Number of results: {}", local_results.len());
    
    for (i, result) in local_results.iter().enumerate() {
        println!("\nResult {}:", i);
        println!("  suggester_name: {}", result.suggester_name);
        println!("  description: {}", result.description);
        
        if let AstNode::Kernel { params, default_grid_size, default_thread_group_size, .. } = &result.ast {
            println!("  params:");
            for p in params {
                println!("    - {} : {:?} (kind: {:?})", p.name, p.dtype, p.kind);
            }
            println!("  default_grid_size: [{:?}, {:?}, {:?}]", 
                     default_grid_size[0], default_grid_size[1], default_grid_size[2]);
            println!("  default_thread_group_size: [{:?}, {:?}, {:?}]", 
                     default_thread_group_size[0], default_thread_group_size[1], default_thread_group_size[2]);
        }
    }
    
    // GroupParallelizationSuggesterでの結果
    let group_suggester = GroupParallelizationSuggester::new();
    let group_results = group_suggester.suggest(&func);
    
    println!("\n=== GroupParallelizationSuggester Results ===");
    println!("Number of results: {}", group_results.len());
    
    for (i, result) in group_results.iter().enumerate() {
        println!("\nResult {}:", i);
        println!("  suggester_name: {}", result.suggester_name);
        println!("  description: {}", result.description);
        
        if let AstNode::Kernel { params, default_grid_size, default_thread_group_size, .. } = &result.ast {
            println!("  params:");
            for p in params {
                println!("    - {} : {:?} (kind: {:?})", p.name, p.dtype, p.kind);
            }
            println!("  default_grid_size: [{:?}, {:?}, {:?}]", 
                     default_grid_size[0], default_grid_size[1], default_grid_size[2]);
            println!("  default_thread_group_size: [{:?}, {:?}, {:?}]", 
                     default_thread_group_size[0], default_thread_group_size[1], default_thread_group_size[2]);
        }
    }
}
