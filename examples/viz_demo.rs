//! 最適化履歴可視化デモ
//!
//! # 実行方法
//! ```bash
//! cargo run --features viz --example viz_demo
//! ```
//!
//! # 操作方法
//! - ←/h: 前のステップへ
//! - →/l: 次のステップへ
//! - ↑/k: 前の候補を選択
//! - ↓/j: 次の候補を選択
//! - q/Esc: 終了

use harp::ast::scope::{Mutability, Scope, VarDecl, VarKind};
use harp::ast::{AstNode, DType, Literal};
use harp::opt::ast::history::{AlternativeCandidate, OptimizationHistory, OptimizationSnapshot};

fn main() {
    // サンプルの最適化履歴を作成
    let history = create_sample_history();

    println!("Starting optimization history viewer...");
    println!("Press q or Esc to quit.");

    // 可視化を起動
    if let Err(e) = harp::viz::run(history) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// パラメータ用のVarDeclを作成するヘルパー
fn make_param(name: &str, dtype: DType) -> VarDecl {
    VarDecl {
        name: name.to_string(),
        dtype,
        mutability: Mutability::Immutable,
        kind: VarKind::Normal,
    }
}

/// Box<AstNode>を作成するヘルパー
fn boxed_const(val: i32) -> Box<AstNode> {
    Box::new(AstNode::Const(Literal::I32(val)))
}

/// サンプルの最適化履歴を作成
fn create_sample_history() -> OptimizationHistory {
    let mut history = OptimizationHistory::new();

    // Step 0: 初期状態
    let initial_ast = create_initial_ast();
    let snapshot0 = OptimizationSnapshot::with_alternatives(
        0,
        initial_ast.clone(),
        100.0,
        "Initial program".to_string(),
        0,
        None,
        vec![],
        1,
        None,
        vec![],
        vec![],
    );
    history.add_snapshot(snapshot0);

    // Step 1: ループ最適化
    let step1_ast = create_step1_ast();
    let step1_alt1 = AlternativeCandidate {
        ast: create_step1_alt_ast("matmul_tiled_8x8"),
        cost: 85.0,
        suggester_name: Some("LoopTilingSuggester".to_string()),
        description: "Tile 8x8".to_string(),
        rank: 1,
    };
    let step1_alt2 = AlternativeCandidate {
        ast: create_step1_alt_ast("matmul_interchanged"),
        cost: 90.0,
        suggester_name: Some("LoopInterchangeSuggester".to_string()),
        description: "Interchange loops i,j".to_string(),
        rank: 2,
    };

    let snapshot1 = OptimizationSnapshot::with_alternatives(
        1,
        step1_ast,
        80.0,
        "Loop tiling 4x4".to_string(),
        0,
        Some("LoopTilingSuggester".to_string()),
        vec!["Applied loop tiling with tile size 4x4".to_string()],
        3,
        Some("LoopTilingSuggester".to_string()),
        vec![step1_alt1, step1_alt2],
        vec![("LoopTilingSuggester".to_string(), "Tile 4x4".to_string())],
    );
    history.add_snapshot(snapshot1);

    // Step 2: 並列化
    let step2_ast = create_step2_ast();
    let step2_alt1 = AlternativeCandidate {
        ast: create_kernel_ast("matmul_local_parallel"),
        cost: 55.0,
        suggester_name: Some("LocalParallelizationSuggester".to_string()),
        description: "Local parallelization only".to_string(),
        rank: 1,
    };

    let snapshot2 = OptimizationSnapshot::with_alternatives(
        2,
        step2_ast,
        50.0,
        "GPU parallelization".to_string(),
        0,
        Some("GroupParallelizationSuggester".to_string()),
        vec!["Applied group-level parallelization".to_string()],
        2,
        Some("GroupParallelizationSuggester".to_string()),
        vec![step2_alt1],
        vec![
            ("LoopTilingSuggester".to_string(), "Tile 4x4".to_string()),
            (
                "GroupParallelizationSuggester".to_string(),
                "GPU parallelization".to_string(),
            ),
        ],
    );
    history.add_snapshot(snapshot2);

    // Step 3: ベクトル化
    let step3_ast = create_step3_ast();
    let snapshot3 = OptimizationSnapshot::with_alternatives(
        3,
        step3_ast,
        40.0,
        "Vectorization float4".to_string(),
        0,
        Some("VectorizationSuggester".to_string()),
        vec!["Applied vectorization with width 4".to_string()],
        1,
        Some("VectorizationSuggester".to_string()),
        vec![],
        vec![
            ("LoopTilingSuggester".to_string(), "Tile 4x4".to_string()),
            (
                "GroupParallelizationSuggester".to_string(),
                "GPU parallelization".to_string(),
            ),
            (
                "VectorizationSuggester".to_string(),
                "Vectorization float4".to_string(),
            ),
        ],
    );
    history.add_snapshot(snapshot3);

    // 最終パスを設定
    history.set_final_path(vec![
        ("LoopTilingSuggester".to_string(), "Tile 4x4".to_string()),
        (
            "GroupParallelizationSuggester".to_string(),
            "GPU parallelization".to_string(),
        ),
        (
            "VectorizationSuggester".to_string(),
            "Vectorization float4".to_string(),
        ),
    ]);

    history
}

// AST作成ヘルパー関数

fn create_initial_ast() -> AstNode {
    // 簡単な行列加算のAST
    AstNode::Function {
        name: Some("matmul".to_string()),
        params: vec![
            make_param("A", DType::Ptr(Box::new(DType::F32))),
            make_param("B", DType::Ptr(Box::new(DType::F32))),
            make_param("C", DType::Ptr(Box::new(DType::F32))),
        ],
        return_type: DType::Unknown,
        body: Box::new(AstNode::Block {
            statements: vec![AstNode::Range {
                var: "i".to_string(),
                start: boxed_const(0),
                stop: boxed_const(64),
                step: boxed_const(1),
                body: Box::new(AstNode::Range {
                    var: "j".to_string(),
                    start: boxed_const(0),
                    stop: boxed_const(64),
                    step: boxed_const(1),
                    body: Box::new(AstNode::Store {
                        ptr: Box::new(AstNode::Var("C".to_string())),
                        offset: Box::new(AstNode::Add(
                            Box::new(AstNode::Mul(
                                Box::new(AstNode::Var("i".to_string())),
                                boxed_const(64),
                            )),
                            Box::new(AstNode::Var("j".to_string())),
                        )),
                        value: Box::new(AstNode::Add(
                            Box::new(AstNode::Load {
                                ptr: Box::new(AstNode::Var("A".to_string())),
                                offset: Box::new(AstNode::Add(
                                    Box::new(AstNode::Mul(
                                        Box::new(AstNode::Var("i".to_string())),
                                        boxed_const(64),
                                    )),
                                    Box::new(AstNode::Var("j".to_string())),
                                )),
                                count: 1,
                                dtype: DType::F32,
                            }),
                            Box::new(AstNode::Load {
                                ptr: Box::new(AstNode::Var("B".to_string())),
                                offset: Box::new(AstNode::Add(
                                    Box::new(AstNode::Mul(
                                        Box::new(AstNode::Var("i".to_string())),
                                        boxed_const(64),
                                    )),
                                    Box::new(AstNode::Var("j".to_string())),
                                )),
                                count: 1,
                                dtype: DType::F32,
                            }),
                        )),
                    }),
                }),
            }],
            scope: Box::new(Scope::default()),
        }),
    }
}

fn create_step1_ast() -> AstNode {
    // ループタイリング後のAST
    AstNode::Function {
        name: Some("matmul_tiled".to_string()),
        params: vec![
            make_param("A", DType::Ptr(Box::new(DType::F32))),
            make_param("B", DType::Ptr(Box::new(DType::F32))),
            make_param("C", DType::Ptr(Box::new(DType::F32))),
        ],
        return_type: DType::Unknown,
        body: Box::new(AstNode::Block {
            statements: vec![AstNode::Range {
                var: "i_outer".to_string(),
                start: boxed_const(0),
                stop: boxed_const(16),
                step: boxed_const(1),
                body: Box::new(AstNode::Range {
                    var: "j_outer".to_string(),
                    start: boxed_const(0),
                    stop: boxed_const(16),
                    step: boxed_const(1),
                    body: Box::new(AstNode::Range {
                        var: "i_inner".to_string(),
                        start: boxed_const(0),
                        stop: boxed_const(4),
                        step: boxed_const(1),
                        body: Box::new(AstNode::Range {
                            var: "j_inner".to_string(),
                            start: boxed_const(0),
                            stop: boxed_const(4),
                            step: boxed_const(1),
                            body: Box::new(AstNode::Block {
                                statements: vec![
                                    AstNode::Assign {
                                        var: "i".to_string(),
                                        value: Box::new(AstNode::Add(
                                            Box::new(AstNode::Mul(
                                                Box::new(AstNode::Var("i_outer".to_string())),
                                                boxed_const(4),
                                            )),
                                            Box::new(AstNode::Var("i_inner".to_string())),
                                        )),
                                    },
                                    AstNode::Assign {
                                        var: "j".to_string(),
                                        value: Box::new(AstNode::Add(
                                            Box::new(AstNode::Mul(
                                                Box::new(AstNode::Var("j_outer".to_string())),
                                                boxed_const(4),
                                            )),
                                            Box::new(AstNode::Var("j_inner".to_string())),
                                        )),
                                    },
                                ],
                                scope: Box::new(Scope::default()),
                            }),
                        }),
                    }),
                }),
            }],
            scope: Box::new(Scope::default()),
        }),
    }
}

fn create_step1_alt_ast(name: &str) -> AstNode {
    AstNode::Function {
        name: Some(name.to_string()),
        params: vec![
            make_param("A", DType::Ptr(Box::new(DType::F32))),
            make_param("B", DType::Ptr(Box::new(DType::F32))),
            make_param("C", DType::Ptr(Box::new(DType::F32))),
        ],
        return_type: DType::Unknown,
        body: Box::new(AstNode::Block {
            statements: vec![AstNode::Range {
                var: "i".to_string(),
                start: boxed_const(0),
                stop: boxed_const(64),
                step: boxed_const(1),
                body: Box::new(AstNode::Range {
                    var: "j".to_string(),
                    start: boxed_const(0),
                    stop: boxed_const(64),
                    step: boxed_const(1),
                    body: Box::new(AstNode::Block {
                        statements: vec![],
                        scope: Box::new(Scope::default()),
                    }),
                }),
            }],
            scope: Box::new(Scope::default()),
        }),
    }
}

fn create_kernel_ast(name: &str) -> AstNode {
    AstNode::Kernel {
        name: Some(name.to_string()),
        params: vec![
            make_param("A", DType::Ptr(Box::new(DType::F32))),
            make_param("B", DType::Ptr(Box::new(DType::F32))),
            make_param("C", DType::Ptr(Box::new(DType::F32))),
        ],
        return_type: DType::Unknown,
        body: Box::new(AstNode::Block {
            statements: vec![
                AstNode::Assign {
                    var: "gid_x".to_string(),
                    value: Box::new(AstNode::Var("get_global_id(0)".to_string())),
                },
                AstNode::Assign {
                    var: "gid_y".to_string(),
                    value: Box::new(AstNode::Var("get_global_id(1)".to_string())),
                },
            ],
            scope: Box::new(Scope::default()),
        }),
        default_grid_size: [boxed_const(64), boxed_const(64), boxed_const(1)],
        default_thread_group_size: [boxed_const(4), boxed_const(4), boxed_const(1)],
    }
}

fn create_step2_ast() -> AstNode {
    create_kernel_ast("matmul_kernel")
}

fn create_step3_ast() -> AstNode {
    // ベクトル化後のAST
    AstNode::Kernel {
        name: Some("matmul_vectorized".to_string()),
        params: vec![
            make_param("A", DType::Ptr(Box::new(DType::F32))),
            make_param("B", DType::Ptr(Box::new(DType::F32))),
            make_param("C", DType::Ptr(Box::new(DType::F32))),
        ],
        return_type: DType::Unknown,
        body: Box::new(AstNode::Block {
            statements: vec![
                AstNode::Assign {
                    var: "gid_x".to_string(),
                    value: Box::new(AstNode::Var("get_global_id(0)".to_string())),
                },
                AstNode::Assign {
                    var: "gid_y".to_string(),
                    value: Box::new(AstNode::Var("get_global_id(1)".to_string())),
                },
                // float4ベクトルロード
                AstNode::Assign {
                    var: "a_vec".to_string(),
                    value: Box::new(AstNode::Load {
                        ptr: Box::new(AstNode::Var("A".to_string())),
                        offset: Box::new(AstNode::Var("offset".to_string())),
                        count: 4,
                        dtype: DType::F32,
                    }),
                },
                AstNode::Assign {
                    var: "b_vec".to_string(),
                    value: Box::new(AstNode::Load {
                        ptr: Box::new(AstNode::Var("B".to_string())),
                        offset: Box::new(AstNode::Var("offset".to_string())),
                        count: 4,
                        dtype: DType::F32,
                    }),
                },
                // ベクトル演算
                AstNode::Assign {
                    var: "c_vec".to_string(),
                    value: Box::new(AstNode::Add(
                        Box::new(AstNode::Var("a_vec".to_string())),
                        Box::new(AstNode::Var("b_vec".to_string())),
                    )),
                },
            ],
            scope: Box::new(Scope::default()),
        }),
        default_grid_size: [boxed_const(16), boxed_const(64), boxed_const(1)],
        default_thread_group_size: [boxed_const(4), boxed_const(4), boxed_const(1)],
    }
}
