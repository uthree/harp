# 並列化処理の設計 (改訂版v4)

このドキュメントはv3からの改訂版で、スレッドID変数を専用の構造体として管理する設計に変更。

## 主要な変更点（v3 → v4）

- **ThreadIdDecl構造体の導入**: スレッドID変数を専用の構造体で管理
- **より型安全な設計**: 通常の変数とスレッドID変数を明確に区別
- **3次元ベクトル方式**: スレッドIDは3要素の固定長配列として扱う（OpenCL/CUDAの実装に近い）

---

## 1. 新しい型定義

### 1.1 ThreadIdDecl構造体

```rust
/// スレッドID変数の宣言（カーネルの暗黙の引数）
/// スレッドIDは3次元ベクトル（Vec<Usize, 3>）として扱われ、[0], [1], [2]でx, y, zにアクセス
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadIdDecl {
    pub name: String,
    pub id_type: ThreadIdType,
    // dimensionフィールドは削除: 3次元をまとめて1つの変数として扱う
}

#[derive(Debug, Clone, PartialEq)]
pub enum ThreadIdType {
    GlobalId,   // global_id (グローバルスレッドID)
    LocalId,    // local_id (ワークグループ内のローカルID)
    GroupId,    // group_id (ワークグループID)
}
```

### 1.2 KernelScope構造体

```rust
/// カーネル専用のスコープ（通常のScopeを拡張）
#[derive(Debug, Clone, PartialEq)]
pub struct KernelScope {
    pub declarations: Vec<VariableDecl>,        // 通常の変数宣言
    pub thread_ids: Vec<ThreadIdDecl>,          // スレッドID変数宣言（3次元ベクトル）
}

impl KernelScope {
    /// デフォルトのスレッドID変数を持つKernelScopeを作成
    /// 各IDタイプ（Global, Local, Group）につき1つの3次元ベクトル変数を作成
    pub fn new_with_default_thread_ids() -> Self {
        Self {
            declarations: vec![],
            thread_ids: vec![
                ThreadIdDecl {
                    name: "global_id".to_string(),
                    id_type: ThreadIdType::GlobalId,
                },
                ThreadIdDecl {
                    name: "local_id".to_string(),
                    id_type: ThreadIdType::LocalId,
                },
                ThreadIdDecl {
                    name: "group_id".to_string(),
                    id_type: ThreadIdType::GroupId,
                },
            ],
        }
    }

    /// 特定の型のスレッドID変数名を取得
    pub fn get_thread_id_name(&self, id_type: &ThreadIdType) -> Option<&str> {
        self.thread_ids
            .iter()
            .find(|decl| decl.id_type == *id_type)
            .map(|decl| decl.name.as_str())
    }

    /// すべてのスレッドID変数名を収集
    pub fn all_thread_id_names(&self) -> Vec<&str> {
        self.thread_ids.iter().map(|decl| decl.name.as_str()).collect()
    }

    /// 名前衝突をチェック
    pub fn has_name_conflict(&self, name: &str) -> bool {
        self.declarations.iter().any(|decl| decl.name == name)
            || self.thread_ids.iter().any(|decl| decl.name == name)
    }

    /// スレッドIDのデータ型を取得（常にVec<Usize, 3>）
    pub fn get_thread_id_dtype() -> DType {
        DType::Vec(Box::new(DType::Usize), 3)
    }
}
```

---

## 2. Kernelノードの設計

### 2.1 基本構造

```rust
pub enum AstNode {
    // ... 既存のノード

    Kernel {
        name: String,
        scope: KernelScope,                   // 通常のScopeではなくKernelScope
        statements: Vec<AstNode>,
        arguments: Vec<(String, DType)>,
        return_type: DType,

        // 並列化情報（3次元）
        global_size: [Box<AstNode>; 3],      // [x, y, z] 総スレッド数
        local_size: [Box<AstNode>; 3],       // [x, y, z] ワークグループサイズ
    },

    CallKernel {
        name: String,
        args: Vec<AstNode>,
        global_size: [Box<AstNode>; 3],
        local_size: [Box<AstNode>; 3],
    },
}
```

### 2.2 スレッドID変数へのアクセス

```rust
// カーネル内部でのアクセス（3次元ベクトルとして）
impl AstNode {
    /// カーネル内でスレッドID変数を参照（ベクトル全体）
    pub fn kernel_thread_id(
        scope: &KernelScope,
        id_type: &ThreadIdType,
    ) -> AstNode {
        let name = scope
            .get_thread_id_name(id_type)
            .expect("Thread ID not found");
        AstNode::Var(name.to_string())
    }

    /// 特定の次元のスレッドIDにアクセス（例: global_id[0]）
    pub fn kernel_thread_id_dim(
        scope: &KernelScope,
        id_type: &ThreadIdType,
        dimension: usize,
    ) -> AstNode {
        let vec_var = Self::kernel_thread_id(scope, id_type);
        // Load { target: &global_id, index: dimension, vector_width: 1 }
        AstNode::Load {
            target: Box::new(vec_var),
            index: Box::new(AstNode::Const(ConstLiteral::Usize(dimension))),
            vector_width: 1,
        }
    }
}
```

---

## 3. 使用例

### 3.1 1D Vector Addition

```rust
Kernel {
    name: "vector_add",
    arguments: [
        ("a", Ptr(F32)),
        ("b", Ptr(F32)),
        ("c", Ptr(F32)),
        ("n", Usize)
    ],
    global_size: [Var("n"), 1.into(), 1.into()],
    local_size: [256.into(), 1.into(), 1.into()],

    scope: KernelScope {
        declarations: [],  // ユーザー定義の変数
        thread_ids: [
            ThreadIdDecl { name: "global_id", id_type: GlobalId },  // 3次元ベクトル
            ThreadIdDecl { name: "local_id", id_type: LocalId },
            ThreadIdDecl { name: "group_id", id_type: GroupId },
        ]
    },

    statements: [
        // global_id[0] を使用（x次元のスレッドID）
        Select {
            cond: LessThan(
                Load { target: Var("global_id"), index: 0.into(), vector_width: 1 },  // global_id[0]
                Var("n")
            ),
            true_val: Store {
                target: Var("c"),
                index: Load { target: Var("global_id"), index: 0.into(), vector_width: 1 },
                value: Add(
                    Load {
                        target: Var("a"),
                        index: Load { target: Var("global_id"), index: 0.into(), vector_width: 1 },
                        vector_width: 1
                    },
                    Load {
                        target: Var("b"),
                        index: Load { target: Var("global_id"), index: 0.into(), vector_width: 1 },
                        vector_width: 1
                    },
                ),
                vector_width: 1,
            },
            false_val: Block::empty(),
        }
    ]
}
```

### 3.2 2D Image Processing

```rust
Kernel {
    name: "image_filter",
    arguments: [
        ("input", Ptr(F32)),
        ("output", Ptr(F32)),
        ("width", Usize),
        ("height", Usize)
    ],
    global_size: [Var("width"), Var("height"), 1.into()],
    local_size: [16.into(), 16.into(), 1.into()],

    scope: KernelScope::new_with_default_thread_ids(),

    statements: [
        // global_id[0] (x) と global_id[1] (y) を使用
        If {
            cond: And(
                LessThan(
                    Load { target: Var("global_id"), index: 0.into(), vector_width: 1 },  // global_id[0]
                    Var("width")
                ),
                LessThan(
                    Load { target: Var("global_id"), index: 1.into(), vector_width: 1 },  // global_id[1]
                    Var("height")
                )
            ),
            then_body: Block {
                statements: [
                    // let index = global_id[1] * width + global_id[0];
                    Assign(
                        "index",
                        Add(
                            Mul(
                                Load { target: Var("global_id"), index: 1.into(), vector_width: 1 },
                                Var("width")
                            ),
                            Load { target: Var("global_id"), index: 0.into(), vector_width: 1 }
                        )
                    ),
                    // output[index] = process(input[index]);
                    Store {
                        target: Var("output"),
                        index: Var("index"),
                        value: CallFunction {
                            name: "process",
                            args: vec![
                                Load { target: Var("input"), index: Var("index"), vector_width: 1 }
                            ]
                        },
                        vector_width: 1,
                    }
                ]
            }
        }
    ]
}
```

---

## 4. KernelizeSuggesterの実装

```rust
pub struct KernelizeSuggester;

impl RewriteSuggester for KernelizeSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        if let AstNode::Function { name, scope, statements, arguments, return_type } = node {
            // FunctionのScopeをKernelScopeに変換
            let kernel_scope = KernelScope {
                declarations: scope.declarations.clone(),
                thread_ids: Self::create_default_thread_ids(),
            };

            // 名前衝突をチェックして必要なら変数名を変更
            let kernel_scope = Self::resolve_name_conflicts(kernel_scope, statements);

            vec![AstNode::Kernel {
                name: name.clone(),
                scope: kernel_scope,
                statements: statements.clone(),
                arguments: arguments.clone(),
                return_type: return_type.clone(),
                global_size: [
                    Box::new(1usize.into()),
                    Box::new(1usize.into()),
                    Box::new(1usize.into()),
                ],
                local_size: [
                    Box::new(1usize.into()),
                    Box::new(1usize.into()),
                    Box::new(1usize.into()),
                ],
            }]
        } else {
            vec![]
        }
    }
}

impl KernelizeSuggester {
    fn create_default_thread_ids() -> Vec<ThreadIdDecl> {
        vec![
            // Global IDs
            ThreadIdDecl {
                name: "gid_x".to_string(),
                id_type: ThreadIdType::GlobalId,
                dimension: 0,
            },
            ThreadIdDecl {
                name: "gid_y".to_string(),
                id_type: ThreadIdType::GlobalId,
                dimension: 1,
            },
            ThreadIdDecl {
                name: "gid_z".to_string(),
                id_type: ThreadIdType::GlobalId,
                dimension: 2,
            },
            // Local IDs
            ThreadIdDecl {
                name: "lid_x".to_string(),
                id_type: ThreadIdType::LocalId,
                dimension: 0,
            },
            ThreadIdDecl {
                name: "lid_y".to_string(),
                id_type: ThreadIdType::LocalId,
                dimension: 1,
            },
            ThreadIdDecl {
                name: "lid_z".to_string(),
                id_type: ThreadIdType::LocalId,
                dimension: 2,
            },
            // Group IDs
            ThreadIdDecl {
                name: "group_x".to_string(),
                id_type: ThreadIdType::GroupId,
                dimension: 0,
            },
            ThreadIdDecl {
                name: "group_y".to_string(),
                id_type: ThreadIdType::GroupId,
                dimension: 1,
            },
            ThreadIdDecl {
                name: "group_z".to_string(),
                id_type: ThreadIdType::GroupId,
                dimension: 2,
            },
        ]
    }

    fn resolve_name_conflicts(
        mut kernel_scope: KernelScope,
        statements: &[AstNode],
    ) -> KernelScope {
        // ユーザー定義の変数名を収集
        let user_defined_names = Self::collect_user_variable_names(statements);

        // スレッドID変数名が衝突している場合はプレフィックスを追加
        for thread_id in &mut kernel_scope.thread_ids {
            while user_defined_names.contains(&thread_id.name.as_str()) {
                thread_id.name = format!("__{}", thread_id.name);
            }
        }

        kernel_scope
    }

    fn collect_user_variable_names(statements: &[AstNode]) -> std::collections::HashSet<String> {
        // ASTを走査してユーザー定義の変数名を収集
        let mut names = std::collections::HashSet::new();
        // ... 実装
        names
    }
}
```

---

## 5. Rendererでの実装

### 5.1 CRenderer (OpenCL/CUDA風)

```rust
impl CRenderer {
    fn render_kernel(&mut self, node: &AstNode) -> String {
        if let AstNode::Kernel {
            name,
            scope,
            statements,
            arguments,
            return_type,
            ..
        } = node {
            let mut buffer = String::new();

            // 関数シグネチャ
            let args_str = arguments
                .iter()
                .map(|(arg_name, dtype)| {
                    let (base_type, array_dims) = Self::render_dtype_recursive(dtype);
                    format!("{} {}{}", base_type, arg_name, array_dims)
                })
                .collect::<Vec<_>>()
                .join(", ");

            writeln!(buffer, "__kernel void {}({})", name, args_str).unwrap();
            writeln!(buffer, "{{").unwrap();

            // スレッドID変数を定義
            for thread_id in &scope.thread_ids {
                let builtin_func = match thread_id.id_type {
                    ThreadIdType::GlobalId => "get_global_id",
                    ThreadIdType::LocalId => "get_local_id",
                    ThreadIdType::GroupId => "get_group_id",
                };
                writeln!(
                    buffer,
                    "  const size_t {} = {}({});",
                    thread_id.name, builtin_func, thread_id.dimension
                )
                .unwrap();
            }

            // 通常の変数宣言
            for decl in &scope.declarations {
                writeln!(buffer, "  {};", self.render_variable_decl(decl)).unwrap();
            }

            // 本体をレンダリング
            for stmt in statements {
                writeln!(buffer, "  {};", self.render_node(stmt)).unwrap();
            }

            writeln!(buffer, "}}").unwrap();

            buffer
        } else {
            panic!("Expected Kernel node");
        }
    }
}
```

### 5.2 CRenderer (OpenMP)

```rust
impl CRenderer {
    fn render_kernel_openmp(&mut self, node: &AstNode) -> String {
        if let AstNode::Kernel {
            name,
            scope,
            statements,
            arguments,
            global_size,
            ..
        } = node {
            let mut buffer = String::new();

            // 関数シグネチャ
            let args_str = arguments
                .iter()
                .map(|(arg_name, dtype)| {
                    let (base_type, array_dims) = Self::render_dtype_recursive(dtype);
                    format!("{} {}{}", base_type, arg_name, array_dims)
                })
                .collect::<Vec<_>>()
                .join(", ");

            writeln!(buffer, "void {}({})", name, args_str).unwrap();
            writeln!(buffer, "{{").unwrap();

            // global_sizeを評価
            let gsize_x = self.render_node(&global_size[0]);
            let gsize_y = self.render_node(&global_size[1]);
            let gsize_z = self.render_node(&global_size[2]);

            // Global IDに対応するループ変数名を取得
            let gid_x = scope.get_thread_id_name(ThreadIdType::GlobalId, 0)
                .unwrap_or("gid_x");
            let gid_y = scope.get_thread_id_name(ThreadIdType::GlobalId, 1)
                .unwrap_or("gid_y");
            let gid_z = scope.get_thread_id_name(ThreadIdType::GlobalId, 2)
                .unwrap_or("gid_z");

            // 3次元ループで展開
            writeln!(buffer, "#pragma omp parallel for collapse(3)").unwrap();
            writeln!(buffer, "  for (size_t {} = 0; {} < {}; {}++)", gid_z, gid_z, gsize_z, gid_z).unwrap();
            writeln!(buffer, "    for (size_t {} = 0; {} < {}; {}++)", gid_y, gid_y, gsize_y, gid_y).unwrap();
            writeln!(buffer, "      for (size_t {} = 0; {} < {}; {}++)", gid_x, gid_x, gsize_x, gid_x).unwrap();
            writeln!(buffer, "      {{").unwrap();

            // Local ID と Group ID は常に0（OpenMPでは意味がない）
            for thread_id in &scope.thread_ids {
                if matches!(thread_id.id_type, ThreadIdType::LocalId | ThreadIdType::GroupId) {
                    writeln!(buffer, "        const size_t {} = 0;", thread_id.name).unwrap();
                }
            }

            // 通常の変数宣言
            for decl in &scope.declarations {
                writeln!(buffer, "        {};", self.render_variable_decl(decl)).unwrap();
            }

            // 本体をレンダリング
            for stmt in statements {
                writeln!(buffer, "        {};", self.render_node(stmt)).unwrap();
            }

            writeln!(buffer, "      }}").unwrap();
            writeln!(buffer, "}}").unwrap();

            buffer
        } else {
            panic!("Expected Kernel node");
        }
    }
}
```

---

## 6. ヘルパー関数

```rust
// src/ast/helper.rs

/// Create a Kernel node with default thread IDs
pub fn kernel(
    name: impl Into<String>,
    arguments: Vec<(String, DType)>,
    return_type: DType,
    declarations: Vec<VariableDecl>,
    statements: Vec<AstNode>,
    global_size: [AstNode; 3],
    local_size: [AstNode; 3],
) -> AstNode {
    AstNode::Kernel {
        name: name.into(),
        scope: KernelScope {
            declarations,
            thread_ids: vec![
                ThreadIdDecl {
                    name: "gid_x".to_string(),
                    id_type: ThreadIdType::GlobalId,
                    dimension: 0,
                },
                ThreadIdDecl {
                    name: "gid_y".to_string(),
                    id_type: ThreadIdType::GlobalId,
                    dimension: 1,
                },
                ThreadIdDecl {
                    name: "gid_z".to_string(),
                    id_type: ThreadIdType::GlobalId,
                    dimension: 2,
                },
                ThreadIdDecl {
                    name: "lid_x".to_string(),
                    id_type: ThreadIdType::LocalId,
                    dimension: 0,
                },
                ThreadIdDecl {
                    name: "lid_y".to_string(),
                    id_type: ThreadIdType::LocalId,
                    dimension: 1,
                },
                ThreadIdDecl {
                    name: "lid_z".to_string(),
                    id_type: ThreadIdType::LocalId,
                    dimension: 2,
                },
                ThreadIdDecl {
                    name: "group_x".to_string(),
                    id_type: ThreadIdType::GroupId,
                    dimension: 0,
                },
                ThreadIdDecl {
                    name: "group_y".to_string(),
                    id_type: ThreadIdType::GroupId,
                    dimension: 1,
                },
                ThreadIdDecl {
                    name: "group_z".to_string(),
                    id_type: ThreadIdType::GroupId,
                    dimension: 2,
                },
            ],
        },
        statements,
        arguments,
        return_type,
        global_size: [
            Box::new(global_size[0]),
            Box::new(global_size[1]),
            Box::new(global_size[2]),
        ],
        local_size: [
            Box::new(local_size[0]),
            Box::new(local_size[1]),
            Box::new(local_size[2]),
        ],
    }
}

/// Create a CallKernel node
pub fn call_kernel(
    name: impl Into<String>,
    args: Vec<AstNode>,
    global_size: [AstNode; 3],
    local_size: [AstNode; 3],
) -> AstNode {
    AstNode::CallKernel {
        name: name.into(),
        args,
        global_size: [
            Box::new(global_size[0]),
            Box::new(global_size[1]),
            Box::new(global_size[2]),
        ],
        local_size: [
            Box::new(local_size[0]),
            Box::new(local_size[1]),
            Box::new(local_size[2]),
        ],
    }
}
```

---

## 7. メリット

### v3 → v4 の改善点

✅ **型安全**: `ThreadIdDecl`で型を明示的に管理
✅ **明確な構造**: `KernelScope`で通常の変数とスレッドID変数を分離
✅ **3次元ベクトル方式**: OpenCL/CUDAのget_global_id(dim)に近い自然な表現
✅ **シンプルな管理**: 9個の変数ではなく3個のベクトル変数で管理
✅ **柔軟なアクセス**: `get_thread_id_name()`でID変数名を取得
✅ **衝突検出が容易**: `has_name_conflict()`で名前衝突を簡単にチェック
✅ **拡張性**: 将来的にスレッドIDの属性を追加しやすい
✅ **デバッグしやすい**: どの変数がスレッドIDなのか一目瞭然

### コード例の比較

**旧方式 (次元ごとにスカラー変数):**
```rust
thread_ids: [
    ThreadIdDecl { name: "gid_x", id_type: GlobalId, dimension: 0 },
    ThreadIdDecl { name: "gid_y", id_type: GlobalId, dimension: 1 },
    ThreadIdDecl { name: "gid_z", id_type: GlobalId, dimension: 2 },
    // ... lid_x, lid_y, lid_z, group_x, group_y, group_z と続く（計9個）
]
// アクセス: Var("gid_x")
```

**新方式 (3次元ベクトル):**
```rust
thread_ids: [
    ThreadIdDecl { name: "global_id", id_type: GlobalId },  // Vec<Usize, 3>
    ThreadIdDecl { name: "local_id", id_type: LocalId },
    ThreadIdDecl { name: "group_id", id_type: GroupId },
]
// アクセス: Load { target: Var("global_id"), index: 0, vector_width: 1 }  // global_id[0]
```

---

## 8. まとめ

### v4の主要な設計（ベクトル方式）

1. ✅ **ThreadIdDecl構造体**: スレッドID変数を型安全に管理
2. ✅ **KernelScope構造体**: 通常の変数とスレッドID変数を分離
3. ✅ **3次元ベクトル型**: `Vec<Usize, 3>`として統一的に管理
4. ✅ **明示的な型情報**: GlobalId, LocalId, GroupIdを区別
5. ✅ **配列アクセス**: `global_id[0]`, `global_id[1]`, `global_id[2]`でx, y, zにアクセス
6. ✅ **柔軟なアクセスメソッド**: `get_thread_id_name()`等のユーティリティ

この設計で実装を進めましょう！

**実装状況**: コア型定義（ThreadIdDecl, KernelScope, Kernel/CallKernelノード）は実装済み。全223テスト通過。
