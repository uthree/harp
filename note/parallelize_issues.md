# 並列化処理プランの問題点と改善提案

このドキュメントは `draft_parallelize.md` の分析結果をまとめたものです。

## 1. ASTノード設計の不整合

### 問題点
- `Load`と`Store`ノードを追加するとあるが、現状で既に`Store`ノードは存在し、`Deref`（ポインタからの読み込み）がある
- "配列単位での読み書き"という表現が曖昧で、既存の`Store`/`Deref`との違いが不明確

### 提案
- 既存の`Store`/`Deref`をそのまま使うか、あるいは新しい`Load`ノードの役割を明確化する
- GPUメモリ空間（グローバル、ローカル、共有メモリなど）を区別したい場合は、メモリ空間属性を追加すべき
  ```rust
  enum MemorySpace {
      Global,
      Local,
      Shared,
      Private,
  }

  // Storeノードにメモリ空間属性を追加
  Store {
      target: Box<Self>,
      index: Box<Self>,
      value: Box<Self>,
      memory_space: Option<MemorySpace>,
  }
  ```

---

## 2. Kernel/CallKernelノードの設計が不十分

### 問題点
- `lid`（ローカルID）と`gid`（グローバルID/グループID）の定義が混乱している
  - ドキュメントでは「gid(ループ番号)」とあるが、通常GPUではgidはグローバルスレッドIDまたはグループIDを意味する
- スレッド数とグループ数の関係が不明確
- 3次元ベクトルを簡略化して1次元だけ使うとしているが、将来の拡張性に問題がある

### 提案: GPU並列化の標準的な用語を使用

**変数の定義:**
- `global_id`: グローバルスレッドID（全スレッド中での一意なID）
- `local_id`: ワークグループ内でのローカルスレッドID
- `group_id`: ワークグループID
- `global_size`: 総スレッド数
- `local_size`: ワークグループあたりのスレッド数
- `num_groups`: ワークグループ数

**関係式:**
```
global_id = group_id * local_size + local_id
global_size = num_groups * local_size
```

**Kernelノードの設計案:**
```rust
Kernel {
    name: String,
    scope: Scope,
    statements: Vec<AstNode>,
    arguments: Vec<(String, DType)>,
    return_type: DType,

    // 並列化情報
    global_size: Box<AstNode>,  // 総スレッド数（実行時に計算可能）
    local_size: Box<AstNode>,   // ワークグループサイズ
    // 将来的に3次元対応: global_size: [Box<AstNode>; 3]
}

CallKernel {
    name: String,
    args: Vec<AstNode>,
    global_size: Box<AstNode>,
    local_size: Box<AstNode>,
}
```

**ビルトイン変数へのアクセス:**
組み込み関数として表現:
```rust
// 例: CallFunction { name: "get_global_id", args: vec![Const(0)] }
get_global_id(0)  // x次元のglobal_id
get_local_id(0)   // x次元のlocal_id
get_group_id(0)   // x次元のgroup_id
get_global_size(0)
get_local_size(0)
get_num_groups(0)
```

---

## 3. メモリアクセス競合の解析方法

### 問題点
- 2つの方法（ラベル付与 vs. ポインタ解析）を提案しているが、どちらも課題がある
- **方法1（ラベル付与）**:
  - ループ変換（interchange, tiling, unroll）時にラベルの維持が複雑
  - Reduceループ以外にも競合が起こりうる（例：累積代入 `a[i] += b[i+1]`）
- **方法2（ポインタ解析）**:
  - 一般的なポインタ解析は決定不能問題に近い
  - 完全な解析は実装コストが高い

### 提案: ハイブリッドアプローチ

#### Phase 1: Lower時の保守的な判定
Lower時にRangeノードに以下の情報を付与:
```rust
Range {
    // ... 既存フィールド
    parallel_hint: ParallelHint,
}

enum ParallelHint {
    Safe,           // 並列化安全と判明
    Unsafe,         // 並列化不可能（Reduceなど）
    Unknown,        // 不明（要解析）
}
```

**判定ルール:**
- Elementwise演算のループ → `Safe`
- Reduce演算のループ → `Unsafe`
- Cumulative演算のループ → `Unsafe`
- その他 → `Unknown`

#### Phase 2: Suggester内での簡易メモリアクセス解析

並列化可能な条件:
1. ループ変数がStore命令のインデックスに直接使われている
2. 同じ反復内でのread-after-write依存がない
3. 異なる反復間でのwrite-after-write, write-after-read依存がない

**簡易解析の実装方針:**
```rust
fn is_parallelizable(loop_var: &str, body: &AstNode) -> bool {
    // Storeノードを収集
    let stores = collect_stores(body);

    for store in stores {
        // インデックス式を解析
        let index_vars = extract_variables(&store.index);

        // ループ変数が含まれているか確認
        if !index_vars.contains(&loop_var.to_string()) {
            // ループ変数がインデックスに使われていない = 全反復で同じ位置に書き込む
            return false;  // 競合あり
        }

        // アフィン変換のみ許可（i, i+1, 2*i など）
        if !is_affine_expression(&store.index, loop_var) {
            return false;  // 複雑な依存関係
        }
    }

    // 全てのチェックを通過
    true
}
```

#### 安全側に倒す原則
不明な場合は並列化しない。パフォーマンスより正確性を優先。

---

## 4. Barrierの扱いが不明確

### 問題点
- 現状のASTに`Barrier`ノードは存在するが、Kernel内での同期の意味論が定義されていない
- GPUでは共有メモリへのアクセスにはバリア同期が必須だが、その挿入タイミングが不明

### 提案: Barrier挿入戦略の明確化

#### バリアの種類
```rust
enum BarrierScope {
    Local,   // ワークグループ内同期
    Global,  // 全スレッド同期（重い、避けるべき）
}

Barrier {
    scope: BarrierScope,
    memory_fence: MemoryFenceFlags,
}

struct MemoryFenceFlags {
    local_memory: bool,
    global_memory: bool,
}
```

#### 挿入戦略
1. **手動挿入**: ユーザーまたはLowererが明示的に挿入
   - 高レベルオペレーション（Reduce, Scan）のLower時に適切な位置にBarrierを挿入
2. **自動挿入**: 最適化パス内で共有メモリアクセスを検出して挿入
   - 共有メモリへのwrite後、他スレッドがreadする前
   - ループタイリングで共有メモリを使う場合

#### 必要なバリアのパターン
```
// パターン1: タイル単位でのデータ共有
for group_id in 0..num_groups {
    shared_mem[local_id] = global_mem[group_id * local_size + local_id];
    barrier(Local);  // 全スレッドがデータをロードするまで待機

    // 共有メモリを使った計算
    result = compute(shared_mem);
    barrier(Local);  // 次の反復前に同期
}
```

---

## 5. RangeExtractの適用条件が曖昧

### 問題点
- "トップレベルのRangeノード"の定義が不明確
- 既にKernelizeされた関数には適用しないとあるが、その判定方法が不明

### 提案: 明確な定義と判定方法

#### トップレベルの定義
Function直下のBlockの中で、最初に現れるRangeノード:
```rust
Function {
    statements: [
        Assign(...),      // これはトップレベルではない
        Range { ... },    // これが"トップレベルのRange"
        Assign(...),
    ]
}
```

#### 複数のRangeノードがある場合
最初のRangeノードのみを抽出対象とする。または:
- 全てのトップレベルRangeノードを個別に抽出
- ユーザーが指定したRangeノードのみ抽出

#### Kernel判定
```rust
fn is_kernel(node: &AstNode) -> bool {
    matches!(node, AstNode::Kernel { .. })
}

// RangeExtractの適用条件
fn can_extract_range(func: &AstNode) -> bool {
    if is_kernel(func) {
        return false;  // Kernelには適用しない
    }

    // Function内にトップレベルのRangeノードが存在するか
    has_top_level_range(func)
}
```

---

## 6. 並列化の段階的戦略

### 問題点
- gid → lid の順で並列化するとあるが、この戦略が常に最適とは限らない
- ワークグループサイズの決定方法が言及されていない

### 提案: 複数の並列化戦略をSuggesterとして提供

#### 戦略1: 完全なgid並列化
```rust
// 元のループ
for i in 0..N {
    a[i] = b[i] + c[i];
}

// 変換後
Kernel {
    global_size: N,
    local_size: 1,  // ワークグループサイズ1 = グループ数 = N
    body: {
        let i = get_global_id(0);
        if i < N {
            a[i] = b[i] + c[i];
        }
    }
}
```

#### 戦略2: gid + lid の2段階並列化
```rust
// 元のループ
for i in 0..N {
    sum += a[i];  // Reduce演算
}

// 変換後（タイリング + 部分Reduce + 最終Reduce）
Kernel {
    global_size: num_groups * local_size,
    local_size: 256,
    body: {
        let gid = get_group_id(0);
        let lid = get_local_id(0);
        let lsize = get_local_size(0);

        // Phase 1: グループごとの部分Reduce
        shared_mem[lid] = 0;
        for i in (gid * lsize + lid)..(gid + 1) * lsize step lsize {
            if i < N {
                shared_mem[lid] += a[i];
            }
        }
        barrier(Local);

        // Phase 2: ワークグループ内でのReduce（ツリーリダクション）
        for stride in [128, 64, 32, 16, 8, 4, 2, 1] {
            if lid < stride {
                shared_mem[lid] += shared_mem[lid + stride];
            }
            barrier(Local);
        }

        // Phase 3: 結果の書き出し
        if lid == 0 {
            partial_sums[gid] = shared_mem[0];
        }
    }
}
// + 最終的なpartial_sumsのReduce（CPU側またはさらにKernel呼び出し）
```

#### ワークグループサイズの最適化
ハードウェア特性を考慮:
- NVIDIA GPU: warp size = 32 → local_size は32の倍数が効率的
- AMD GPU: wavefront size = 64
- 共有メモリサイズの制約
- レジスタ使用量

**Suggesterでの複数候補提案:**
```rust
impl ParallelizeSuggester {
    fn suggest_local_sizes(&self) -> Vec<usize> {
        vec![32, 64, 128, 256, 512, 1024]
            .into_iter()
            .filter(|&size| size <= self.max_local_size)
            .collect()
    }
}
```

---

## 7. OpenMPバックエンドとの整合性

### 問題点
- CPUでのOpenMP対応はRenderer側で実装するとあるが、Kernel/CallKernelノードとの対応が不明確
- Kernelノードのスレッド数/グループ数をどうOpenMPディレクティブに変換するか言及なし

### 提案: Renderer側での変換ルール

#### CRenderer拡張（OpenMP対応）
```rust
impl CRenderer {
    fn render_kernel(&mut self, node: &AstNode) -> String {
        if let AstNode::Kernel {
            name,
            scope,
            statements,
            arguments,
            return_type,
            global_size,
            local_size,
        } = node {
            let mut buffer = String::new();

            // 関数シグネチャ
            writeln!(buffer, "{} {}(...) {{",
                     self.render_dtype(return_type), name);

            // OpenMPディレクティブ
            writeln!(buffer, "#ifdef _OPENMP");
            writeln!(buffer, "#pragma omp parallel for");
            writeln!(buffer, "#endif");

            // ループ本体
            writeln!(buffer, "for (size_t __gid = 0; __gid < {}; __gid++) {{",
                     self.render_node(global_size));

            // gid, lidの定義（OpenMPではlidは通常無視）
            writeln!(buffer, "  size_t gid = __gid;");
            writeln!(buffer, "  size_t lid = 0;  // OpenMP: not used");

            // 本体をレンダリング
            // ... statements

            writeln!(buffer, "}}");
            writeln!(buffer, "}}");

            buffer
        }
    }
}
```

#### CallKernelのレンダリング
```c
// Kernel呼び出し
kernel_func(args...);  // OpenMPディレクティブは関数定義内にあるので、普通に呼び出すだけ
```

#### デバイス依存の最適化
- GPU用: `local_size`を活用して共有メモリ最適化
- CPU用: `local_size`を無視してシンプルな`parallel for`

---

## 8. 型システムとの整合性

### 問題点
- `lid`, `gid`の型が明示されていない（Usize? 3要素のベクトル?）
- 関数引数としてどう扱うかが不明

### 提案: ビルトイン関数として扱う

#### 組み込み関数の定義
```rust
// ASTノードとして表現
CallFunction {
    name: "get_global_id".to_string(),
    args: vec![Const(ConstLiteral::Usize(0))]  // 次元指定
}

// 型: fn get_global_id(dim: Usize) -> Usize
// 型: fn get_local_id(dim: Usize) -> Usize
// 型: fn get_group_id(dim: Usize) -> Usize
// etc.
```

#### Rendererでの変換
```rust
// CRenderer (OpenCL風)
"get_global_id" => "get_global_id"
"get_local_id" => "get_local_id"

// CRenderer (OpenMP)
"get_global_id" => "__gid"  // ループ変数に置き換え
"get_local_id" => "0"       // 常に0
```

#### 将来の3次元対応
```rust
// 現在: get_global_id(0) のみサポート
// 将来: get_global_id(0), get_global_id(1), get_global_id(2)
```

---

## 9. 最適化パイプラインの位置づけ

### 問題点
- Kernelize → Parallelize の順序は決まっているが、他の最適化（tiling, unroll）との関係が不明確

### 提案: 最適化パイプラインの明確化

#### 推奨される最適化順序
```
入力AST (Function)
    ↓
(1) ループ変換 (LoopInterchange, LoopTiling)
    ↓
(2) RangeExtract（必要に応じて）
    ↓
(3) Kernelize (Function → Kernel)
    ↓
(4) Parallelize - Stage 1: gid並列化
    ↓
(5) Parallelize - Stage 2: lid並列化（必要に応じて）
    ↓
(6) LoopUnroll（残ったループに対して）
    ↓
(7) AlgebraicSimplification, ConstantFolding
    ↓
出力AST (Kernel)
```

#### 各ステージの役割

**Stage 1-2: 前処理**
- メモリアクセスパターンの最適化
- 並列化しやすい形に変換

**Stage 3: Kernelize**
- Function → Kernel への変換
- 初期状態: global_size=1, local_size=1（逐次実行と同等）

**Stage 4-5: 並列化**
- 最外ループをgidで並列化
- さらに内側のループをlidで並列化（オプション）

**Stage 6-7: 後処理**
- 残ったシーケンシャルループの最適化
- 式の簡約化

#### Suggesterの優先順位
```rust
pub fn kernel_optimization_suggesters() -> Vec<Box<dyn RewriteSuggester>> {
    vec![
        // Phase 1: ループ構造の最適化
        Box::new(LoopInterchangeSuggester),
        Box::new(LoopTilingSuggester::new(64)),

        // Phase 2: カーネル化
        Box::new(RangeExtractSuggester),
        Box::new(KernelizeSuggester),

        // Phase 3: 並列化
        Box::new(ParallelizeSuggester::new(ParallelizeMode::GlobalId)),
        Box::new(ParallelizeSuggester::new(ParallelizeMode::LocalId)),

        // Phase 4: 細部最適化
        Box::new(LoopUnrollSuggester::new()),
        Box::new(AlgebraicLawSuggester),
    ]
}
```

---

## 10. エラーハンドリング

### 問題点
- 並列化不可能な場合の処理が言及されていない
- 不正なKernel定義の検証方法が不明

### 提案: バリデーション層の追加

#### Kernelバリデータ
```rust
pub struct KernelValidator;

impl KernelValidator {
    pub fn validate(kernel: &AstNode) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check 1: Kernel内でさらにKernelを呼び出していないか
        if Self::contains_kernel_call(kernel) {
            errors.push(ValidationError::NestedKernelCall);
        }

        // Check 2: 並列化可能性の検証
        if !Self::is_parallelizable(kernel) {
            errors.push(ValidationError::NotParallelizable(
                "Memory dependency detected".to_string()
            ));
        }

        // Check 3: スレッド数の妥当性
        if let Some(size) = Self::extract_global_size(kernel) {
            if size > Self::max_threads() {
                errors.push(ValidationError::TooManyThreads {
                    requested: size,
                    max: Self::max_threads(),
                });
            }
        }

        // Check 4: 共有メモリ使用量
        let shared_mem_usage = Self::calculate_shared_memory(kernel);
        if shared_mem_usage > Self::max_shared_memory() {
            errors.push(ValidationError::SharedMemoryExceeded {
                requested: shared_mem_usage,
                max: Self::max_shared_memory(),
            });
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

pub enum ValidationError {
    NestedKernelCall,
    NotParallelizable(String),
    TooManyThreads { requested: usize, max: usize },
    SharedMemoryExceeded { requested: usize, max: usize },
}
```

#### 並列化不可能な場合の処理
```rust
impl ParallelizeSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        // メモリ依存性チェック
        if !self.is_safe_to_parallelize(node) {
            // 並列化しない（空のsuggestionを返す）
            return vec![];
        }

        // 並列化を提案
        vec![self.parallelize(node)]
    }
}
```

---

## まとめ: 実装前に解決すべき重要課題

### 優先度: 高
1. **Kernel/CallKernelノードの詳細設計**
   - lid/gidの明確な定義と標準用語への統一
   - ビルトイン関数としての実装方針

2. **メモリアクセス解析の実装方針**
   - ハイブリッドアプローチ（ラベル + 簡易解析）
   - 並列化の安全性を保証する方法

3. **最適化パイプライン全体の設計**
   - 各最適化の適用順序と依存関係
   - Suggesterの優先順位

### 優先度: 中
4. **Barrierの挿入戦略**
   - 共有メモリ使用時の同期
   - 自動挿入 vs. 手動挿入

5. **デバイス固有の調整**
   - GPU vs. CPU での挙動の違い
   - OpenMPとの対応

### 優先度: 低（将来の拡張）
6. **3次元並列化のサポート**
7. **動的なワークグループサイズ最適化**
8. **高度なメモリアクセスパターン解析**

これらを事前に明確化することで、より堅牢で保守性の高い並列化機能を実装できます。
