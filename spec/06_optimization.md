# Optimization モジュール仕様

## 概要

最適化モジュールは計算の効率化を担います。2つの主要な段階で最適化が実行されます：
1. **グラフレベル最適化** (`opt/graph`): テンソル演算の融合と再構成
2. **ASTレベル最適化** (`opt/ast`): ループと代数的簡約化

## グラフレベル最適化 (opt/graph)

### fusion.rs - 演算融合

複数の演算を1つのカーネルに統合することで、メモリアクセスとカーネル起動のオーバーヘッドを削減します。

#### 融合可能な演算パターン

**1. Elementwise融合**
```rust
// 融合前: 3つの独立したカーネル
let b = a + 1.0;
let c = b * 2.0;
let d = c.sqrt();

// 融合後: 1つのカーネル
// d = sqrt((a + 1.0) * 2.0)
```

**2. Elementwise-Reduce融合**
```rust
// 融合前: 要素演算 + 縮約
let b = a * a;
let c = b.sum(axis);

// 융合後: 1つのカーネル
// c = sum(a * a, axis)
```

**3. Elementwise-Cumulative融合**
```rust
// 융合前: 要素演算 + 累積演算
let b = a * 2.0;
let c = b.cumsum(axis);

// 융合後: 1つのカーネル
// c = cumsum(a * 2.0, axis)
```

**4. Multi-Reduce融合**
```rust
// 융合전: 複数軸の独立した縮約
let b = a.sum(0);
let c = b.sum(1);

// 융합후: 複数軸の同時縮約
// c = sum(a, axes=[0, 1])
```

#### 融合の制約

融合が適用される条件：
1. **データ依存関係**: 融合する演算間に直接的なデータフローがある
2. **分岐なし**: ノードの強参照カウントが1（他で使用されていない）
3. **メモリレイアウト**: 融合可能な演算（View操作は融合しない）

#### 融合アルゴリズム

```rust
pub fn fuse_graph(graph: &mut Graph) {
    loop {
        let mut changed = false;

        for output_node in &graph.outputs {
            if let Some(fused) = try_fuse_elementwise(output_node) {
                // 出力ノードを融合済みノードに置き換え
                changed = true;
            }
        }

        if !changed {
            break;  // 融合可能な演算がなくなるまで繰り返す
        }
    }
}
```

#### ビジュアライザー連携

環境変数`VIZ=1`を設定すると、各最適化ステップのスナップショットを記録：

```rust
if is_viz_enabled() {
    add_global_snapshot(graph.clone(), "Initial graph");
    // ... 最適化処理 ...
    add_global_snapshot(graph.clone(), "After fusion");
}
```

## ASTレベル最適化 (opt/ast)

### constant_folding.rs - 定数畳み込み

コンパイル時に計算可能な式を評価します。

**最適化例:**
```rust
// 最適化前
Add(Const(2), Const(3))

// 最適化後
Const(5)
```

**対応演算:**
- 算術演算: Add, Mul, Neg, Recip
- 数学関数: Sin, Sqrt, Log2, Exp2
- 比較演算: LessThan, Eq
- 条件選択: Select

### simplify.rs - 代数的簡約化

代数的な恒等式を利用して式を簡略化します。

**最適化ルール:**

1. **加算の恒等元**
   - `a + 0` → `a`
   - `0 + a` → `a`

2. **乗算の恒等元と零元**
   - `a * 1` → `a`
   - `1 * a` → `a`
   - `a * 0` → `0`
   - `0 * a` → `0`

3. **二重否定**
   - `Neg(Neg(a))` → `a`

4. **二重逆数**
   - `Recip(Recip(a))` → `a`

5. **最大値の簡約化**
   - `Max(a, a)` → `a`

6. **条件選択の簡約化**
   - `Select(true, a, b)` → `a`
   - `Select(false, a, b)` → `b`
   - `Select(cond, a, a)` → `a`

### heuristic/ - ヒューリスティック最適化

#### optimizer.rs - ビームサーチ最適化

複数の最適化パスを試行し、コストが最も低いものを選択します。

**アルゴリズム:**
```rust
pub fn optimize_with_beam_search(
    ast: AstNode,
    beam_width: usize,
    max_depth: usize,
) -> AstNode {
    let mut beam = vec![ast];

    for depth in 0..max_depth {
        let mut candidates = Vec::new();

        // 各候補に対して全ての変換を適用
        for state in &beam {
            for suggestion in suggest_transformations(state) {
                candidates.push(suggestion.apply(state));
            }
        }

        // コストでソートして上位beam_width個を残す
        candidates.sort_by_key(|ast| estimate_cost(ast));
        beam = candidates.into_iter().take(beam_width).collect();

        // 早期終了: 全候補のコストが変化しなければ終了
        if no_improvement(&beam) {
            break;
        }
    }

    beam.into_iter().min_by_key(|ast| estimate_cost(ast)).unwrap()
}
```

**ビームサーチパラメータ:**
- `beam_width`: 各世代で保持する候補数（デフォルト: 10）
- `max_depth`: 最大探索深度（デフォルト: 100）

#### cost_estimator.rs - コスト推定

ASTの実行コストを推定します。

**コスト計算:**
```rust
pub fn estimate_cost(ast: &AstNode) -> usize {
    match ast {
        AstNode::Add(l, r) | AstNode::Mul(l, r) => {
            1 + estimate_cost(l) + estimate_cost(r)
        }
        AstNode::Neg(n) => 1 + estimate_cost(n),
        AstNode::Recip(n) => 5 + estimate_cost(n),  // 除算は高コスト
        AstNode::Sin(n) => 20 + estimate_cost(n),   // 三角関数は高コスト
        AstNode::Sqrt(n) => 10 + estimate_cost(n),
        AstNode::Range { body, max, .. } => {
            // ループのコスト = 繰り返し回数 * ボディのコスト
            let iterations = estimate_iterations(max);
            iterations * estimate_cost(body)
        }
        AstNode::Const(_) | AstNode::Var(_) => 0,
        _ => ast.children().iter().map(|c| estimate_cost(c)).sum(),
    }
}
```

#### suggester/ - 変換提案

各サジェスターが特定の最適化パターンを提案します。

**algebraic.rs - 代数的変換**
- 分配法則: `a * (b + c)` → `a*b + a*c`
- 結合法則: `(a + b) + c` → `a + (b + c)`

**commutative.rs - 可換則**
- 交換法則: `a + b` → `b + a`
- 定数を右側に移動: `2 + a` → `a + 2`

**factorization.rs - 因数分解**
- `a*b + a*c` → `a * (b + c)`

**inverse.rs - 逆数最適化**
- `a / b / c` → `a / (b * c)`
- `1 / (1 / a)` → `a`

**reciprocal.rs - 逆数の最適化**
- 複数の除算を1回の逆数計算にまとめる

**log_exp_law.rs - 対数・指数法則**
- `log2(a * b)` → `log2(a) + log2(b)`
- `exp2(a + b)` → `exp2(a) * exp2(b)`

**sqrt_law.rs - 平方根法則**
- `sqrt(a * b)` → `sqrt(a) * sqrt(b)`
- `sqrt(a) * sqrt(a)` → `a`

**max_law.rs - 最大値法則**
- `max(a, max(b, c))` → `max(max(a, b), c)`

**bitwise.rs - ビット演算最適化**
- `a & 0` → `0`
- `a | 0` → `a`
- `a ^ a` → `0`

**loop_transform.rs - ループ変換**
- ループの融合
- ループの分割

**loop_interchange.rs - ループ交換**
- ループの順序を変更してキャッシュ効率を向上

**loop_tiling.rs - ループタイリング**
- ループをタイル単位に分割

**loop_unroll.rs - ループアンロール**
- 小さなループを展開

**unroll_hint.rs - アンロールヒント**
- `#pragma unroll`の挿入を提案

**loop_extraction.rs - ループ抽出**
- ループを別の関数に抽出
- GPU並列化の前処理として使用

**kernelize.rs - カーネル化**
- 関数をGPUカーネルに変換
- スレッドID変数を含むKernelScopeを生成

**parallelize.rs - GPU並列化**
- ループをGPUスレッドで並列実行
- 境界チェック（`if (i < n)`）の自動挿入

**rule_based.rs - ルールベース最適化**
- 定義済みのルールセットを適用

### GPU並列化パイプライン

GPU並列化は3段階のパイプラインで実行されます:

**1. ループ抽出 (Loop Extraction)**
```rust
// 変換前
fn main() {
    for i in 0..n {
        output[i] = input[i] * 2.0;
    }
}

// 変換後
fn extracted_loop(output, input, n) {
    for i in 0..n {
        output[i] = input[i] * 2.0;
    }
}

fn main() {
    extracted_loop(output, input, n);
}
```

**2. カーネル化 (Kernelization)**
```rust
// 変換前: Function
fn extracted_loop(output, input, n) { ... }

// 変換後: Kernel
kernel extracted_loop_kernel(output, input, n) { ... }
```

**3. 並列化 (Parallelization)**
```rust
// 変換前
kernel my_kernel(output, input, n) {
    for i in 0..n {
        output[i] = input[i] * 2.0;
    }
}

// 変換後
kernel my_kernel(output, input, n) {
    size_t global_id[3] = get_global_id();
    size_t i = global_id[0];
    if (i < n) {
        output[i] = input[i] * 2.0;
    }
}
```

### analysis/ - 並列化可能性分析

GPU並列化の安全性を保証するための分析ツール。

**index_analysis.rs - インデックスパターン解析**
```rust
pub enum IndexPattern {
    Identity,              // a[i]
    Offset(isize),        // a[i + 5]
    Scaled(isize),        // a[i * 2]
    ScaledOffset(isize, isize), // a[i * 2 + 3]
    Constant(isize),      // a[42]
    Complex,              // その他
}
```

インデックスパターンの分離性判定:
- `Identity` vs `Identity`: 競合あり（同じインデックス）
- `Identity` vs `Offset(k)`: 分離（k ≠ 0）
- `Scaled(2)` vs `Scaled(2) + Offset(1)`: 分離（奇数と偶数）

**memory_access.rs - メモリアクセス分析**

ループ内のメモリアクセスパターンを収集:
```rust
pub struct MemoryAccess {
    pub variable: String,
    pub index: Box<AstNode>,
    pub is_write: bool,
}
```

**parallelizable.rs - 並列化可能性判定**
```rust
pub enum ParallelizabilityResult {
    Safe,                    // 並列化安全
    UnsafeReadWrite(String), // Read-Write競合
    UnsafeWriteWrite(String),// Write-Write競合
}

pub fn is_loop_parallelizable(
    body: &AstNode,
    counter_name: &str,
) -> ParallelizabilityResult
```

**並列化安全条件:**
1. 異なるイテレーション間でRead-Write競合がない
2. 異なるイテレーション間でWrite-Write競合がない
3. インデックスパターンが分離可能

**variable_usage.rs - 変数使用分析**

ASTサブツリー内で使用される変数を収集:
```rust
pub fn collect_used_variables(node: &AstNode) -> HashSet<String>
```

ループ抽出時に必要な引数を決定するために使用。

## 最適化の適用順序

1. **グラフ最適化**
   ```rust
   let mut graph = build_graph();
   fuse_graph(&mut graph);
   ```

2. **Lowering**
   ```rust
   let mut lowerer = Lowerer::new();
   let program = lowerer.lower(&graph);
   ```

3. **AST最適化**
   ```rust
   let optimized = optimize_with_beam_search(
       program,
       beam_width: 10,
       max_depth: 100,
   );
   ```

4. **定数畳み込み**
   ```rust
   let folded = constant_folding(optimized);
   ```

5. **簡約化**
   ```rust
   let simplified = simplify(folded);
   ```

## 最適化の効果測定

```rust
let original_cost = estimate_cost(&original_ast);
let optimized_cost = estimate_cost(&optimized_ast);
let improvement = (original_cost - optimized_cost) as f64 / original_cost as f64;
println!("Cost reduction: {:.1}%", improvement * 100.0);
```
