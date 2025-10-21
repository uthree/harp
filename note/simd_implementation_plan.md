# SIMD演算の実装計画

## 現状の問題点

### 実装されている機能

現在の`src/backend/c_like.rs`では、`vector_width`に基づいてベクトル型キャストを生成している：

```rust
// Load の場合
match vector_width {
    1 => write!(buffer, "*({} + {})", target_str, index_str).unwrap(),
    2 | 4 | 8 | 16 => {
        write!(
            buffer,
            "*((float{}*)({} + {}))",
            vector_width, target_str, index_str
        ).unwrap();
    }
    _ => panic!("Unsupported vector width: {}", vector_width),
}
```

**生成されるコード例:**
```c
*((float8*)(input_0 + ridx0)) = *((float8*)(input_1 + ridx0)) + *((float8*)(input_2 + ridx0));
```

### 問題

1. **型定義が生成されない**: `float2`, `float4`, `float8`, `float16`などの型が未定義のため、標準C環境ではコンパイルエラー
2. **SIMD拡張の選択ができない**: SSE/AVX/AVX-512/NEONなど、ターゲット環境に応じた選択ができない
3. **データ型がfloatに固定**: int, doubleなどの他の型に対応していない
4. **演算の対応が不明確**: 加算・乗算以外のSIMD演算（比較、選択など）の扱いが未定義

## 目標

1. **移植性**: 標準C/C++環境でコンパイル可能なコードを生成
2. **柔軟性**: ターゲットアーキテクチャに応じてSIMD拡張を選択可能
3. **制御性**: Renderer Optionで利用可能なSIMD機能を細かく制御
4. **拡張性**: 新しいSIMD命令セットや演算に容易に対応

## 設計案

### 1. SIMD拡張の抽象化

#### SIMD拡張の種類

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdExtension {
    /// コンパイラ自動ベクトル化に任せる（型定義なし）
    AutoVectorize,

    /// GCC/Clang ベクトル拡張 (__attribute__((vector_size(N))))
    GccVectorExtension,

    /// Intel Intrinsics (SSE/AVX/AVX-512)
    IntelIntrinsics(IntelSimdLevel),

    /// ARM NEON
    ArmNeon,

    /// OpenCL ベクトル型
    OpenCL,

    /// CUDA ベクトル型
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntelSimdLevel {
    SSE,     // 128bit: float4
    AVX,     // 256bit: float8
    AVX512,  // 512bit: float16
}
```

### 2. データ型とベクトル幅の対応

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorTypeInfo {
    pub base_type: DType,
    pub vector_width: usize,
}

impl VectorTypeInfo {
    /// ベクトル型名を取得
    pub fn type_name(&self, extension: SimdExtension) -> String {
        match extension {
            SimdExtension::GccVectorExtension => {
                // float4, double2, int8 など
                format!("{}{}", self.base_type_name(), self.vector_width)
            }
            SimdExtension::IntelIntrinsics(level) => {
                match (self.base_type, self.vector_width, level) {
                    (DType::F32, 4, IntelSimdLevel::SSE) => "__m128".to_string(),
                    (DType::F32, 8, IntelSimdLevel::AVX) => "__m256".to_string(),
                    (DType::F32, 16, IntelSimdLevel::AVX512) => "__m512".to_string(),
                    (DType::F64, 2, IntelSimdLevel::SSE) => "__m128d".to_string(),
                    (DType::F64, 4, IntelSimdLevel::AVX) => "__m256d".to_string(),
                    (DType::I32, 4, IntelSimdLevel::SSE) => "__m128i".to_string(),
                    (DType::I32, 8, IntelSimdLevel::AVX) => "__m256i".to_string(),
                    _ => panic!("Unsupported vector type configuration"),
                }
            }
            SimdExtension::OpenCL => {
                // float4, double2, int8 など（OpenCL組み込み型）
                format!("{}{}", self.base_type_name(), self.vector_width)
            }
            _ => unimplemented!(),
        }
    }

    fn base_type_name(&self) -> &str {
        match self.base_type {
            DType::F32 => "float",
            DType::F64 => "double",
            DType::I32 => "int",
            DType::Isize => "ssize_t",
            DType::Usize => "size_t",
            _ => panic!("Unsupported base type for SIMD"),
        }
    }
}
```

### 3. 型定義生成

#### GCC/Clang ベクトル拡張

```rust
fn generate_vector_type_definitions(
    used_types: &HashSet<VectorTypeInfo>,
    extension: SimdExtension,
) -> String {
    match extension {
        SimdExtension::GccVectorExtension => {
            let mut defs = String::new();
            defs.push_str("/* SIMD vector type definitions */\n");

            for type_info in used_types {
                let base_name = type_info.base_type_name();
                let type_name = type_info.type_name(extension);
                let byte_size = type_info.vector_width * type_info.base_type.size_bytes();

                defs.push_str(&format!(
                    "typedef {} {} __attribute__((vector_size({})));\n",
                    base_name, type_name, byte_size
                ));
            }
            defs.push_str("\n");
            defs
        }
        SimdExtension::IntelIntrinsics(_) => {
            // Intrinsicsの場合はヘッダーインクルードのみ
            "#include <immintrin.h>\n\n".to_string()
        }
        SimdExtension::OpenCL | SimdExtension::Cuda => {
            // 組み込み型なので定義不要
            String::new()
        }
        SimdExtension::AutoVectorize => {
            // 自動ベクトル化の場合は型定義不要
            String::new()
        }
        _ => unimplemented!(),
    }
}
```

**生成例 (GCC Vector Extension):**
```c
/* SIMD vector type definitions */
typedef float float4 __attribute__((vector_size(16)));
typedef float float8 __attribute__((vector_size(32)));
typedef double double2 __attribute__((vector_size(16)));
typedef int int4 __attribute__((vector_size(16)));
```

### 4. Renderer Option による制御

```rust
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// 使用するSIMD拡張
    pub extension: SimdExtension,

    /// サポートする最大ベクトル幅
    pub max_vector_width: usize,

    /// データ型ごとのベクトル化設定
    pub type_configs: HashMap<DType, TypeSimdConfig>,

    /// 無効化する演算
    pub disabled_operations: HashSet<SimdOperation>,
}

#[derive(Debug, Clone)]
pub struct TypeSimdConfig {
    /// この型で許可するベクトル幅
    pub allowed_widths: Vec<usize>,

    /// デフォルトで使用するベクトル幅
    pub default_width: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimdOperation {
    Load,
    Store,
    Add,
    Mul,
    Sub,
    Div,
    Sqrt,
    Sin,
    Exp,
    Max,
    Min,
    Compare,
    Select,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            extension: SimdExtension::GccVectorExtension,
            max_vector_width: 8,  // AVX相当
            type_configs: Self::default_type_configs(),
            disabled_operations: HashSet::new(),
        }
    }
}

impl SimdConfig {
    fn default_type_configs() -> HashMap<DType, TypeSimdConfig> {
        let mut configs = HashMap::new();

        // F32: 2, 4, 8, 16 をサポート
        configs.insert(DType::F32, TypeSimdConfig {
            allowed_widths: vec![2, 4, 8, 16],
            default_width: Some(8),
        });

        // F64: 2, 4 のみサポート（レジスタサイズの制約）
        configs.insert(DType::F64, TypeSimdConfig {
            allowed_widths: vec![2, 4],
            default_width: Some(2),
        });

        configs
    }

    /// ベクトル幅が許可されているか確認
    pub fn is_width_allowed(&self, dtype: DType, width: usize) -> bool {
        if width > self.max_vector_width {
            return false;
        }

        self.type_configs
            .get(&dtype)
            .map(|config| config.allowed_widths.contains(&width))
            .unwrap_or(false)
    }

    /// 演算がSIMD化可能か確認
    pub fn is_operation_enabled(&self, op: SimdOperation) -> bool {
        !self.disabled_operations.contains(&op)
    }
}
```

### 5. CRendererOption への統合

```rust
#[derive(Debug, Clone)]
pub struct CRendererOption {
    /// OpenMP並列化を使用するか
    pub use_openmp: bool,

    /// SIMD設定
    pub simd: SimdConfig,
}

impl Default for CRendererOption {
    fn default() -> Self {
        Self {
            use_openmp: true,
            simd: SimdConfig::default(),
        }
    }
}

impl CRendererOption {
    /// SSE相当の設定
    pub fn with_sse() -> Self {
        Self {
            use_openmp: true,
            simd: SimdConfig {
                extension: SimdExtension::IntelIntrinsics(IntelSimdLevel::SSE),
                max_vector_width: 4,
                ..Default::default()
            },
        }
    }

    /// AVX相当の設定
    pub fn with_avx() -> Self {
        Self {
            use_openmp: true,
            simd: SimdConfig {
                extension: SimdExtension::IntelIntrinsics(IntelSimdLevel::AVX),
                max_vector_width: 8,
                ..Default::default()
            },
        }
    }

    /// AVX-512相当の設定
    pub fn with_avx512() -> Self {
        Self {
            use_openmp: true,
            simd: SimdConfig {
                extension: SimdExtension::IntelIntrinsics(IntelSimdLevel::AVX512),
                max_vector_width: 16,
                ..Default::default()
            },
        }
    }

    /// SIMD無効化
    pub fn without_simd() -> Self {
        Self {
            use_openmp: true,
            simd: SimdConfig {
                extension: SimdExtension::AutoVectorize,
                max_vector_width: 1,
                ..Default::default()
            },
        }
    }
}
```

### 6. レンダリング時の型定義収集

```rust
impl CRenderer {
    /// プログラム全体をレンダリング
    pub fn render(&mut self, program: AstNode) -> String {
        // 使用されているベクトル型を収集
        let used_vector_types = self.collect_vector_types(&program);

        let mut code = String::new();

        // 標準ヘッダー
        code.push_str(&self.render_standard_headers());

        // SIMD型定義
        if !used_vector_types.is_empty() {
            code.push_str(&generate_vector_type_definitions(
                &used_vector_types,
                self.simd_config.extension,
            ));
        }

        // プログラム本体
        code.push_str(&self.render_program_body(program));

        code
    }

    /// AST内で使用されているベクトル型を収集
    fn collect_vector_types(&self, node: &AstNode) -> HashSet<VectorTypeInfo> {
        let mut types = HashSet::new();
        self.collect_vector_types_recursive(node, &mut types);
        types
    }

    fn collect_vector_types_recursive(
        &self,
        node: &AstNode,
        types: &mut HashSet<VectorTypeInfo>,
    ) {
        match node {
            AstNode::Load { vector_width, .. } | AstNode::Store { vector_width, .. } => {
                if *vector_width > 1 {
                    // TODO: 実際のデータ型を追跡する必要がある
                    types.insert(VectorTypeInfo {
                        base_type: DType::F32,  // 暫定
                        vector_width: *vector_width,
                    });
                }
            }
            _ => {}
        }

        // 再帰的に子ノードを探索
        for child in node.children() {
            self.collect_vector_types_recursive(child, types);
        }
    }
}
```

### 7. Load/Store のレンダリング更新

```rust
impl CLikeRenderer for CRenderer {
    fn render_node(&mut self, node: &AstNode) -> String {
        match node {
            AstNode::Load { target, index, vector_width } => {
                let target_str = self.render_node(target);
                let index_str = self.render_node(index);

                if *vector_width == 1 {
                    // スカラー読み込み
                    format!("*({} + {})", target_str, index_str)
                } else {
                    // ベクトル読み込み
                    if !self.simd_config.is_operation_enabled(SimdOperation::Load) {
                        panic!("SIMD load is disabled");
                    }

                    let type_info = VectorTypeInfo {
                        base_type: DType::F32,  // TODO: 実際の型を取得
                        vector_width: *vector_width,
                    };

                    if !self.simd_config.is_width_allowed(type_info.base_type, *vector_width) {
                        panic!("Vector width {} not allowed for {:?}",
                               vector_width, type_info.base_type);
                    }

                    let type_name = type_info.type_name(self.simd_config.extension);

                    match self.simd_config.extension {
                        SimdExtension::GccVectorExtension | SimdExtension::OpenCL => {
                            format!("*(({type_name}*)({target_str} + {index_str}))")
                        }
                        SimdExtension::IntelIntrinsics(level) => {
                            // Intrinsic関数を使用
                            self.render_intel_load(type_info, &target_str, &index_str, level)
                        }
                        _ => unimplemented!(),
                    }
                }
            }

            AstNode::Store { target, index, value, vector_width } => {
                let target_str = self.render_node(target);
                let index_str = self.render_node(index);
                let value_str = self.render_node(value);

                if *vector_width == 1 {
                    // スカラー書き込み
                    format!("*({} + {}) = {}", target_str, index_str, value_str)
                } else {
                    // ベクトル書き込み
                    if !self.simd_config.is_operation_enabled(SimdOperation::Store) {
                        panic!("SIMD store is disabled");
                    }

                    let type_info = VectorTypeInfo {
                        base_type: DType::F32,  // TODO: 実際の型を取得
                        vector_width: *vector_width,
                    };

                    let type_name = type_info.type_name(self.simd_config.extension);

                    match self.simd_config.extension {
                        SimdExtension::GccVectorExtension | SimdExtension::OpenCL => {
                            format!("*(({type_name}*)({target_str} + {index_str})) = {value_str}")
                        }
                        SimdExtension::IntelIntrinsics(level) => {
                            // Intrinsic関数を使用
                            self.render_intel_store(type_info, &target_str, &index_str, &value_str, level)
                        }
                        _ => unimplemented!(),
                    }
                }
            }

            _ => {
                // 既存の実装
                // ...
            }
        }
    }
}
```

### 8. Intel Intrinsics のレンダリング

```rust
impl CRenderer {
    fn render_intel_load(
        &self,
        type_info: VectorTypeInfo,
        target: &str,
        index: &str,
        level: IntelSimdLevel,
    ) -> String {
        match (type_info.base_type, type_info.vector_width, level) {
            (DType::F32, 4, IntelSimdLevel::SSE) => {
                format!("_mm_loadu_ps({} + {})", target, index)
            }
            (DType::F32, 8, IntelSimdLevel::AVX) => {
                format!("_mm256_loadu_ps({} + {})", target, index)
            }
            (DType::F32, 16, IntelSimdLevel::AVX512) => {
                format!("_mm512_loadu_ps({} + {})", target, index)
            }
            (DType::F64, 2, IntelSimdLevel::SSE) => {
                format!("_mm_loadu_pd({} + {})", target, index)
            }
            (DType::F64, 4, IntelSimdLevel::AVX) => {
                format!("_mm256_loadu_pd({} + {})", target, index)
            }
            _ => panic!("Unsupported Intel intrinsic configuration"),
        }
    }

    fn render_intel_store(
        &self,
        type_info: VectorTypeInfo,
        target: &str,
        index: &str,
        value: &str,
        level: IntelSimdLevel,
    ) -> String {
        match (type_info.base_type, type_info.vector_width, level) {
            (DType::F32, 4, IntelSimdLevel::SSE) => {
                format!("_mm_storeu_ps({} + {}, {})", target, index, value)
            }
            (DType::F32, 8, IntelSimdLevel::AVX) => {
                format!("_mm256_storeu_ps({} + {}, {})", target, index, value)
            }
            (DType::F32, 16, IntelSimdLevel::AVX512) => {
                format!("_mm512_storeu_ps({} + {}, {})", target, index, value)
            }
            (DType::F64, 2, IntelSimdLevel::SSE) => {
                format!("_mm_storeu_pd({} + {}, {})", target, index, value)
            }
            (DType::F64, 4, IntelSimdLevel::AVX) => {
                format!("_mm256_storeu_pd({} + {}, {})", target, index, value)
            }
            _ => panic!("Unsupported Intel intrinsic configuration"),
        }
    }
}
```

## 実装計画

### Phase 1: 基本機能（GCC Vector Extension）

**目標**: GCC/Clangベクトル拡張でSIMD型定義を生成

1. **型情報の追跡**
   - [ ] ASTノードにデータ型情報を追加（現在はF32固定）
   - [ ] Load/Storeノードから正しいDTypeを取得

2. **SimdConfig の実装**
   - [ ] `SimdExtension` enum の定義
   - [ ] `SimdConfig` struct の実装
   - [ ] `CRendererOption` への統合

3. **型定義生成**
   - [ ] `collect_vector_types()` の実装
   - [ ] `generate_vector_type_definitions()` の実装（GCC拡張のみ）

4. **テスト**
   - [ ] 単体テスト: 型定義生成
   - [ ] 統合テスト: ベクトル化されたコードのコンパイルと実行

### Phase 2: Intel Intrinsics サポート

**目標**: SSE/AVX/AVX-512の明示的なIntrinsicsサポート

1. **Intrinsics レンダリング**
   - [ ] `render_intel_load()` の実装
   - [ ] `render_intel_store()` の実装
   - [ ] 演算ノード（Add, Mul等）のIntrinsicsレンダリング

2. **設定プリセット**
   - [ ] `with_sse()`, `with_avx()`, `with_avx512()` の実装

3. **テスト**
   - [ ] SSE/AVX/AVX-512それぞれでのコンパイル・実行テスト

### Phase 3: 演算のSIMD化

**目標**: Load/Store以外の演算もSIMD化

1. **演算ノードの拡張**
   - [ ] Add, Mul, Sub のベクトル演算レンダリング
   - [ ] 比較演算（LessThan, Eq）のベクトル化
   - [ ] Select のベクトル化

2. **Intel Intrinsics 演算**
   - [ ] `_mm_add_ps()`, `_mm256_add_ps()` など
   - [ ] `_mm_mul_ps()`, `_mm256_mul_ps()` など

3. **GCC Vector Extension 演算**
   - [ ] ベクトル型の演算子オーバーロード利用

### Phase 4: その他のSIMD拡張

**目標**: ARM NEON、OpenCL、CUDAのサポート

1. **ARM NEON**
   - [ ] NEON型定義とIntrinsics

2. **OpenCL/CUDA**
   - [ ] 組み込みベクトル型の利用

### Phase 5: 自動ベクトル化フォールバック

**目標**: SIMD非対応環境でのフォールバック

1. **AutoVectorize モード**
   - [ ] 型定義なし、コンパイラ最適化に任せる
   - [ ] ループヒントの生成（`#pragma omp simd` など）

## 使用例

### 基本的な使用方法

```rust
use harp::backend::c::renderer::{CRenderer, CRendererOption};
use harp::backend::c::renderer::simd::{SimdConfig, SimdExtension};
use harp::backend::Renderer;

// デフォルト（GCC Vector Extension、AVX相当）
let mut renderer = CRenderer::new();
let option = CRendererOption::default();
renderer.with_option(option);

let code = renderer.render(ast);
```

### AVX-512 を使用

```rust
let mut renderer = CRenderer::new();
let option = CRendererOption::with_avx512();
renderer.with_option(option);

let code = renderer.render(ast);
```

### カスタム設定

```rust
let mut renderer = CRenderer::new();
let mut option = CRendererOption::default();

// F32のみベクトル化、F64は無効化
option.simd.type_configs.get_mut(&DType::F64).unwrap().allowed_widths.clear();

// 最大ベクトル幅を4に制限（SSE相当）
option.simd.max_vector_width = 4;

// Sin/Exp演算のSIMD化を無効化（精度の問題などで）
option.simd.disabled_operations.insert(SimdOperation::Sin);
option.simd.disabled_operations.insert(SimdOperation::Exp);

renderer.with_option(option);
let code = renderer.render(ast);
```

### SIMD完全無効化

```rust
let mut renderer = CRenderer::new();
let option = CRendererOption::without_simd();
renderer.with_option(option);

let code = renderer.render(ast);
```

## 生成コード例

### GCC Vector Extension

```c
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* SIMD vector type definitions */
typedef float float8 __attribute__((vector_size(32)));

void kernel_impl(float* input_0, float* input_1, float* output_0)
{
    for (size_t ridx0 = 0; ridx0 < 128; ridx0 += 1)
        *((float8*)(output_0 + ridx0)) = *((float8*)(input_0 + ridx0)) + *((float8*)(input_1 + ridx0));
}
```

### Intel AVX Intrinsics

```c
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

void kernel_impl(float* input_0, float* input_1, float* output_0)
{
    for (size_t ridx0 = 0; ridx0 < 128; ridx0 += 8)
        _mm256_storeu_ps(output_0 + ridx0,
            _mm256_add_ps(
                _mm256_loadu_ps(input_0 + ridx0),
                _mm256_loadu_ps(input_1 + ridx0)
            )
        );
}
```

## 代替案・将来の拡張

### 1. LLVM IR 生成

より移植性の高い中間表現として、LLVM IRを生成する選択肢もある：

```llvm
define void @kernel_impl(float* %input_0, float* %input_1, float* %output_0) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %addr0 = getelementptr float, float* %input_0, i64 %i
  %addr1 = getelementptr float, float* %input_1, i64 %i
  %addr_out = getelementptr float, float* %output_0, i64 %i

  %vec0 = load <8 x float>, <8 x float>* %addr0, align 32
  %vec1 = load <8 x float>, <8 x float>* %addr1, align 32
  %result = fadd <8 x float> %vec0, %vec1
  store <8 x float> %result, <8 x float>* %addr_out, align 32

  %i.next = add i64 %i, 8
  %cond = icmp ult i64 %i.next, 128
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
```

**利点**:
- アーキテクチャ非依存
- LLVM の強力な最適化を利用可能
- 複数のターゲット（x86, ARM, WebAssembly等）に対応

**欠点**:
- LLVMライブラリへの依存が必要
- デバッグが困難

### 2. ポリモーフィックなIntrinsics

ジェネリックなIntrinsicsラッパーを提供し、コンパイル時に適切な実装を選択：

```c
// コンパイル時に適切な実装が選択される
#ifdef __AVX2__
    #define SIMD_ADD_PS _mm256_add_ps
    #define SIMD_LOAD_PS _mm256_loadu_ps
    typedef __m256 simd_float;
#elif defined(__SSE__)
    #define SIMD_ADD_PS _mm_add_ps
    #define SIMD_LOAD_PS _mm_loadu_ps
    typedef __m128 simd_float;
#endif
```

### 3. 実行時ディスパッチ

実行時にCPU機能を検出し、最適な実装を選択：

```c
void kernel_impl(float* input_0, float* input_1, float* output_0) {
    if (cpu_has_avx512()) {
        kernel_impl_avx512(input_0, input_1, output_0);
    } else if (cpu_has_avx()) {
        kernel_impl_avx(input_0, input_1, output_0);
    } else {
        kernel_impl_scalar(input_0, input_1, output_0);
    }
}
```

## デバイス情報からのSIMD自動検出

### 概要

実行環境のCPU機能を自動検出し、利用可能な最適なSIMD拡張を選択する仕組み。これにより、ユーザーが手動で設定することなく、実行環境に最適化されたコードを生成できる。

### 検出方式

#### 1. コンパイル時検出（Compile-time Detection）

コンパイラが定義するマクロを利用して、ターゲット環境のSIMD機能を検出：

```rust
/// コンパイル時に利用可能なSIMD拡張を検出
pub fn detect_compile_time_simd() -> SimdExtension {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    return SimdExtension::IntelIntrinsics(IntelSimdLevel::AVX512);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    return SimdExtension::IntelIntrinsics(IntelSimdLevel::AVX);

    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.2"))]
    return SimdExtension::IntelIntrinsics(IntelSimdLevel::SSE);

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    return SimdExtension::ArmNeon;

    SimdExtension::AutoVectorize
}
```

**Cコード生成時:**
```c
// コンパイラマクロを利用した条件分岐
#ifdef __AVX512F__
    #define SIMD_LEVEL 3
    typedef __m512 simd_float;
#elif defined(__AVX2__)
    #define SIMD_LEVEL 2
    typedef __m256 simd_float;
#elif defined(__SSE4_2__)
    #define SIMD_LEVEL 1
    typedef __m128 simd_float;
#else
    #define SIMD_LEVEL 0
    typedef float simd_float;
#endif
```

**利点:**
- オーバーヘッドなし
- コンパイル時に最適化される
- 実装がシンプル

**欠点:**
- 実行環境が変わると最適化が合わない可能性
- ビルド時のターゲット設定が必要

#### 2. 実行時検出（Runtime Detection）

実行時にCPUID命令などを使用して、実際のCPU機能を検出：

**x86/x64 の場合:**

```rust
/// CPU機能を実行時に検出
pub struct CpuFeatures {
    pub has_sse: bool,
    pub has_sse2: bool,
    pub has_sse3: bool,
    pub has_ssse3: bool,
    pub has_sse4_1: bool,
    pub has_sse4_2: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_avx512dq: bool,
    pub has_fma: bool,
}

impl CpuFeatures {
    /// CPUID命令を使用してCPU機能を検出
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        // is_x86_feature_detected!マクロを使用（std提供）
        Self {
            has_sse: is_x86_feature_detected!("sse"),
            has_sse2: is_x86_feature_detected!("sse2"),
            has_sse3: is_x86_feature_detected!("sse3"),
            has_ssse3: is_x86_feature_detected!("ssse3"),
            has_sse4_1: is_x86_feature_detected!("sse4.1"),
            has_sse4_2: is_x86_feature_detected!("sse4.2"),
            has_avx: is_x86_feature_detected!("avx"),
            has_avx2: is_x86_feature_detected!("avx2"),
            has_avx512f: is_x86_feature_detected!("avx512f"),
            has_avx512dq: is_x86_feature_detected!("avx512dq"),
            has_fma: is_x86_feature_detected!("fma"),
        }
    }

    /// 最も高度なサポートされているSIMDレベルを取得
    pub fn best_simd_level(&self) -> SimdExtension {
        if self.has_avx512f {
            SimdExtension::IntelIntrinsics(IntelSimdLevel::AVX512)
        } else if self.has_avx2 {
            SimdExtension::IntelIntrinsics(IntelSimdLevel::AVX)
        } else if self.has_sse4_2 {
            SimdExtension::IntelIntrinsics(IntelSimdLevel::SSE)
        } else {
            SimdExtension::AutoVectorize
        }
    }

    /// 推奨されるベクトル幅を取得
    pub fn recommended_vector_width(&self, dtype: DType) -> usize {
        match dtype {
            DType::F32 => {
                if self.has_avx512f { 16 }
                else if self.has_avx { 8 }
                else if self.has_sse { 4 }
                else { 1 }
            }
            DType::F64 => {
                if self.has_avx512f { 8 }
                else if self.has_avx { 4 }
                else if self.has_sse2 { 2 }
                else { 1 }
            }
            _ => 1,
        }
    }
}
```

**ARM (AArch64) の場合:**

```rust
#[cfg(target_arch = "aarch64")]
impl CpuFeatures {
    pub fn detect() -> Self {
        // Linux: /proc/cpuinfo を読む、または getauxval() を使用
        // macOS: sysctlbyname() を使用
        Self {
            has_neon: Self::detect_neon(),
            has_sve: Self::detect_sve(),
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_neon() -> bool {
        use std::fs::read_to_string;
        if let Ok(cpuinfo) = read_to_string("/proc/cpuinfo") {
            cpuinfo.contains("neon")
        } else {
            false
        }
    }

    #[cfg(target_os = "macos")]
    fn detect_neon() -> bool {
        // AArch64 macOS では NEON は常にサポートされている
        true
    }
}
```

**生成されるCコード（実行時検出版）:**

```c
#include <cpuid.h>  // x86/x64
#include <stdbool.h>

typedef struct {
    bool has_sse;
    bool has_avx;
    bool has_avx2;
    bool has_avx512f;
} cpu_features_t;

// CPUID命令を使用してCPU機能を検出
cpu_features_t detect_cpu_features() {
    cpu_features_t features = {0};
    unsigned int eax, ebx, ecx, edx;

    // CPUID function 1: 基本機能
    __cpuid(1, eax, ebx, ecx, edx);
    features.has_sse = (edx & (1 << 25)) != 0;

    // CPUID function 7: 拡張機能
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    features.has_avx2 = (ebx & (1 << 5)) != 0;
    features.has_avx512f = (ebx & (1 << 16)) != 0;

    return features;
}

// 複数バージョンのカーネル実装
void kernel_impl_avx512(float* in0, float* in1, float* out);
void kernel_impl_avx(float* in0, float* in1, float* out);
void kernel_impl_sse(float* in0, float* in1, float* out);
void kernel_impl_scalar(float* in0, float* in1, float* out);

// 実行時ディスパッチ
void kernel_impl(float* in0, float* in1, float* out) {
    static cpu_features_t features;
    static bool detected = false;

    if (!detected) {
        features = detect_cpu_features();
        detected = true;
    }

    if (features.has_avx512f) {
        kernel_impl_avx512(in0, in1, out);
    } else if (features.has_avx2) {
        kernel_impl_avx(in0, in1, out);
    } else if (features.has_sse) {
        kernel_impl_sse(in0, in1, out);
    } else {
        kernel_impl_scalar(in0, in1, out);
    }
}
```

**利点:**
- 実行環境に最適な実装を自動選択
- 同じバイナリで複数の環境に対応
- ユーザー設定不要

**欠点:**
- コードサイズが増加（複数バージョンを生成）
- 若干のディスパッチオーバーヘッド
- 実装が複雑

### 自動設定API

```rust
impl SimdConfig {
    /// 実行環境を検出して自動設定
    pub fn auto_detect() -> Self {
        let features = CpuFeatures::detect();
        let extension = features.best_simd_level();

        let mut type_configs = HashMap::new();

        // F32の設定
        type_configs.insert(DType::F32, TypeSimdConfig {
            allowed_widths: vec![2, 4, 8, 16]
                .into_iter()
                .filter(|&w| w <= features.recommended_vector_width(DType::F32))
                .collect(),
            default_width: Some(features.recommended_vector_width(DType::F32)),
        });

        // F64の設定
        type_configs.insert(DType::F64, TypeSimdConfig {
            allowed_widths: vec![2, 4, 8]
                .into_iter()
                .filter(|&w| w <= features.recommended_vector_width(DType::F64))
                .collect(),
            default_width: Some(features.recommended_vector_width(DType::F64)),
        });

        Self {
            extension,
            max_vector_width: features.recommended_vector_width(DType::F32),
            type_configs,
            disabled_operations: HashSet::new(),
        }
    }

    /// 特定の機能が利用可能か確認
    pub fn is_feature_available(&self, feature: &str) -> bool {
        let features = CpuFeatures::detect();
        match feature {
            "sse" => features.has_sse,
            "sse2" => features.has_sse2,
            "avx" => features.has_avx,
            "avx2" => features.has_avx2,
            "avx512" => features.has_avx512f,
            "fma" => features.has_fma,
            _ => false,
        }
    }
}

impl CRendererOption {
    /// 自動検出した設定で初期化
    pub fn auto_detect() -> Self {
        Self {
            use_openmp: true,
            simd: SimdConfig::auto_detect(),
        }
    }
}
```

### 実行時ディスパッチ生成

実行時ディスパッチを有効にした場合、複数バージョンのカーネルを生成：

```rust
#[derive(Debug, Clone)]
pub struct RuntimeDispatchConfig {
    /// 生成するバージョン
    pub variants: Vec<SimdExtension>,

    /// ディスパッチ関数名
    pub dispatch_function_name: String,

    /// 検出を一度だけ実行（static変数に保存）
    pub cache_detection: bool,
}

impl CRendererOption {
    /// 実行時ディスパッチを有効化
    pub fn with_runtime_dispatch(variants: Vec<SimdExtension>) -> Self {
        Self {
            use_openmp: true,
            simd: SimdConfig {
                extension: SimdExtension::RuntimeDispatch(RuntimeDispatchConfig {
                    variants,
                    dispatch_function_name: "detect_and_dispatch".to_string(),
                    cache_detection: true,
                }),
                ..Default::default()
            },
        }
    }
}
```

**生成コード例:**

```c
// それぞれのSIMDレベルに対応した実装
void kernel_impl_avx512(float* in0, float* in1, float* out, size_t n) {
    for (size_t i = 0; i < n; i += 16) {
        _mm512_storeu_ps(out + i,
            _mm512_add_ps(
                _mm512_loadu_ps(in0 + i),
                _mm512_loadu_ps(in1 + i)
            )
        );
    }
}

void kernel_impl_avx(float* in0, float* in1, float* out, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        _mm256_storeu_ps(out + i,
            _mm256_add_ps(
                _mm256_loadu_ps(in0 + i),
                _mm256_loadu_ps(in1 + i)
            )
        );
    }
}

void kernel_impl_scalar(float* in0, float* in1, float* out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = in0[i] + in1[i];
    }
}

// ディスパッチャー
void kernel_impl(float* in0, float* in1, float* out, size_t n) {
    static int simd_level = -1;

    // 初回のみ検出
    if (simd_level == -1) {
        cpu_features_t features = detect_cpu_features();
        if (features.has_avx512f) {
            simd_level = 2;
        } else if (features.has_avx) {
            simd_level = 1;
        } else {
            simd_level = 0;
        }
    }

    // 適切な実装にディスパッチ
    switch (simd_level) {
        case 2: kernel_impl_avx512(in0, in1, out, n); break;
        case 1: kernel_impl_avx(in0, in1, out, n); break;
        default: kernel_impl_scalar(in0, in1, out, n); break;
    }
}
```

### 使用例

```rust
// 方法1: 完全自動検出
let mut renderer = CRenderer::new();
let option = CRendererOption::auto_detect();
renderer.with_option(option);

// 方法2: 実行時ディスパッチ（複数バージョン生成）
let option = CRendererOption::with_runtime_dispatch(vec![
    SimdExtension::IntelIntrinsics(IntelSimdLevel::AVX512),
    SimdExtension::IntelIntrinsics(IntelSimdLevel::AVX),
    SimdExtension::IntelIntrinsics(IntelSimdLevel::SSE),
]);

// 方法3: 検出結果を確認してカスタマイズ
let mut config = SimdConfig::auto_detect();
println!("Detected SIMD: {:?}", config.extension);

// 特定の演算を無効化（例：精度の問題）
if !config.is_feature_available("fma") {
    config.disabled_operations.insert(SimdOperation::Mul);
}
```

### プラットフォーム別の検出方法

| プラットフォーム | 検出方法 | 利用可能な情報 |
|---|---|---|
| **Linux x86_64** | `is_x86_feature_detected!` macro, `/proc/cpuinfo` | SSE, AVX, AVX2, AVX-512, FMA |
| **Windows x86_64** | `IsProcessorFeaturePresent`, `__cpuid` | 同上 |
| **macOS x86_64** | `sysctlbyname("hw.optional.*")` | 同上 |
| **macOS ARM64** | `sysctlbyname` | NEON（常時有効） |
| **Linux ARM64** | `/proc/cpuinfo`, `getauxval(AT_HWCAP)` | NEON, SVE |
| **Android ARM** | `android_getCpuFeatures()` | NEON, FP16 |

### Phase 6: デバイス自動検出機能（将来実装）

実装計画に追加：

1. **CPU機能検出ライブラリの実装**
   - [ ] `CpuFeatures` 構造体の実装
   - [ ] プラットフォーム別の検出コード
   - [ ] テスト（様々なCPUでの検証）

2. **自動設定API**
   - [ ] `SimdConfig::auto_detect()` の実装
   - [ ] `CRendererOption::auto_detect()` の実装

3. **実行時ディスパッチ生成**
   - [ ] 複数バージョンのカーネル生成
   - [ ] ディスパッチャー関数の生成
   - [ ] CPU検出コードの生成

4. **ドキュメントとサンプル**
   - [ ] 自動検出機能の使用方法
   - [ ] パフォーマンスベンチマーク

### 注意事項

1. **セキュリティ**: CPUID命令は一部の仮想化環境でエミュレートされる場合がある
2. **精度**: 一部のSIMD演算（特に超越関数）は精度が異なる場合がある
3. **互換性**: 古いCPUでは一部の命令が使用できない可能性
4. **キャッシュ**: 検出結果はキャッシュすることでオーバーヘッドを最小化

## 参考資料

- [GCC Vector Extensions](https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [OpenCL Vector Types](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_C.html#vector-data-types)
- [Rust std::arch module](https://doc.rust-lang.org/std/arch/index.html) - CPU機能検出
- [CPUID Wikipedia](https://en.wikipedia.org/wiki/CPUID) - CPUID命令の詳細
