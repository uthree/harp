# バックエンド

## 概要

Eclatは6つのバックエンドをサポート。プラグイン可能なアーキテクチャ。

| バックエンド | ターゲット | 出力 | 用途 |
|--------------|------------|------|------|
| C | CPU | C言語 | シーケンシャル実行 |
| OpenMP | CPU | C + pragma | マルチスレッド並列 |
| Rust | CPU | Rust cdylib | Rustネイティブ |
| CUDA | NVIDIA GPU | CUDA | GPU並列 |
| Metal | Apple GPU | Metal Shading | macOS GPU |
| OpenCL | 汎用GPU | OpenCL | クロスプラットフォームGPU |

## トレイト定義

### Device

デバイス抽象化。

```rust
pub trait Device: Send + Sync {
    fn allocate_buffer(&self, size: usize, dtype: DType) -> Result<Box<dyn Buffer>>;
    fn profile(&self) -> DeviceProfile;
    fn name(&self) -> &str;
    fn device_type(&self) -> DeviceType;
}
```

### Buffer

メモリバッファ。

```rust
pub trait Buffer: Send + Sync {
    fn copy_from(&mut self, data: &[u8]) -> Result<()>;
    fn copy_to(&self, data: &mut [u8]) -> Result<()>;
    fn size(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
}
```

### Compiler

コンパイラ。

```rust
pub trait Compiler {
    fn compile(&self, ast: &AstNode) -> Result<Box<dyn Kernel>>;
}
```

### Renderer

コード生成。

```rust
pub trait Renderer: Send + Sync {
    fn render(&self, ast: &AstNode, opt_level: OptimizationLevel) -> String;
}
```

### Kernel

実行可能カーネル。

```rust
pub trait Kernel: Send + Sync {
    fn call(&self, args: &[Box<dyn Buffer>]) -> Result<()>;
}
```

## DeviceProfile

ハードウェア特性。

```rust
pub struct DeviceProfile {
    pub device_type: DeviceType,        // CPU, GPU, Accelerator
    pub compute_units: usize,           // 計算ユニット数
    pub max_work_group_size: usize,     // 最大ワークグループサイズ
    pub local_memory_size: usize,       // ローカルメモリサイズ
    pub warp_size: usize,               // ワープ/ウェーブサイズ
    pub preferred_tile_sizes: Vec<usize>,
    pub simd_capabilities: SimdCapabilities,
}
```

## 各バックエンドの詳細

### C バックエンド

シーケンシャルなC言語コード生成。

```c
// 出力例
void kernel_0(float* buf0, float* buf1, long long n) {
    for (long long i = 0; i < n; i++) {
        buf1[i] = buf0[i] * 2.0f;
    }
}
```

### OpenMP バックエンド

OpenMPプラグマ付きCコード。

```c
void kernel_0(float* buf0, float* buf1, long long n) {
    #pragma omp parallel for
    for (long long i = 0; i < n; i++) {
        buf1[i] = buf0[i] * 2.0f;
    }
}
```

### CUDA バックエンド

CUDAカーネル生成。

```cuda
__global__ void kernel_0(float* buf0, float* buf1, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        buf1[i] = buf0[i] * 2.0f;
    }
}
```

スレッド階層:
- `global_id`: グローバルスレッドID
- `local_id`: ブロック内スレッドID
- `group_id`: ブロックID

**Tensor Core サポート**:
- WMMA API を使用した16x16x16タイル行列積
- 対応アーキテクチャ: Volta以降 (sm_70+)
- 入力: F16, 累積/出力: F32

### Metal バックエンド

Metal Shading Language生成。

```metal
kernel void kernel_0(
    device float* buf0 [[buffer(0)]],
    device float* buf1 [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    buf1[i] = buf0[i] * 2.0f;
}
```

### OpenCL バックエンド

OpenCLカーネル生成。

```opencl
__kernel void kernel_0(
    __global float* buf0,
    __global float* buf1,
    long n
) {
    long i = get_global_id(0);
    if (i < n) {
        buf1[i] = buf0[i] * 2.0f;
    }
}
```

### Rust バックエンド

Rust cdylib生成。

```rust
#[no_mangle]
pub extern "C" fn kernel_0(buf0: *const f32, buf1: *mut f32, n: i64) {
    for i in 0..n as usize {
        unsafe {
            *buf1.add(i) = *buf0.add(i) * 2.0;
        }
    }
}
```

## 型マッピング

| DType | C/OpenMP | CUDA | Metal | OpenCL | Rust |
|-------|----------|------|-------|--------|------|
| Bool | `_Bool` | `bool` | `bool` | `bool` | `bool` |
| I8 | `signed char` | `char` | `char` | `char` | `i8` |
| I32 | `int` | `int` | `int` | `int` | `i32` |
| I64 | `long long` | `long long` | `long` | `long` | `i64` |
| F16 | `_Float16` | `half` | `half` | `half` | `half::f16` |
| F32 | `float` | `float` | `float` | `float` | `f32` |
| F64 | `double` | `double` | `double`* | `double` | `f64` |

*Metalではdoubleのサポートは限定的

## デバイス選択

```rust
use eclat::backend::{set_device_str, get_device};

// 文字列で指定
set_device_str("cuda:0")?;      // CUDA GPU 0
set_device_str("metal:0")?;     // Metal GPU 0
set_device_str("opencl:0")?;    // OpenCL device 0
set_device_str("cpu")?;         // CPU (OpenMP)

// 現在のデバイス取得
let device = get_device();
```

## 実行パイプライン

```
AST
 ↓ Renderer
コード文字列
 ↓ Compiler (clang, nvcc, etc.)
共有ライブラリ / GPU binary
 ↓ Kernel::call()
実行
```
