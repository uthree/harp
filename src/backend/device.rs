//! デバイス抽象化レイヤー
//!
//! PyTorchの`torch.device`に相当する機能を提供します。
//! デバイスごとにPipelineをシングルトンとして管理し、
//! コンパイル済みカーネルのキャッシュを効率的に共有します。

use crate::backend::{Compiler, Pipeline};
use crate::graph::Graph;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[cfg(target_os = "macos")]
use crate::backend::metal::{MetalCompiler, MetalKernel, MetalPipeline, MetalRenderer};
use crate::backend::{
    c::{CBuffer, CCompiler, CKernel, CPipeline, CRenderer},
    opencl::{OpenCLBuffer, OpenCLCompiler, OpenCLKernel, OpenCLPipeline, OpenCLRenderer},
};

/// デバイスの種類
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPUバックエンド (C/C++)
    Cpu,
    /// Metalバックエンド (macOS GPU)
    #[cfg(target_os = "macos")]
    Metal,
    /// OpenCLバックエンド (クロスプラットフォームGPU)
    OpenCL,
}

/// デバイス指定（PyTorchの torch.device に相当）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    pub device_type: DeviceType,
    /// デバイスインデックス（将来の複数GPU対応用）
    pub index: usize,
}

impl Default for Device {
    /// デフォルトデバイスを取得（利用可能な最速のデバイス）
    fn default() -> Self {
        Self::auto_select()
    }
}

impl Device {
    /// 利用可能なデバイスを自動選択
    /// 優先順位: Metal > OpenCL > CPU
    pub fn auto_select() -> Self {
        #[cfg(target_os = "macos")]
        {
            let metal = MetalCompiler::new();
            if metal.is_available() {
                return Self::metal(0);
            }
        }

        let opencl = OpenCLCompiler::new();
        if opencl.is_available() {
            return Self::opencl(0);
        }

        Self::cpu()
    }

    /// CPUデバイスを作成
    pub fn cpu() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            index: 0,
        }
    }

    /// Metalデバイスを作成 (macOSのみ)
    #[cfg(target_os = "macos")]
    pub fn metal(index: usize) -> Self {
        Self {
            device_type: DeviceType::Metal,
            index,
        }
    }

    /// OpenCLデバイスを作成
    pub fn opencl(index: usize) -> Self {
        Self {
            device_type: DeviceType::OpenCL,
            index,
        }
    }

    /// デバイスが利用可能かチェック
    pub fn is_available(&self) -> bool {
        match self.device_type {
            DeviceType::Cpu => CCompiler::new().is_available(),
            #[cfg(target_os = "macos")]
            DeviceType::Metal => MetalCompiler::new().is_available(),
            DeviceType::OpenCL => OpenCLCompiler::new().is_available(),
        }
    }

    /// このデバイスに対応する共有Pipelineを取得
    ///
    /// 同じデバイスに対しては常に同じPipelineインスタンスが返されるため、
    /// コンパイル済みカーネルのキャッシュが効率的に共有されます。
    ///
    /// 注意: スレッドローカルなシングルトンを使用しているため、
    /// 各スレッドで独立したPipelineインスタンスが作成されます。
    pub fn get_pipeline(&self) -> Result<SharedPipeline, String> {
        DeviceManager::get()
            .borrow_mut()
            .get_or_create_pipeline(*self)
    }
}

// 文字列からのパース
impl std::str::FromStr for Device {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();
        let device_type = match parts[0] {
            "cpu" => DeviceType::Cpu,
            #[cfg(target_os = "macos")]
            "metal" => DeviceType::Metal,
            "opencl" => DeviceType::OpenCL,
            _ => return Err(format!("Unknown device type: {}", parts[0])),
        };

        let index = if parts.len() > 1 {
            parts[1]
                .parse()
                .map_err(|_| "Invalid device index".to_string())?
        } else {
            0
        };

        Ok(Self { device_type, index })
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.device_type {
            DeviceType::Cpu => write!(f, "cpu:{}", self.index),
            #[cfg(target_os = "macos")]
            DeviceType::Metal => write!(f, "metal:{}", self.index),
            DeviceType::OpenCL => write!(f, "opencl:{}", self.index),
        }
    }
}

/// デバイス非依存のPipelineラッパー
///
/// 内部で各バックエンドのPipelineを保持し、統一的なインターフェースを提供します。
pub enum DevicePipeline {
    Cpu(CPipeline),
    #[cfg(target_os = "macos")]
    Metal(MetalPipeline),
    OpenCL(OpenCLPipeline),
}

impl DevicePipeline {
    /// 指定されたデバイスでPipelineを作成
    fn new(device: Device) -> Result<Self, String> {
        if !device.is_available() {
            return Err(format!("Device {} is not available", device));
        }

        match device.device_type {
            DeviceType::Cpu => {
                let renderer = CRenderer::new();
                let compiler = CCompiler::new();
                Ok(Self::Cpu(CPipeline::new(renderer, compiler)))
            }
            #[cfg(target_os = "macos")]
            DeviceType::Metal => {
                let renderer = MetalRenderer::new();
                let compiler = MetalCompiler::new();
                Ok(Self::Metal(MetalPipeline::new(renderer, compiler)))
            }
            DeviceType::OpenCL => {
                let renderer = OpenCLRenderer::new();
                let compiler = OpenCLCompiler::new();
                Ok(Self::OpenCL(OpenCLPipeline::new(renderer, compiler)))
            }
        }
    }

    /// グラフをコンパイル
    pub fn compile_graph(&mut self, graph: Graph) -> Result<DeviceKernel, String> {
        match self {
            Self::Cpu(pipeline) => {
                let kernel = pipeline.compile_graph(graph)?;
                Ok(DeviceKernel::Cpu(kernel))
            }
            #[cfg(target_os = "macos")]
            Self::Metal(pipeline) => {
                let kernel = pipeline.compile_graph(graph)?;
                Ok(DeviceKernel::Metal(kernel))
            }
            Self::OpenCL(pipeline) => {
                let kernel = pipeline.compile_graph(graph)?;
                Ok(DeviceKernel::OpenCL(kernel))
            }
        }
    }

    /// キャッシュ済みカーネルが存在するか確認
    pub fn has_cached_kernel(&self, key: &str) -> bool {
        match self {
            Self::Cpu(pipeline) => pipeline.get_cached_kernel(key).is_some(),
            #[cfg(target_os = "macos")]
            Self::Metal(pipeline) => pipeline.get_cached_kernel(key).is_some(),
            Self::OpenCL(pipeline) => pipeline.get_cached_kernel(key).is_some(),
        }
    }

    /// グラフをコンパイルしてキャッシュ
    pub fn compile_and_cache(&mut self, key: String, graph: Graph) -> Result<(), String> {
        match self {
            Self::Cpu(pipeline) => {
                pipeline.compile_and_cache(key, graph)?;
                Ok(())
            }
            #[cfg(target_os = "macos")]
            Self::Metal(pipeline) => {
                pipeline.compile_and_cache(key, graph)?;
                Ok(())
            }
            Self::OpenCL(pipeline) => {
                pipeline.compile_and_cache(key, graph)?;
                Ok(())
            }
        }
    }

    /// キャッシュをクリア
    pub fn clear_cache(&mut self) {
        match self {
            Self::Cpu(pipeline) => pipeline.clear_cache(),
            #[cfg(target_os = "macos")]
            Self::Metal(pipeline) => pipeline.clear_cache(),
            Self::OpenCL(pipeline) => pipeline.clear_cache(),
        }
    }
}

/// 共有可能なPipeline（同一スレッド内で共有）
///
/// `Rc<RefCell<>>`でラップされているため、同一スレッド内で効率的に共有できます。
/// 注意: `Graph`が`Rc`を使用しているため、スレッド間での共有は不可能です。
pub type SharedPipeline = Rc<RefCell<DevicePipeline>>;

/// デバイスマネージャー（スレッドローカルシングルトン）
///
/// デバイスごとにPipelineインスタンスを管理し、同じデバイスに対しては
/// 常に同じPipelineを返すことで、カーネルキャッシュを効率的に共有します。
///
/// 注意: `Graph`が`Rc`を使用しているため、各スレッドで独立したインスタンスを保持します。
struct DeviceManager {
    pipelines: HashMap<Device, SharedPipeline>,
}

impl DeviceManager {
    /// 新しいDeviceManagerを作成
    fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }

    /// スレッドローカルなシングルトンインスタンスを取得
    fn get() -> &'static RefCell<Self> {
        thread_local! {
            static INSTANCE: RefCell<DeviceManager> = RefCell::new(DeviceManager::new());
        }
        INSTANCE.with(|instance| unsafe {
            // Safety: スレッドローカルなので、この参照は有効
            &*(instance as *const RefCell<DeviceManager>)
        })
    }

    /// デバイスに対応するPipelineを取得（存在しない場合は作成）
    fn get_or_create_pipeline(&mut self, device: Device) -> Result<SharedPipeline, String> {
        // 既に存在する場合はそれを返す
        if let Some(pipeline) = self.pipelines.get(&device) {
            return Ok(Rc::clone(pipeline));
        }

        // 新規作成
        let pipeline = DevicePipeline::new(device)?;
        let shared = Rc::new(RefCell::new(pipeline));
        self.pipelines.insert(device, Rc::clone(&shared));

        Ok(shared)
    }

    /// 全てのPipelineキャッシュをクリア
    #[allow(dead_code)]
    fn clear_all_caches(&mut self) {
        for pipeline in self.pipelines.values() {
            pipeline.borrow_mut().clear_cache();
        }
    }

    /// 登録されているデバイス数を取得
    #[allow(dead_code)]
    fn device_count(&self) -> usize {
        self.pipelines.len()
    }
}

/// デバイス非依存のKernelラッパー
pub enum DeviceKernel {
    Cpu(CKernel),
    #[cfg(target_os = "macos")]
    Metal(MetalKernel),
    OpenCL(OpenCLKernel),
}

/// デバイス非依存のBufferラッパー
pub enum DeviceBuffer {
    Cpu(CBuffer),
    #[cfg(target_os = "macos")]
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    Metal(crate::backend::metal::MetalBuffer),
    OpenCL(OpenCLBuffer),
}

impl DeviceBuffer {
    /// CPUにデータをコピー
    pub fn to_cpu(&self) -> Result<Vec<u8>, String> {
        use crate::backend::Buffer;
        match self {
            Self::Cpu(b) => Ok(b.to_bytes()),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => Ok(b.to_bytes()),
            Self::OpenCL(b) => Ok(b.to_bytes()),
        }
    }

    /// 型付きベクタとしてデータを取得
    pub fn to_vec<T: Clone + 'static>(&self) -> Result<Vec<T>, String> {
        use crate::backend::Buffer;
        match self {
            Self::Cpu(b) => b.to_vec(),
            #[cfg(target_os = "macos")]
            Self::Metal(b) => b.to_vec(),
            Self::OpenCL(b) => b.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        let cpu = Device::cpu();
        assert_eq!(cpu.device_type, DeviceType::Cpu);
        assert_eq!(cpu.index, 0);

        #[cfg(target_os = "macos")]
        {
            let metal = Device::metal(0);
            assert_eq!(metal.device_type, DeviceType::Metal);
            assert_eq!(metal.index, 0);
        }

        let opencl = Device::opencl(0);
        assert_eq!(opencl.device_type, DeviceType::OpenCL);
        assert_eq!(opencl.index, 0);
    }

    #[test]
    fn test_device_from_str() {
        let cpu: Device = "cpu:0".parse().unwrap();
        assert_eq!(cpu.device_type, DeviceType::Cpu);
        assert_eq!(cpu.index, 0);

        let cpu_default: Device = "cpu".parse().unwrap();
        assert_eq!(cpu_default.index, 0);

        #[cfg(target_os = "macos")]
        {
            let metal: Device = "metal:0".parse().unwrap();
            assert_eq!(metal.device_type, DeviceType::Metal);
        }

        let opencl: Device = "opencl:1".parse().unwrap();
        assert_eq!(opencl.device_type, DeviceType::OpenCL);
        assert_eq!(opencl.index, 1);

        assert!("invalid:0".parse::<Device>().is_err());
    }

    #[test]
    fn test_device_display() {
        let cpu = Device::cpu();
        assert_eq!(cpu.to_string(), "cpu:0");

        #[cfg(target_os = "macos")]
        {
            let metal = Device::metal(0);
            assert_eq!(metal.to_string(), "metal:0");
        }

        let opencl = Device::opencl(1);
        assert_eq!(opencl.to_string(), "opencl:1");
    }

    #[test]
    fn test_device_auto_select() {
        let device = Device::auto_select();
        // 自動選択されたデバイスが利用可能であることを確認
        assert!(device.is_available());
    }

    #[test]
    fn test_device_singleton() {
        // 同じデバイスに対して同じPipelineが返されることを確認
        let device = Device::cpu();

        let pipeline1 = device.get_pipeline().unwrap();
        let pipeline2 = device.get_pipeline().unwrap();

        // Rc::ptrでポインタを比較
        assert!(Rc::ptr_eq(&pipeline1, &pipeline2));
    }

    #[test]
    fn test_device_cache_sharing() {
        use crate::graph::DType;

        let device = Device::cpu();
        let pipeline = device.get_pipeline().unwrap();

        // キャッシュが空であることを確認
        assert!(!pipeline.borrow().has_cached_kernel("test_kernel"));

        // グラフを作成してコンパイル
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let c = a + b;
        graph.output("result", c);

        // コンパイルしてキャッシュ
        let key = "test_kernel".to_string();
        pipeline
            .borrow_mut()
            .compile_and_cache(key.clone(), graph)
            .unwrap();

        // キャッシュに追加されていることを確認
        assert!(pipeline.borrow().has_cached_kernel(&key));

        // 同じデバイスから再度Pipelineを取得してもキャッシュが共有されていることを確認
        let pipeline2 = device.get_pipeline().unwrap();
        assert!(pipeline2.borrow().has_cached_kernel(&key));
    }
}
