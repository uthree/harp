//! デバイス識別子
//!
//! PyTorchの`torch.device`に相当する機能を提供します。
//! デバイスの利用可否チェックや転送先指定に使用します。

use std::fmt;

/// デバイス識別子
///
/// 計算を実行するデバイスを指定するためのenum。
/// 各バリアントは対応するfeatureが有効な場合のみ利用可能です。
///
/// # 例
///
/// ```ignore
/// use harp_lazy_array::Device;
///
/// // デバイスの利用可否チェック
/// if Device::Metal.is_available() {
///     println!("Metal is available!");
/// }
///
/// // 配列をデバイスに転送
/// let arr = Array2::zeros([3, 4]);
/// let metal_arr = arr.to(Device::Metal)?;
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    /// Apple Metal (macOS/iOS)
    Metal,
    /// OpenCL (クロスプラットフォーム)
    OpenCL,
}

impl Device {
    /// デバイス名を取得
    ///
    /// # 例
    ///
    /// ```
    /// use harp_lazy_array::Device;
    ///
    /// assert_eq!(Device::Metal.name(), "metal");
    /// assert_eq!(Device::OpenCL.name(), "opencl");
    /// ```
    pub fn name(&self) -> &'static str {
        match self {
            Device::Metal => "metal",
            Device::OpenCL => "opencl",
        }
    }

    /// このデバイスが利用可能かチェック
    ///
    /// 対応するfeatureが有効な場合に`true`を返します。
    ///
    /// # 例
    ///
    /// ```ignore
    /// use harp_lazy_array::Device;
    ///
    /// if Device::Metal.is_available() {
    ///     let arr = Array2::zeros([3, 4]).to(Device::Metal)?;
    /// }
    /// ```
    #[allow(unreachable_code)]
    pub fn is_available(&self) -> bool {
        match self {
            Device::Metal => {
                #[cfg(feature = "metal")]
                return true;
                false
            }
            Device::OpenCL => {
                #[cfg(feature = "opencl")]
                return true;
                false
            }
        }
    }

    /// 利用可能なデバイス一覧を取得
    ///
    /// # 例
    ///
    /// ```ignore
    /// use harp_lazy_array::Device;
    ///
    /// for device in Device::available() {
    ///     println!("{} is available", device.name());
    /// }
    /// ```
    pub fn available() -> Vec<Device> {
        [Device::Metal, Device::OpenCL]
            .into_iter()
            .filter(|d| d.is_available())
            .collect()
    }

    /// デフォルトデバイスを取得
    ///
    /// 優先順位: Metal > OpenCL
    ///
    /// # Panics
    ///
    /// どのバックエンドも利用可能でない場合にパニックします。
    #[allow(unreachable_code)]
    pub fn default_device() -> Device {
        #[cfg(feature = "metal")]
        return Device::Metal;

        #[cfg(all(feature = "opencl", not(feature = "metal")))]
        return Device::OpenCL;

        panic!("No backend available. Enable at least one backend feature: metal or opencl");
    }

    /// 全てのデバイスバリアントを取得
    pub fn all() -> &'static [Device] {
        &[Device::Metal, Device::OpenCL]
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::default_device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_name() {
        assert_eq!(Device::Metal.name(), "metal");
        assert_eq!(Device::OpenCL.name(), "opencl");
    }

    #[test]
    fn test_device_display() {
        assert_eq!(format!("{}", Device::Metal), "metal");
        assert_eq!(format!("{}", Device::OpenCL), "opencl");
    }

    #[test]
    fn test_device_equality() {
        assert_eq!(Device::Metal, Device::Metal);
        assert_ne!(Device::Metal, Device::OpenCL);
    }

    #[test]
    fn test_device_all() {
        let all = Device::all();
        assert_eq!(all.len(), 2);
        assert!(all.contains(&Device::Metal));
        assert!(all.contains(&Device::OpenCL));
    }
}
