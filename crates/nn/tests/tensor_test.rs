//! Tensor型の結合テスト

use harp_nn::tensor::{
    Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, TensorD, TensorInit, TensorRandInit,
};

mod initialization {
    use super::*;

    #[test]
    fn test_tensor2_zeros_f32() {
        let tensor = <Tensor2<f32>>::zeros([3, 4]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[3, 4]);
    }

    #[test]
    fn test_tensor2_ones_f32() {
        let tensor = <Tensor2<f32>>::ones([3, 4]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[3, 4]);
    }

    #[test]
    fn test_tensor2_zeros_i32() {
        let tensor = <Tensor2<i32>>::zeros([2, 5]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[2, 5]);
    }

    #[test]
    fn test_tensor2_ones_i32() {
        let tensor = <Tensor2<i32>>::ones([2, 5]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[2, 5]);
    }

    #[test]
    fn test_tensor1_zeros() {
        let tensor = <Tensor1<f32>>::zeros([10]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[10]);
    }

    #[test]
    fn test_tensor3_zeros() {
        let tensor = <Tensor3<f32>>::zeros([2, 3, 4]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_tensor4_zeros() {
        let tensor = <Tensor4<f32>>::zeros([2, 3, 4, 5]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[2, 3, 4, 5]);
    }

    #[test]
    fn test_tensord_zeros() {
        let tensor = <TensorD<f32>>::zeros([2, 3, 4, 5, 6]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_tensor0_zeros() {
        let tensor = <Tensor0<f32>>::zeros([]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[]);
    }
}

mod rand_initialization {
    use super::*;

    #[test]
    fn test_tensor2_rand() {
        let tensor = <Tensor2<f32>>::rand([3, 4]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[3, 4]);
    }

    #[test]
    fn test_tensor1_rand() {
        let tensor = <Tensor1<f32>>::rand([10]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[10]);
    }

    #[test]
    fn test_tensor3_rand() {
        let tensor = <Tensor3<f32>>::rand([2, 3, 4]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_tensord_rand() {
        let tensor = <TensorD<f32>>::rand([2, 3, 4, 5]);
        let lazy_array = tensor.value();
        assert_eq!(lazy_array.shape(), &[2, 3, 4, 5]);
    }
}

mod type_aliases {
    use super::*;
    use std::any::TypeId;

    #[test]
    fn test_type_aliases_are_distinct() {
        // 各型エイリアスが正しく異なる次元を持つことを確認
        assert_ne!(TypeId::of::<Tensor1<f32>>(), TypeId::of::<Tensor2<f32>>());
        assert_ne!(TypeId::of::<Tensor2<f32>>(), TypeId::of::<Tensor3<f32>>());
    }
}
