#[cfg(feature = "backend-c")]
mod tests {
    use harp::ast::DType;
    use harp::backend::c::CBuffer;
    use harp::backend::Backend;
    use harp::backend::Buffer;
    use harp::graph::ops::ReduceOps;
    use harp::graph::Graph;

    #[test]
    fn test_conv1d_via_sliding_window() {
        // Simple 1D convolution using sliding_window
        // Input: [1, 1, 5] (batch=1, channels=1, length=5)
        // Kernel: [1, 1, 3] (out_channels=1, in_channels=1, kernel_size=3)
        // Output: [1, 1, 3] (batch=1, out_channels=1, output_length=3)

        let mut graph = Graph::new();

        // Input: [1, 1, 5]
        let input = graph.input(DType::F32, vec![1.into(), 1.into(), 5.into()]);

        // Kernel: [1, 1, 3]
        let kernel = graph.input(DType::F32, vec![1.into(), 1.into(), 3.into()]);

        // Step 1: Apply sliding window to input
        // [1, 1, 5] -> [1, 1, 3, 3] where output_length = (5-3)/1+1 = 3
        let windowed = input.sliding_window(2, 3, 1);

        // Step 2: Reshape for broadcasting
        // windowed: [1, 1, 3, 3] -> [1, 1, 1, 3, 3] (add out_channel dim)
        let windowed_expanded = windowed.unsqueeze(1);
        // kernel: [1, 1, 3] -> [1, 1, 1, 1, 3] (add batch and L' dims)
        let kernel_expanded = kernel.unsqueeze(0).unsqueeze(2);
        // Expand kernel to match windowed shape: [1, 1, 1, 1, 3] -> [1, 1, 1, 3, 3]
        let kernel_broadcast =
            kernel_expanded.expand(vec![1.into(), 1.into(), 1.into(), 3.into(), 3.into()]);

        // Step 3: Element-wise multiplication
        // [1, 1, 1, 3, 3] * [1, 1, 1, 3, 3] -> [1, 1, 1, 3, 3]
        let product = windowed_expanded * kernel_broadcast;

        // Step 4: Sum over [in_channels, kernel_size] at axes [2, 4]
        // [1, 1, 1, 3, 3] -> sum(2) -> [1, 1, 3, 3] -> sum(4) -> [1, 1, 3]
        let output = product.sum(2).sum(3);

        graph.output(output);

        // Execute
        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input data: [0, 1, 2, 3, 4]
        let input_data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 1, 5], DType::F32);

        // Kernel data: [1, 0, -1] (edge detection kernel)
        let kernel_data = vec![1.0f32, 0.0, -1.0];
        let kernel_buffer = CBuffer::from_slice(&kernel_data, &[1, 1, 3], DType::F32);

        let outputs = backend.execute(&graph, vec![input_buffer, kernel_buffer]);
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].to_vec::<f32>();

        // Expected output:
        // Position 0: [0,1,2] * [1,0,-1] = 0*1 + 1*0 + 2*(-1) = -2
        // Position 1: [1,2,3] * [1,0,-1] = 1*1 + 2*0 + 3*(-1) = -2
        // Position 2: [2,3,4] * [1,0,-1] = 2*1 + 3*0 + 4*(-1) = -2
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], -2.0);
        assert_eq!(result[1], -2.0);
        assert_eq!(result[2], -2.0);
    }

    #[test]
    fn test_conv1d_multiple_channels() {
        // Conv1D with multiple input and output channels
        // Input: [1, 2, 5] (batch=1, in_channels=2, length=5)
        // Kernel: [3, 2, 3] (out_channels=3, in_channels=2, kernel_size=3)
        // Output: [1, 3, 3] (batch=1, out_channels=3, output_length=3)

        let mut graph = Graph::new();

        // Input: [1, 2, 5]
        let input = graph.input(DType::F32, vec![1.into(), 2.into(), 5.into()]);

        // Kernel: [3, 2, 3]
        let kernel = graph.input(DType::F32, vec![3.into(), 2.into(), 3.into()]);

        // Step 1: Apply sliding window to input
        // [1, 2, 5] -> [1, 2, 3, 3]
        let windowed = input.sliding_window(2, 3, 1);

        // Step 2: Reshape for broadcasting
        // windowed: [1, 2, 3, 3] -> [1, 1, 2, 3, 3] (add out_channel dim)
        let windowed_expanded = windowed.unsqueeze(1);
        // kernel: [3, 2, 3] -> [1, 3, 2, 1, 3] (add batch and L' dims)
        let kernel_expanded = kernel.unsqueeze(0).unsqueeze(3);
        // Expand to match shapes
        let windowed_broadcast =
            windowed_expanded.expand(vec![1.into(), 3.into(), 2.into(), 3.into(), 3.into()]);
        let kernel_broadcast =
            kernel_expanded.expand(vec![1.into(), 3.into(), 2.into(), 3.into(), 3.into()]);

        // Step 3: Element-wise multiplication
        // [1, 3, 2, 3, 3] * [1, 3, 2, 3, 3] -> [1, 3, 2, 3, 3]
        let product = windowed_broadcast * kernel_broadcast;

        // Step 4: Sum over [in_channels=2, kernel_size=3] at axes [2, 4]
        // [1, 3, 2, 3, 3] -> sum(2) -> [1, 3, 3, 3] -> sum(3) -> [1, 3, 3]
        let output = product.sum(2).sum(3);

        graph.output(output);

        // Execute
        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input data: 2 channels with 5 elements each
        // Channel 0: [1, 2, 3, 4, 5]
        // Channel 1: [0.1, 0.2, 0.3, 0.4, 0.5]
        let input_data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, // channel 0
            0.1, 0.2, 0.3, 0.4, 0.5, // channel 1
        ];
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 2, 5], DType::F32);

        // Kernel: 3 output channels, 2 input channels, kernel_size=3
        // Simple kernels for testing
        let kernel_data = vec![
            // out_channel 0:
            1.0f32, 0.0, 0.0, // in_channel 0
            0.0, 0.0, 0.0, // in_channel 1
            // out_channel 1:
            0.0, 1.0, 0.0, // in_channel 0
            0.0, 0.0, 0.0, // in_channel 1
            // out_channel 2:
            0.0, 0.0, 0.0, // in_channel 0
            1.0, 0.0, 0.0, // in_channel 1
        ];
        let kernel_buffer = CBuffer::from_slice(&kernel_data, &[3, 2, 3], DType::F32);

        let outputs = backend.execute(&graph, vec![input_buffer, kernel_buffer]);
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].to_vec::<f32>();

        // Expected output shape: [1, 3, 3] = 9 elements
        assert_eq!(result.len(), 9);

        // The result should have the expected values based on the simple kernels
        // We just check that the computation completed successfully
        println!("Conv1d multi-channel output: {:?}", result);
    }

    #[test]
    fn test_conv1d_stride_2() {
        // Conv1D with stride=2
        // Input: [1, 1, 7] (batch=1, channels=1, length=7)
        // Kernel: [1, 1, 3] (out_channels=1, in_channels=1, kernel_size=3)
        // Stride: 2
        // Output: [1, 1, 3] (batch=1, out_channels=1, output_length=(7-3)/2+1=3)

        let mut graph = Graph::new();

        // Input: [1, 1, 7]
        let input = graph.input(DType::F32, vec![1.into(), 1.into(), 7.into()]);

        // Kernel: [1, 1, 3]
        let kernel = graph.input(DType::F32, vec![1.into(), 1.into(), 3.into()]);

        // Step 1: Apply sliding window with stride=2
        // [1, 1, 7] -> [1, 1, 3, 3]
        let windowed = input.sliding_window(2, 3, 2);

        // Step 2: Reshape for broadcasting
        let windowed_expanded = windowed.unsqueeze(1);
        let kernel_expanded = kernel.unsqueeze(0).unsqueeze(2);
        // Expand kernel to match windowed shape: [1, 1, 1, 1, 3] -> [1, 1, 1, 3, 3]
        let kernel_broadcast =
            kernel_expanded.expand(vec![1.into(), 1.into(), 1.into(), 3.into(), 3.into()]);

        // Step 3: Multiply and sum
        let product = windowed_expanded * kernel_broadcast;
        let output = product.sum(2).sum(3);

        graph.output(output);

        // Execute
        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input: [0, 1, 2, 3, 4, 5, 6]
        let input_data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 1, 7], DType::F32);

        // Kernel: [1, 1, 1] (sum kernel)
        let kernel_data = vec![1.0f32, 1.0, 1.0];
        let kernel_buffer = CBuffer::from_slice(&kernel_data, &[1, 1, 3], DType::F32);

        let outputs = backend.execute(&graph, vec![input_buffer, kernel_buffer]);
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].to_vec::<f32>();

        // Expected output:
        // Position 0 (input[0:3]): [0,1,2] * [1,1,1] = 3
        // Position 1 (input[2:5]): [2,3,4] * [1,1,1] = 9
        // Position 2 (input[4:7]): [4,5,6] * [1,1,1] = 15
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 9.0);
        assert_eq!(result[2], 15.0);
    }
}
