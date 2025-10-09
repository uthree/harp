#[cfg(feature = "backend-c")]
mod tests {
    use harp::ast::DType;
    use harp::backend::c::CBuffer;
    use harp::backend::Backend;
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

    #[test]
    fn test_conv2d_via_sliding_window() {
        // Simple 2D convolution using sliding_window
        // Input: [1, 1, 5, 5] (batch=1, channels=1, height=5, width=5)
        // Kernel: [1, 1, 3, 3] (out_channels=1, in_channels=1, kernel_h=3, kernel_w=3)
        // Output: [1, 1, 3, 3] (batch=1, out_channels=1, out_h=3, out_w=3)

        let mut graph = Graph::new();

        // Input: [1, 1, 5, 5]
        let input = graph.input(DType::F32, vec![1.into(), 1.into(), 5.into(), 5.into()]);

        // Kernel: [1, 1, 3, 3]
        let kernel = graph.input(DType::F32, vec![1.into(), 1.into(), 3.into(), 3.into()]);

        // Step 1: Apply sliding window along height (dim=2)
        // [1, 1, 5, 5] -> [1, 1, 3, 5, 3] where out_h = (5-3)/1+1 = 3
        let windowed_h = input.sliding_window(2, 3, 1);

        // Step 2: Apply sliding window along width (dim=3, shifted because Kh was added)
        // [1, 1, 3, 5, 3] -> [1, 1, 3, 3, 3, 3] where out_w = (5-3)/1+1 = 3
        let windowed_hw = windowed_h.sliding_window(3, 3, 1);

        // Step 3: Reshape for broadcasting
        // windowed: [1, 1, 3, 3, 3, 3] -> [1, 1, 1, 3, 3, 3, 3] (add out_channel dim)
        let windowed_expanded = windowed_hw.unsqueeze(1);
        // kernel: [1, 1, 3, 3] -> [1, 1, 1, 1, 1, 3, 3] (add batch, H', W' dims)
        let kernel_expanded = kernel.unsqueeze(0).unsqueeze(2).unsqueeze(3);
        // Expand to match shapes
        let kernel_broadcast = kernel_expanded.expand(vec![
            1.into(),
            1.into(),
            1.into(),
            3.into(),
            3.into(),
            3.into(),
            3.into(),
        ]);

        // Step 4: Element-wise multiplication
        // [1, 1, 1, 3, 3, 3, 3] * [1, 1, 1, 3, 3, 3, 3] -> [1, 1, 1, 3, 3, 3, 3]
        let product = windowed_expanded * kernel_broadcast;

        // Step 5: Sum over [in_channels, Kh, Kw] at axes [2, 5, 6]
        // [1, 1, 1, 3, 3, 3, 3] -> sum(2) -> [1, 1, 3, 3, 3, 3] -> sum(4) -> [1, 1, 3, 3, 3] -> sum(4) -> [1, 1, 3, 3]
        let output = product.sum(2).sum(4).sum(4);

        graph.output(output);

        // Execute
        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input data: 5x5 grid with values 0-24
        let input_data: Vec<f32> = (0..25).map(|x| x as f32).collect();
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 1, 5, 5], DType::F32);

        // Kernel: 3x3 sum kernel (all ones)
        let kernel_data = vec![1.0f32; 9];
        let kernel_buffer = CBuffer::from_slice(&kernel_data, &[1, 1, 3, 3], DType::F32);

        let outputs = backend.execute(&graph, vec![input_buffer, kernel_buffer]);
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].to_vec::<f32>();

        // Expected output: 3x3 grid where each element is the sum of a 3x3 window
        // For position (0,0): sum of elements [0,1,2, 5,6,7, 10,11,12] = 54
        // For position (0,1): sum of elements [1,2,3, 6,7,8, 11,12,13] = 63
        // etc.
        assert_eq!(result.len(), 9);
        assert_eq!(result[0], 54.0); // top-left
        assert_eq!(result[1], 63.0); // top-center
        assert_eq!(result[2], 72.0); // top-right
    }

    #[test]
    fn test_conv2d_stride() {
        // Conv2D with stride=2
        // Input: [1, 1, 7, 7]
        // Kernel: [1, 1, 3, 3]
        // Stride: (2, 2)
        // Output: [1, 1, 3, 3] where out_h = out_w = (7-3)/2+1 = 3

        let mut graph = Graph::new();

        let input = graph.input(DType::F32, vec![1.into(), 1.into(), 7.into(), 7.into()]);
        let kernel = graph.input(DType::F32, vec![1.into(), 1.into(), 3.into(), 3.into()]);

        // Apply sliding window with stride=2 in both dimensions
        let windowed_h = input.sliding_window(2, 3, 2);
        let windowed_hw = windowed_h.sliding_window(3, 3, 2);

        let windowed_expanded = windowed_hw.unsqueeze(1);
        let kernel_expanded = kernel.unsqueeze(0).unsqueeze(2).unsqueeze(3);
        let kernel_broadcast = kernel_expanded.expand(vec![
            1.into(),
            1.into(),
            1.into(),
            3.into(),
            3.into(),
            3.into(),
            3.into(),
        ]);

        let product = windowed_expanded * kernel_broadcast;
        let output = product.sum(2).sum(4).sum(4);

        graph.output(output);

        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input: 7x7 grid
        let input_data: Vec<f32> = (0..49).map(|x| x as f32).collect();
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 1, 7, 7], DType::F32);

        // Kernel: 3x3 average (divide by 9 manually in expected values)
        let kernel_data = vec![1.0f32; 9];
        let kernel_buffer = CBuffer::from_slice(&kernel_data, &[1, 1, 3, 3], DType::F32);

        let outputs = backend.execute(&graph, vec![input_buffer, kernel_buffer]);
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].to_vec::<f32>();
        assert_eq!(result.len(), 9); // 3x3 output

        // Just verify the computation completed successfully
        println!("Conv2d stride=2 output: {:?}", result);
    }

    #[test]
    fn test_conv3d_via_sliding_window() {
        // Simple 3D convolution using sliding_window
        // Input: [1, 1, 4, 4, 4] (batch=1, channels=1, depth=4, height=4, width=4)
        // Kernel: [1, 1, 2, 2, 2] (out_channels=1, in_channels=1, kd=2, kh=2, kw=2)
        // Output: [1, 1, 3, 3, 3] (batch=1, out_channels=1, out_d=3, out_h=3, out_w=3)

        let mut graph = Graph::new();

        // Input: [1, 1, 4, 4, 4]
        let input = graph.input(
            DType::F32,
            vec![1.into(), 1.into(), 4.into(), 4.into(), 4.into()],
        );

        // Kernel: [1, 1, 2, 2, 2]
        let kernel = graph.input(
            DType::F32,
            vec![1.into(), 1.into(), 2.into(), 2.into(), 2.into()],
        );

        // Step 1: Apply sliding window along depth (dim=2)
        // [1, 1, 4, 4, 4] -> [1, 1, 3, 4, 4, 2] where out_d = (4-2)/1+1 = 3
        let windowed_d = input.sliding_window(2, 2, 1);

        // Step 2: Apply sliding window along height (dim=3, shifted because Kd was added)
        // [1, 1, 3, 4, 4, 2] -> [1, 1, 3, 3, 4, 2, 2] where out_h = (4-2)/1+1 = 3
        let windowed_dh = windowed_d.sliding_window(3, 2, 1);

        // Step 3: Apply sliding window along width (dim=4, shifted because Kd and Kh were added)
        // [1, 1, 3, 3, 4, 2, 2] -> [1, 1, 3, 3, 3, 2, 2, 2] where out_w = (4-2)/1+1 = 3
        let windowed_dhw = windowed_dh.sliding_window(4, 2, 1);

        // Step 4: Reshape for broadcasting
        // windowed: [1, 1, 3, 3, 3, 2, 2, 2] -> [1, 1, 1, 3, 3, 3, 2, 2, 2] (add out_channel dim)
        let windowed_expanded = windowed_dhw.unsqueeze(1);
        // kernel: [1, 1, 2, 2, 2] -> [1, 1, 1, 1, 1, 1, 2, 2, 2] (add batch, D', H', W' dims)
        let kernel_expanded = kernel.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4);
        // Expand to match shapes
        let kernel_broadcast = kernel_expanded.expand(vec![
            1.into(),
            1.into(),
            1.into(),
            3.into(),
            3.into(),
            3.into(),
            2.into(),
            2.into(),
            2.into(),
        ]);

        // Step 5: Element-wise multiplication
        let product = windowed_expanded * kernel_broadcast;

        // Step 6: Sum over [in_channels, Kd, Kh, Kw] at axes [2, 6, 7, 8]
        // [1, 1, 1, 3, 3, 3, 2, 2, 2] -> sum(2) -> [1, 1, 3, 3, 3, 2, 2, 2]
        // -> sum(5) -> [1, 1, 3, 3, 3, 2, 2] -> sum(5) -> [1, 1, 3, 3, 3, 2] -> sum(5) -> [1, 1, 3, 3, 3]
        let output = product.sum(2).sum(5).sum(5).sum(5);

        graph.output(output);

        // Execute
        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input data: 4x4x4 grid with values 0-63
        let input_data: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 1, 4, 4, 4], DType::F32);

        // Kernel: 2x2x2 sum kernel (all ones)
        let kernel_data = vec![1.0f32; 8];
        let kernel_buffer = CBuffer::from_slice(&kernel_data, &[1, 1, 2, 2, 2], DType::F32);

        let outputs = backend.execute(&graph, vec![input_buffer, kernel_buffer]);
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].to_vec::<f32>();

        // Expected output: 3x3x3 grid where each element is the sum of a 2x2x2 cube
        // For position (0,0,0): sum of 8 elements [0,1,4,5,16,17,20,21] = 84
        assert_eq!(result.len(), 27); // 3x3x3 output
        assert_eq!(result[0], 84.0); // corner cube

        println!("Conv3d output (first few): {:?}", &result[..3]);
    }

    #[test]
    fn test_conv3d_stride() {
        // Conv3D with stride=2
        // Input: [1, 1, 5, 5, 5]
        // Kernel: [1, 1, 2, 2, 2]
        // Stride: (2, 2, 2)
        // Output: [1, 1, 2, 2, 2] where out_d = out_h = out_w = (5-2)/2+1 = 2

        let mut graph = Graph::new();

        let input = graph.input(
            DType::F32,
            vec![1.into(), 1.into(), 5.into(), 5.into(), 5.into()],
        );
        let kernel = graph.input(
            DType::F32,
            vec![1.into(), 1.into(), 2.into(), 2.into(), 2.into()],
        );

        // Apply sliding window with stride=2 in all three dimensions
        let windowed_d = input.sliding_window(2, 2, 2);
        let windowed_dh = windowed_d.sliding_window(3, 2, 2);
        let windowed_dhw = windowed_dh.sliding_window(4, 2, 2);

        let windowed_expanded = windowed_dhw.unsqueeze(1);
        let kernel_expanded = kernel.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4);
        let kernel_broadcast = kernel_expanded.expand(vec![
            1.into(),
            1.into(),
            1.into(),
            2.into(),
            2.into(),
            2.into(),
            2.into(),
            2.into(),
            2.into(),
        ]);

        let product = windowed_expanded * kernel_broadcast;
        let output = product.sum(2).sum(5).sum(5).sum(5);

        graph.output(output);

        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input: 5x5x5 grid
        let input_data: Vec<f32> = (0..125).map(|x| x as f32).collect();
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 1, 5, 5, 5], DType::F32);

        // Kernel: 2x2x2 sum kernel
        let kernel_data = vec![1.0f32; 8];
        let kernel_buffer = CBuffer::from_slice(&kernel_data, &[1, 1, 2, 2, 2], DType::F32);

        let outputs = backend.execute(&graph, vec![input_buffer, kernel_buffer]);
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].to_vec::<f32>();
        assert_eq!(result.len(), 8); // 2x2x2 output

        println!("Conv3d stride=2 output: {:?}", result);
    }
}
