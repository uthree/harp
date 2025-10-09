#[cfg(feature = "backend-c")]
mod tests {
    use harp::ast::DType;
    use harp::backend::c::CBuffer;
    use harp::backend::Backend;
    use harp::graph::Graph;

    #[test]
    fn test_conv1d_hlops() {
        // Test Conv1D using the high-level API
        let mut graph = Graph::new();

        // Input: [1, 2, 5] (batch=1, in_channels=2, length=5)
        let input = graph.input(DType::F32, vec![1.into(), 2.into(), 5.into()]);

        // Kernel: [3, 2, 3] (out_channels=3, in_channels=2, kernel_size=3)
        let kernel = graph.input(DType::F32, vec![3.into(), 2.into(), 3.into()]);

        // Apply Conv1D with stride=1
        let output = input.conv1d(kernel, 1);

        graph.output(output);

        // Execute
        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input data: 2 channels with 5 elements each
        let input_data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, // channel 0
            0.1, 0.2, 0.3, 0.4, 0.5, // channel 1
        ];
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 2, 5], DType::F32);

        // Kernel: simple test kernels
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
        // Output shape: [1, 3, 3] = 9 elements
        assert_eq!(result.len(), 9);

        println!("Conv1D hlops output: {:?}", result);
    }

    #[test]
    fn test_conv2d_hlops() {
        // Test Conv2D using the high-level API
        let mut graph = Graph::new();

        // Input: [1, 1, 5, 5]
        let input = graph.input(DType::F32, vec![1.into(), 1.into(), 5.into(), 5.into()]);

        // Kernel: [1, 1, 3, 3]
        let kernel = graph.input(DType::F32, vec![1.into(), 1.into(), 3.into(), 3.into()]);

        // Apply Conv2D with stride=1
        let output = input.conv2d(kernel, 1);

        graph.output(output);

        // Execute
        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input: 5x5 grid with values 0-24
        let input_data: Vec<f32> = (0..25).map(|x| x as f32).collect();
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 1, 5, 5], DType::F32);

        // Kernel: 3x3 sum kernel (all ones)
        let kernel_data = vec![1.0f32; 9];
        let kernel_buffer = CBuffer::from_slice(&kernel_data, &[1, 1, 3, 3], DType::F32);

        let outputs = backend.execute(&graph, vec![input_buffer, kernel_buffer]);
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].to_vec::<f32>();
        assert_eq!(result.len(), 9); // 3x3 output

        // Verify expected values
        assert_eq!(result[0], 54.0); // top-left
        assert_eq!(result[1], 63.0); // top-center
        assert_eq!(result[2], 72.0); // top-right

        println!("Conv2D hlops output: {:?}", result);
    }

    #[test]
    fn test_conv2d_stride_hlops() {
        // Test Conv2D with stride=2 using high-level API
        let mut graph = Graph::new();

        let input = graph.input(DType::F32, vec![1.into(), 1.into(), 7.into(), 7.into()]);
        let kernel = graph.input(DType::F32, vec![1.into(), 1.into(), 3.into(), 3.into()]);

        // Apply Conv2D with stride=2
        let output = input.conv2d(kernel, 2);

        graph.output(output);

        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input: 7x7 grid
        let input_data: Vec<f32> = (0..49).map(|x| x as f32).collect();
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 1, 7, 7], DType::F32);

        // Kernel: 3x3 ones
        let kernel_data = vec![1.0f32; 9];
        let kernel_buffer = CBuffer::from_slice(&kernel_data, &[1, 1, 3, 3], DType::F32);

        let outputs = backend.execute(&graph, vec![input_buffer, kernel_buffer]);
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].to_vec::<f32>();
        assert_eq!(result.len(), 9); // 3x3 output

        println!("Conv2D stride=2 hlops output: {:?}", result);
    }

    #[test]
    fn test_conv3d_hlops() {
        // Test Conv3D using the high-level API
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

        // Apply Conv3D with stride=1
        let output = input.conv3d(kernel, 1);

        graph.output(output);

        // Execute
        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input: 4x4x4 grid with values 0-63
        let input_data: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 1, 4, 4, 4], DType::F32);

        // Kernel: 2x2x2 sum kernel
        let kernel_data = vec![1.0f32; 8];
        let kernel_buffer = CBuffer::from_slice(&kernel_data, &[1, 1, 2, 2, 2], DType::F32);

        let outputs = backend.execute(&graph, vec![input_buffer, kernel_buffer]);
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].to_vec::<f32>();
        assert_eq!(result.len(), 27); // 3x3x3 output
        assert_eq!(result[0], 84.0); // corner cube

        println!("Conv3D hlops output (first few): {:?}", &result[..3]);
    }

    #[test]
    fn test_conv3d_stride_hlops() {
        // Test Conv3D with stride=2 using high-level API
        let mut graph = Graph::new();

        let input = graph.input(
            DType::F32,
            vec![1.into(), 1.into(), 5.into(), 5.into(), 5.into()],
        );
        let kernel = graph.input(
            DType::F32,
            vec![1.into(), 1.into(), 2.into(), 2.into(), 2.into()],
        );

        // Apply Conv3D with stride=2
        let output = input.conv3d(kernel, 2);

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

        println!("Conv3D stride=2 hlops output: {:?}", result);
    }
}
