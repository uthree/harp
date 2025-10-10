#[cfg(feature = "backend-c")]
mod tests {
    use harp::ast::DType;
    use harp::backend::c::CBuffer;
    use harp::backend::Backend;
    use harp::graph::Graph;

    #[test]
    fn test_conv1d_grouped() {
        // Test grouped Conv1D with groups=2
        // Input: [1, 4, 5] (batch=1, in_channels=4, length=5)
        // Kernel: [4, 2, 3] (out_channels=4, in_channels_per_group=2, kernel_size=3)
        // Groups: 2 (so 2 in_channels per group, 2 out_channels per group)

        let mut graph = Graph::new();

        let input = graph.input(DType::F32, vec![1.into(), 4.into(), 5.into()]);
        let kernel = graph.input(DType::F32, vec![4.into(), 2.into(), 3.into()]);

        // Apply grouped Conv1D with groups=2
        let output = input.conv1d_grouped(kernel, 1, 1, 2);

        graph.output(output);

        // Execute
        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input data: 4 channels with 5 elements each
        let input_data = vec![
            // Group 0: channels 0-1
            1.0f32, 2.0, 3.0, 4.0, 5.0, // channel 0
            0.1, 0.2, 0.3, 0.4, 0.5, // channel 1
            // Group 1: channels 2-3
            2.0, 3.0, 4.0, 5.0, 6.0, // channel 2
            0.2, 0.3, 0.4, 0.5, 0.6, // channel 3
        ];
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 4, 5], DType::F32);

        // Kernel: 4 output channels, 2 input channels per group, kernel_size=3
        let kernel_data = vec![
            // Group 0 output channels (0-1):
            // out_channel 0:
            1.0f32, 0.0, 0.0, // in_channel 0 (of group 0)
            0.0, 0.0, 0.0, // in_channel 1 (of group 0)
            // out_channel 1:
            0.0, 1.0, 0.0, // in_channel 0 (of group 0)
            0.0, 0.0, 0.0, // in_channel 1 (of group 0)
            // Group 1 output channels (2-3):
            // out_channel 2:
            1.0, 0.0, 0.0, // in_channel 0 (of group 1 = global channel 2)
            0.0, 0.0, 0.0, // in_channel 1 (of group 1 = global channel 3)
            // out_channel 3:
            0.0, 1.0, 0.0, // in_channel 0 (of group 1 = global channel 2)
            0.0, 0.0, 0.0, // in_channel 1 (of group 1 = global channel 3)
        ];
        let kernel_buffer = CBuffer::from_slice(&kernel_data, &[4, 2, 3], DType::F32);

        let outputs = backend.execute(&graph, vec![input_buffer, kernel_buffer]);
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].to_vec::<f32>();
        // Output shape: [1, 4, 3] = 12 elements
        assert_eq!(result.len(), 12);

        println!("Grouped Conv1D output: {:?}", result);
    }

    #[test]
    fn test_conv1d_depthwise() {
        // Test depthwise Conv1D (groups = in_channels)
        // Input: [1, 3, 5] (batch=1, in_channels=3, length=5)
        // Kernel: [3, 1, 3] (out_channels=3, in_channels_per_group=1, kernel_size=3)
        // Groups: 3 (depthwise: each channel is convolved independently)

        let mut graph = Graph::new();

        let input = graph.input(DType::F32, vec![1.into(), 3.into(), 5.into()]);
        let kernel = graph.input(DType::F32, vec![3.into(), 1.into(), 3.into()]);

        // Apply depthwise Conv1D with groups=3
        let output = input.conv1d_grouped(kernel, 1, 1, 3);

        graph.output(output);

        // Execute
        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        // Input data: 3 channels with 5 elements each
        let input_data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, // channel 0
            2.0, 3.0, 4.0, 5.0, 6.0, // channel 1
            3.0, 4.0, 5.0, 6.0, 7.0, // channel 2
        ];
        let input_buffer = CBuffer::from_slice(&input_data, &[1, 3, 5], DType::F32);

        // Kernel: 3 output channels, 1 input channel per group, kernel_size=3
        // Each channel gets its own kernel
        let kernel_data = vec![
            1.0f32, 1.0, 1.0, // kernel for channel 0
            1.0, 1.0, 1.0, // kernel for channel 1
            1.0, 1.0, 1.0, // kernel for channel 2
        ];
        let kernel_buffer = CBuffer::from_slice(&kernel_data, &[3, 1, 3], DType::F32);

        let outputs = backend.execute(&graph, vec![input_buffer, kernel_buffer]);
        assert_eq!(outputs.len(), 1);

        let result = outputs[0].to_vec::<f32>();
        // Output shape: [1, 3, 3] = 9 elements
        assert_eq!(result.len(), 9);

        // Expected values (sum of 3 consecutive elements):
        // Channel 0: [1+2+3=6, 2+3+4=9, 3+4+5=12]
        // Channel 1: [2+3+4=9, 3+4+5=12, 4+5+6=15]
        // Channel 2: [3+4+5=12, 4+5+6=15, 5+6+7=18]
        assert_eq!(result[0], 6.0); // channel 0, position 0
        assert_eq!(result[1], 9.0); // channel 0, position 1
        assert_eq!(result[2], 12.0); // channel 0, position 2
        assert_eq!(result[3], 9.0); // channel 1, position 0
        assert_eq!(result[4], 12.0); // channel 1, position 1
        assert_eq!(result[5], 15.0); // channel 1, position 2
        assert_eq!(result[6], 12.0); // channel 2, position 0
        assert_eq!(result[7], 15.0); // channel 2, position 1
        assert_eq!(result[8], 18.0); // channel 2, position 2

        println!("Depthwise Conv1D output: {:?}", result);
    }

    #[test]
    fn test_conv1d_groups_1_equals_standard() {
        // Verify that groups=1 gives the same result as standard conv1d
        let mut graph1 = Graph::new();
        let mut graph2 = Graph::new();

        let input1 = graph1.input(DType::F32, vec![1.into(), 2.into(), 5.into()]);
        let kernel1 = graph1.input(DType::F32, vec![3.into(), 2.into(), 3.into()]);
        let output1 = input1.clone().conv1d(kernel1.clone(), 1, 1);
        graph1.output(output1);

        let input2 = graph2.input(DType::F32, vec![1.into(), 2.into(), 5.into()]);
        let kernel2 = graph2.input(DType::F32, vec![3.into(), 2.into(), 3.into()]);
        let output2 = input2.conv1d_grouped(kernel2, 1, 1, 1);
        graph2.output(output2);

        let mut backend = harp::backend::CBackend::new();
        if !backend.is_available() {
            println!("C backend not available, skipping test");
            return;
        }

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let kernel_data = vec![
            1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            0.0,
        ];

        let input_buffer1 = CBuffer::from_slice(&input_data, &[1, 2, 5], DType::F32);
        let kernel_buffer1 = CBuffer::from_slice(&kernel_data, &[3, 2, 3], DType::F32);
        let input_buffer2 = CBuffer::from_slice(&input_data, &[1, 2, 5], DType::F32);
        let kernel_buffer2 = CBuffer::from_slice(&kernel_data, &[3, 2, 3], DType::F32);

        let outputs1 = backend.execute(&graph1, vec![input_buffer1, kernel_buffer1]);
        let outputs2 = backend.execute(&graph2, vec![input_buffer2, kernel_buffer2]);

        let result1 = outputs1[0].to_vec::<f32>();
        let result2 = outputs2[0].to_vec::<f32>();

        assert_eq!(result1.len(), result2.len());
        for (i, (v1, v2)) in result1.iter().zip(result2.iter()).enumerate() {
            assert!(
                (v1 - v2).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                v1,
                v2
            );
        }

        println!("Groups=1 matches standard conv1d: PASS");
    }
}
