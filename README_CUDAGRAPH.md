# CUDA Graph Sample

## Compile

`bazel build --config=cuda --config opt //tensorflow/cc:tutorials_example_cudagraph`

The cuda graph sample source code: tensorflow/cc/tutorials/cuda_graph_inference.cc

## Run

`./tutorials_example_cudagraph <model.pb> <num_inputs> <input_1_name> <input_2_name> ...  <num_outputs> <output_1_name> <output_2_name> ...  <precision> <batch_size> <test_iterations> <num_streams> <optional custom op lib>`

*  \<model.pb\>:  the freezed model file
*  \<num_inputs\>:  the number of inputs
*  \<input_N_name\>:  the names of inputs
*  \<num_outputs\>:  the number of outputs
*  \<output_N_name\>: the names of outputs
*  \<precision\>:  fp16 or fp32 (depends on the saved model's precision)
*  \<batch_size\>: testing batch size
*  \<test_iterations\>: the test iterations for each stream/thread, the total iterations will be \<test_iterations\> * \<num_streams\>
*  \<num_streams\>: the number of streams used for executing cuda graphs, in the sample, \<num_streams\> graphs will be captured for a model
*  \<optional custom op lib\>: if the model uses custom op, also need to specify the custom op lib path

## Constraints
*  Only support local environment (DirectSession)
*  Only support models with all the operations executing on GPU
*  Only support single GPU (use CUDA_VISIBLE_DEVICES to control the GPU visibility)
*  No online CUDA graph updating

## Capture & Launch Procedure
1.  Create a DirectSession object (with allow_growth option set to false)
2.  Execute the normal session run (to make sure TF completes all the initializations)
3.  Turn on Graph Capture mode
    ```
    assert(session->SupportsCudaGraph());
    cudaStream_t stream = session->EnableGraphCapture("TestModel");
    ```
4.  Prepare inputs & outputs for graph capture (to run multiple graphs in parallel, we need to create multiple independent graphs)
    ```
    std::vector<Tensor> input_tensors_cuda_graph[MAX_NUM_STREAMS];
    std::vector<Tensor> output_tensors_cuda_graph[MAX_NUM_STREAMS];
    ```
5.  Run mutiple times of sessin run (with corresponding inputs & outputs)
    ```
    // capture multiple graphs
    for(int i = 0; i < num_streams; i ++){
        TF_CHECK_OK(session->Run(inputs_cuda_graph[i], output_names, {}, &output_tensors_cuda_graph[i]));
    }
    ```
6.  Turn off CUDA Graph Capture mode (the session run will behave normally)
    ```
    // turn off graph capture mode
    session->DisableGraphCapture();
    ```
7.  Launch CUDA Graphs into multiple streams
    ```
    // specify model_name, graph_index, and stream
    sess->RunCudaGraph("TestModel", graph_idx, stream);
    ```
    Each Graph are binded to the input_tensors & output_tensors we specified during capturing,
    Inputs are passed by direclty modifying the contents of input_tensors (get the tensor's underlying data ptr, and copy values)
    Outputs are feteched by directly accessing the contents of the output_tensors.

8.  Release the CUDA Graph resources
    ```
    session->DestroyCudaGraphs();
    ```
    The resources include: CUDA Graphs, CUDA Graph Execute Instances, Tensors (input, output, intermedidate tensors) for CUDA Graph launches.
    Refer to the function -- **LogCudaGraphStatus** for how to query the status
    
