#undef NDEBUG

#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <dlfcn.h>
#include <unistd.h>
#include <thread>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/platform/env.h"
#include <cuda_fp16.h>
#include <fstream>

using tensorflow::string;
using tensorflow::int32;


#define BATCH_SIZE 64 // default batch size 
#define INFER_NUM 1000 // default infer iterations for each stream
#define NUM_STREAMS 2 // default total infer iterations num will be NUM_STREAMS * INFER_NUM
#define MAX_NUM_STREAMS 1024
#define MAX_NUM_THREADS 1024
#define cudaEventBlockingSync 0x01

// after the capturing, the H2D nodes corresponding to the input tensors will be removed
// users should do the H2D copy mannually
#define REMOVE_H2D


namespace tensorflow {

//cpu allocator
static Allocator * host_allocator=nullptr;


void CheckCudaError (cudaError_t ERR) {
    if((ERR) != cudaSuccess){ 
        std::cout << "cuda error: " << ERR << " " << cudaGetErrorString(ERR) << std::endl;
    }
}

namespace example {

typedef std::vector<std::pair<std::string, Tensor>> InputsMap;

struct CopyInfo{
    void * src;
    void * dst;
    size_t num_bytes;
};

typedef std::map<std::pair<std::string, int>, std::vector<CopyInfo>> CopyMapping;

void CudaGraphRun(Session * sess, cudaStream_t * streams, int num_infers_per_thread, int num_streams,
                  CopyMapping * copy_mapping, int start_graph_idx = 0) {
    // launch graphs
    for (int i = 0; i < num_infers_per_thread; i++) {
        int stream_idx = i % num_streams;
        cudaEvent_t event;
        CheckCudaError(cudaEventCreateWithFlags(&event, cudaEventBlockingSync));
 #ifdef REMOVE_H2D
        // do h2d copies first
        auto & copy_infos = (*copy_mapping)[std::pair<std::string, int>("TestModel", stream_idx + start_graph_idx)];
        for(int k = 0; k < copy_infos.size(); k ++){
            CheckCudaError(cudaMemcpyAsync(copy_infos[k].dst, copy_infos[k].src, copy_infos[k].num_bytes,
                                           cudaMemcpyHostToDevice, streams[stream_idx]));
        }
#endif      
        // specify model_name, graph_index, and stream
        TF_CHECK_OK(sess->RunCudaGraph("TestModel", stream_idx + start_graph_idx, streams[stream_idx]));
        CheckCudaError(cudaEventRecord(event, streams[stream_idx]));
        CheckCudaError(cudaEventSynchronize(event));
        CheckCudaError(cudaEventDestroy(event));
    }
}

int LaunchGraphs(Session * sess, cudaStream_t * streams, int num_infers_per_thread, int num_streams, int num_threads,
                 CopyMapping & copy_mapping, int start_graph_idx = 0){ 
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i++){
        threads.push_back(std::thread(CudaGraphRun, sess, streams, num_infers_per_thread, 
                                      num_streams, &copy_mapping, start_graph_idx));
    }
    for(auto & thread : threads){
        thread.join();
    }
    return 0;
}

// thread function for multiple thread TF session run
void TFRun(Session * sess, int infer_num, std::vector<std::pair<std::string, Tensor>> *inputs,
           std::vector<std::string> *output_names,  std::vector<Tensor> *output_tensors){
    
    for(int i = 0; i < infer_num; i ++){
        TF_CHECK_OK(sess->Run(*inputs, *output_names, {}, output_tensors));
    }
}


TensorShape getNodeShape(const GraphDef & graph_def, const std::string name, int batch_size){
    for(int i =0; i < graph_def.node_size(); i ++){
        auto n = graph_def.node(i);
        if(n.name() == name){
            auto shape = n.attr().at("shape").shape();
            int dims = shape.dim_size();
            TensorShape tensorShape;
            
            for(int d = 0; d < dims; d ++){
                int dim_size = shape.dim(d).size();
                
                if( d == 0 && dim_size == -1){
                    int new_size = batch_size;
                    
                    // assume the first dimension is batch size, note that it may not be true for some models.
                    LOG(INFO) << "change batch size from: " << dim_size << " to " << new_size << std::endl;
                    dim_size = new_size;
                }
                tensorShape.AddDim(dim_size);
            }
            
            return tensorShape;
        }
    }
    LOG(ERROR) << "Cannot find the node" << name << std::endl;
    exit(1);
}


DataType getNodeType(const GraphDef & graph_def, const std::string name){
    for(int i =0; i < graph_def.node_size(); i ++){
        auto n = graph_def.node(i);
        if(n.name() == name){
            auto dtype = n.attr().at("dtype").type();
            return dtype;
        }
    }
    LOG(ERROR) << "Cannot find the node" << name << std::endl;
    exit(1);
}

void RandomInitialize(Tensor& t)
{
    int num_elements = t.NumElements();
    if(t.dtype() == DT_HALF){
        __half * data = reinterpret_cast<__half*>(t.flat<Eigen::half>().data());
        for(int i = 0; i < num_elements; i ++){
            float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
            data[i] = __float2half(value);
        }
    }else if(t.dtype() == DT_FLOAT){
        float * data = t.flat<float>().data();
        for(int i =0; i < num_elements; i ++){
            float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
            data[i] = value;
        }
    }else if(t.dtype() == DT_INT32){
        int * data = t.flat<int>().data();
        for(int i =0; i < num_elements; i ++){
            int value = static_cast<int>(rand() % 10);
            data[i] = value;
        }
    }else if(t.dtype() == DT_BOOL){
        bool * data = t.flat<bool>().data();
        for(int i =0; i < num_elements; i ++){
            bool value = static_cast<bool>(rand() % 2);
            data[i] = value;
        }
    }else if(t.dtype() == DT_INT64){
        int64 * data = t.flat<int64>().data();
        for(int i =0; i < num_elements; i ++){
            int64 value = static_cast<int64>(rand() % 10);
            data[i] = value;
        }
    }
    else{
        std::cout << t.dtype() << std::endl;
        std::cout << "Random init: unsupported data type." << std::endl;
    }
}

void PrintTensorData(Tensor &t){
    void * data;
    if(t.dtype() == DT_HALF){
        data = static_cast<void*>(t.flat<Eigen::half>().data());
    }else if(t.dtype() == DT_FLOAT){
        data = static_cast<void*>(t.flat<float>().data());
    }else if(t.dtype() == DT_BOOL){
        data = static_cast<void*>(t.flat<bool>().data());
    }else if(t.dtype() == DT_INT32){
        data = static_cast<void*>(t.flat<int>().data());
    }else{
        std::cout << "Print Tensor: Unsupported data type!" << std::endl;
        return;
    }

    int dims = t.dims();
    std::cout << "shape: " << std::endl;
    for(int i = 0; i < dims; i ++){
        std::cout << t.dim_size(i) << ", ";
    }
    std::cout << std::endl;
    
    int size = t.NumElements();
    size = size > 32 ? 32 : size;
    
    for(int i = 0; i < size; i ++){
        float value;
        if(t.dtype() == DT_HALF){
            value = __half2float(static_cast<__half*>(data)[i]);
        }else if(t.dtype() == DT_INT32){
            value = static_cast<int*>(data)[i];
        }else if(t.dtype() == DT_BOOL){
            value = static_cast<bool*>(data)[i];
        }
        else{
            value = static_cast<float*>(data)[i];
        }        
        std::cout << value << ", ";
    }
    std::cout << std::endl;
}

void GenerateInputs(GraphDef & graph_def, 
                    const std::vector<string> &input_names,
                    std::vector<Tensor> & input_tensors, int batch_size){
    
    input_tensors.clear();
    for(int i = 0; i < input_names.size(); i ++){
        auto tensorshape = getNodeShape(graph_def, input_names[i], batch_size);
        auto tensortype = getNodeType(graph_def, input_names[i]);
                
        Tensor t;
        t = Tensor(host_allocator, tensortype, tensorshape);
        RandomInitialize(t);
        
        input_tensors.push_back(t);
    }
}


void FillInputsMap(InputsMap & inputs_map,
                  std::vector<std::string> & input_names,
                  std::vector<Tensor> & input_tensors){
    assert(input_names.size() == input_tensors.size());
    
    for(size_t i = 0; i < input_tensors.size(); i ++){
        inputs_map.push_back(std::pair<std::string, Tensor>(input_names[i], input_tensors[i]));
    }
}


void CopyTensorContents(Tensor &dst_tensor, Tensor &src_tensor){
    //assert(dst_tensor.AllocatedBytes() == src_tensor.AllocatedBytes());
    int dst_eles = dst_tensor.NumElements();
    int src_eles = src_tensor.NumElements();
    assert(dst_eles == src_eles);
    
    assert(dst_tensor.dtype() == src_tensor.dtype());
    
    char *dst, *src;
    int ele_size;
    if(dst_tensor.dtype() == DT_HALF){
        dst = reinterpret_cast<char*>(dst_tensor.flat<Eigen::half>().data());
        src = reinterpret_cast<char*>(src_tensor.flat<Eigen::half>().data());
        ele_size = 2;
    }else if(dst_tensor.dtype() == DT_FLOAT){
        dst = reinterpret_cast<char*>(dst_tensor.flat<float>().data());
        src = reinterpret_cast<char*>(src_tensor.flat<float>().data());
        ele_size = 4;
    }else if(dst_tensor.dtype() == DT_INT32){
        dst = reinterpret_cast<char*>(dst_tensor.flat<int>().data());
        src = reinterpret_cast<char*>(src_tensor.flat<int>().data());
        ele_size = 4;
    }else if(dst_tensor.dtype() == DT_BOOL){
        dst = reinterpret_cast<char*>(dst_tensor.flat<bool>().data());
        src = reinterpret_cast<char*>(src_tensor.flat<bool>().data());
        ele_size = 1;
    }else if(dst_tensor.dtype() == DT_INT64){
        dst = reinterpret_cast<char*>(dst_tensor.flat<int64>().data());
        src = reinterpret_cast<char*>(src_tensor.flat<int64>().data());
        ele_size = 8;
    }else{
        std::cout << "Copy Tensor: Unsupported data type!" << std::endl;
        return;
    }
    
    for(int i = 0; i < src_eles * ele_size; i ++){
        dst[i] = src[i];
    }
}
    
Status Test(GraphDef & graph_def, 
            std::vector<std::string> & input_names,
            std::vector<std::string> & output_names,
            int batch_size,
            int num_infers_per_thread,
            int num_streams,
            int num_threads){
    
    assert(num_streams <= MAX_NUM_STREAMS);
    
    // Creates a session.
    SessionOptions options;
    options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
    options.config.mutable_gpu_options()->set_allow_growth(false);
    // for cudagraph config
    std::unique_ptr<Session> session(NewSession(options));
    
    if (options.target.empty()) {
        graph::SetDefaultDevice("/device:GPU:0", &graph_def);
    }
    graph::CheckNodeDevice("/device:GPU:0", &graph_def);
    TF_CHECK_OK(session->Create(graph_def));

    const DeviceMgr * device_manager;
    TF_CHECK_OK(session->LocalDeviceManager(&device_manager));
    std::vector<Device*> devices=device_manager->ListDevices();
    for (auto * d : devices){
        if(d->name().find("CPU") != std::string::npos){
            std::cout << "CPU device:" << d->name() << std::endl;
            host_allocator = dynamic_cast<ThreadPoolDevice*>(d)->GetAllocator(AllocatorAttributes());
        }
    }
    
    std::vector<Tensor> input_tensors_tf; // input tensors for Normal TF runs
    GenerateInputs(graph_def, input_names, input_tensors_tf, batch_size);
    
    // first normal TF session run 
    InputsMap inputs_tf; // input map for Normal TF run
    FillInputsMap(inputs_tf, input_names, input_tensors_tf);
    
    // TF Multiple threads runs
    // Run session.run in multiple threads
    // The number of threads are same with num_streams
    std::vector<Tensor> output_tensors_tf[MAX_NUM_STREAMS];
    std::vector<std::thread> threads;

        
    auto start  = std::chrono::system_clock::now();
    for(int i = 0; i < num_threads; i ++){
        threads.push_back(std::thread(TFRun, session.get(), num_infers_per_thread,
                                      &inputs_tf, &output_names, &output_tensors_tf[i]));
    }
    for(auto & thread : threads){
        thread.join();
    }
    auto end = std::chrono::system_clock::now();
    
    return Status();
  
    for(int i = 0; i < num_threads; i ++){
        LOG(INFO) << "TF results: ";
        PrintTensorData(output_tensors_tf[i][0]); // print first output tensor
        output_tensors_tf[i].clear();
    }
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double duration_seconds = duration.count() * 1.0 / 1000;
    LOG(INFO) << "[TF + Multiple Threads] Duration = " << duration_seconds << " seconds." << std::endl;
    double qps = num_infers_per_thread * num_threads * 1.0 / duration_seconds;
    LOG(INFO) << "[TF + Multiple Threads] Reference Average QPS = " << qps << std::endl;
    
    // capture the cuda graph
    assert(session->SupportsCudaGraph());
    cudaStream_t streams[MAX_NUM_STREAMS];
    
    // create launching streams
    for (int i=0; i < num_streams; i ++){
        CheckCudaError(cudaStreamCreate(&streams[i]));
    }
    
    cudaStream_t stream = session->EnableGraphCapture("TestModel");
    LOG(INFO) << "capturing on stream -- " << stream;
    if (stream == NULL){
        return Status(error::Code::INTERNAL, "Get stream for graph capturing failed.");
    }
    
    // For multiple-stream runs, 
    // We need to capture multiple independent cuda graphs
    // with seperated inputs/output buffers, and buffers for intermedidate layers
    InputsMap inputs_cuda_graph[MAX_NUM_STREAMS];
    std::vector<Tensor> input_tensors_cuda_graph[MAX_NUM_STREAMS];
    std::vector<Tensor> output_tensors_cuda_graph[MAX_NUM_STREAMS];

    // prepare inputs
    for(int i = 0; i < num_streams; i ++){
        GenerateInputs(graph_def, input_names, input_tensors_cuda_graph[i], batch_size);
        
        // copy from the input tensors for normal TF runs,
        // so we can compare the results
        for(int ii = 0; ii < input_names.size(); ii ++){
            CopyTensorContents(input_tensors_cuda_graph[i][ii], input_tensors_tf[ii]);
        }
        
        FillInputsMap(inputs_cuda_graph[i], input_names, input_tensors_cuda_graph[i]);
    }
    
    // capture multiple graphs
    for(int i = 0; i < num_streams; i ++){
        TF_CHECK_OK(session->Run(inputs_cuda_graph[i], output_names, {}, &output_tensors_cuda_graph[i]));
    }
    
    // turn off graph capture mode
    session->DisableGraphCapture();
    
  //  LogCudaGraphStatus(session.get());
    
    LOG(INFO) << "Run the cuda graphs in multiple streams";
    
    // get the mappings first
    CopyMapping copy_mapping;

#ifdef REMOVE_H2D
    for(int i = 0; i < num_streams; i ++){
        std::vector<std::pair<void*, void*>> mappings = session->GetSrcDstMapping("TestModel", i);

        std::cout << "all mappings: " << std::endl;
        for(auto & p : mappings){
            std::cout << p.first << " -> " << p.second << std::endl;
        }
        
        std::vector<Tensor> & in_tensors = input_tensors_cuda_graph[i];
        std::vector<CopyInfo> copy_infors;
        
        for(auto& t: in_tensors){
            void* host_buffer;
            size_t ele_size = 1;
            if(t.dtype() == DT_HALF){
                ele_size = 2;
                host_buffer = reinterpret_cast<void*>(t.flat<Eigen::half>().data());
            }else if(t.dtype() == DT_FLOAT){
                ele_size = 4;
                host_buffer = reinterpret_cast<void*>(t.flat<float>().data());
            }else if(t.dtype() == DT_INT32){
                ele_size = 4;
                host_buffer = reinterpret_cast<void*>(t.flat<int>().data());
            }else if(t.dtype() == DT_BOOL){
                ele_size = 1;
                host_buffer = reinterpret_cast<void*>(t.flat<bool>().data());
            }else if(t.dtype() == DT_INT64){
                ele_size = 8;
                host_buffer = reinterpret_cast<void*>(t.flat<int64>().data());
            }else{
                std::cout << "Unsupported data type!" << std::endl;
                exit(1);
            }
            
            size_t num_elements = t.NumElements();
            std::cout << "num elements: " << t.NumElements() << std::endl;
            std::cout << "host buffer: " << host_buffer << std::endl;
            
            size_t num_bytes = num_elements * ele_size;
            void* device_buffer = NULL;
            
            for(auto& p: mappings){
                if(p.first == host_buffer){
                    device_buffer = p.second;
                    break;
                }
            }
            
            if(device_buffer == NULL){
                // some input tensors may not have corresponding H2D nodes
                // (some op kernel has host_memory constraints for its inputs, though the op's on GPU)
                // if this input tensor is const (given specific input shapes), then we allow it hasn't H2D node, 
                // as it's values are captured into the launch parameters into the CUDA Graph nodes.
                std::cout << "Failed to find src to dst mapping." << std::endl;                
            }
            
            if(device_buffer != NULL){
                copy_infors.push_back(CopyInfo{host_buffer, device_buffer, num_bytes});
            }
        }
        copy_mapping[std::pair<std::string, int>("TestModel", i)] = copy_infors;
    }
#endif
    
    start  = std::chrono::system_clock::now();
    std::cout << "start launching..." << std::endl;
    
    // launch multiple graphs in indepent streams
    LaunchGraphs(session.get(), streams, num_infers_per_thread, num_streams, num_threads, copy_mapping);
                
    end = std::chrono::system_clock::now();
    
    LOG(INFO) << "Cuda Graph results:";
    for(int i = 0; i < num_streams; i ++){
         PrintTensorData(output_tensors_cuda_graph[i][0]);
    }
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    duration_seconds = duration.count() * 1.0 / 1000;
    LOG(INFO) << "[Cuda Graph + Multiple Streams] Duration = " << duration_seconds << " seconds." << std::endl;
    qps = num_infers_per_thread * num_threads * 1.0 / duration_seconds;
    LOG(INFO) << "[Cuda Graph + Multiple Streams] Average QPS = " << qps << std::endl;
    
    // Test the normal Cuda Graph runs again
    threads.clear();
    start  = std::chrono::system_clock::now();
    for(int i = 0; i < num_threads; i ++){
        threads.push_back(std::thread(TFRun, session.get(), num_infers_per_thread,
                                      &inputs_tf, &output_names, &output_tensors_tf[i]));
    }
    
    for(auto & thread : threads){
        thread.join();
    }
    
    end = std::chrono::system_clock::now();
    
    // print TF outputs again, to verify the correctness of diabling cuda graph
    for(int i = 0; i < num_threads; i ++){
        LOG(INFO) << "TF results: ";
        PrintTensorData(output_tensors_tf[i][0]);
        output_tensors_tf[i].clear();
    }
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    duration_seconds = duration.count() * 1.0 / 1000;
    LOG(INFO) << "[TF + Multiple Threads] Duration = " << duration_seconds << " seconds." << std::endl;
    qps = num_infers_per_thread * num_threads * 1.0 / duration_seconds;
    LOG(INFO) << "[TF + Multiple Threads] Reference Average QPS = " << qps << std::endl;
    
    TF_CHECK_OK(session->DestroyCudaGraphs());
    TF_CHECK_OK(session->Close());
    return Status();
}
        
}  // end namespace example

}  // end namespace tensorflow

using namespace tensorflow;


int main(int argc, char* argv[]) {
    // Example: ./application model_path in_num input_name [input_names] out_num output_name [output_names] 
    //                        batch_size infer_num_per_stream num_streams [custom_op_lib_path]
    // read command line arguments
    int arg_idx = 1;
    std::string model_path = argv[arg_idx++];
    int input_num = std::stoi(argv[arg_idx++]);
    assert(input_num >= 1);
    
    std::vector<std::string> input_names;
    std::cout << input_num <<  " inputs: "; 
    for(int i = 0; i < input_num; i ++){
        input_names.push_back(argv[arg_idx++]);
        std::cout << argv[arg_idx - 1] << ",";
    }
    std::cout << std::endl;
    
    int output_num = std::stoi(argv[arg_idx++]);
    assert(output_num >= 1);
    std::vector<std::string> output_names;
    std::cout << output_num << " outputs: ";
    for(int i = 0; i < output_num; i ++){
        output_names.push_back(argv[arg_idx++]);
        std::cout << argv[arg_idx - 1] << ",";
    }
    std::cout << std::endl;

    //std::cout << "mode = " << mode << std::endl;
    
    int batch_size = BATCH_SIZE;
    if(argc > arg_idx){
        batch_size = std::stoi(argv[arg_idx++]);
        assert(batch_size >= 1);
    }
    std::cout << "batch size = " << batch_size << std::endl;

    int num_infers_per_thread = INFER_NUM;
    if(argc > arg_idx){
        num_infers_per_thread = std::stoi(argv[arg_idx++]);
        assert(num_infers_per_thread >= 1);
    }
    std::cout << "num_infers_per_thread = " << num_infers_per_thread << std::endl;

    int num_streams = NUM_STREAMS;
    if(argc > arg_idx){
        num_streams = std::stoi(argv[arg_idx++]);
        assert(num_streams >= 1);
        assert(num_streams <= MAX_NUM_STREAMS);
    }
    std::cout << "num_streams = " << num_streams << std::endl;

    // default threads count is equal to stream, one stream per thread.
    int num_threads = num_streams;
    if (argc > arg_idx) {
        num_threads = std::stoi(argv[arg_idx++]);
        assert(num_threads >= 1);
        assert(num_threads <= MAX_NUM_THREADS);
    }
    std::cout << "num_threads = " << num_threads << std::endl;
    
    if(argc > arg_idx){
        const char * custom_op_lib = argv[arg_idx++];
        dlopen(custom_op_lib, RTLD_LAZY);
        std::cout << "with custom op lib: " << custom_op_lib << std::endl;
    }
    // command line arguments reading done.
    
    GraphDef graph_def;
    Status status;

    if(model_path.find(".pbtxt") == std::string::npos){
        status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
    }else{
        status = ReadTextProto(Env::Default(), model_path, &graph_def);
    }
    
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }
    
    example::Test(graph_def, input_names, output_names,
                  batch_size, num_infers_per_thread, num_streams, num_threads);
    return 0;
}
