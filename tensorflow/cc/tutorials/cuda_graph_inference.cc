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
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/common_runtime/cuda_graph_mgr.h"
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
#include "tensorflow/core/util/dump_graph.h"
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
    /*
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
    */
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
  //          float value = 0.1;
            data[i] = __float2half(value);
        }
    }else if(t.dtype() == DT_FLOAT){
        float * data = t.flat<float>().data();
        for(int i =0; i < num_elements; i ++){
            float value = static_cast<float>(rand() % 101 - 50) / 100.0f;
  //          float value = 0.1;
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

void PrepareSessionOption(SessionOptions& options, bool cg_enable = false) {
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(true);
  if (cg_enable) {
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_cuda_graph_enable(true);
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_cuda_graph_capture(true);
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->add_output_names_with_cg("output");
    SubgraphDescription* subgraph = options.config.mutable_graph_options()
                                        ->mutable_optimizer_options()
                                        ->add_subgraph_descriptions();
    subgraph->add_cuda_graph_batch_sizes(64);
    subgraph->set_subgraph_name("test");
    subgraph->add_output_node_names("MatMul_3");
    SubgraphInputTensor* input = subgraph->add_input_tensors();
    input->set_tensor_provider_name("MatMul_1");
    input->set_tensor_provider_slot(0);
    input->set_ph_name("ph");
    input->set_type(DataType::DT_FLOAT);
    input->add_shape(-1);
    input->add_shape(512);
  }
}

void PrepareSessionOptionForGamma(SessionOptions& options, bool cg_enable = false) {
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(true);
  if (cg_enable) {
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_cuda_graph_enable(true);
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_cuda_graph_capture(true);
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->add_output_names_with_cg("p4p_output");
    SubgraphDescription* subgraph = options.config.mutable_graph_options()
                                        ->mutable_optimizer_options()
                                        ->add_subgraph_descriptions();
    subgraph->add_cuda_graph_batch_sizes(4);                         
    subgraph->set_subgraph_name("main");
    subgraph->add_output_node_names("p4p_Main_Score_Network/hiddenlayer_4/hiddenlayer_4/LeakyRelu");
    SubgraphInputTensor* input= subgraph->add_input_tensors();
    input->set_tensor_provider_name("p4p_Main_Score_Network/concat");
    input->set_tensor_provider_slot(0);
    input->set_ph_name("ph");
    input->set_type(DataType::DT_FLOAT);
    input->add_shape(-1);
    input->add_shape(4440);

    SubgraphInputTensor* input1 = subgraph->add_input_tensors();
    input1->set_tensor_provider_name("p4p_Main_Score_Network/column_extend/concat");
    input1->set_tensor_provider_slot(0);
    input1->set_ph_name("ph1");
    input1->set_type(DataType::DT_FLOAT);
    input1->add_shape(-1);
    input1->add_shape(948);

  }
}
void PrepareSessionOptionForDarvin(SessionOptions& options, bool cg_enable = false) {
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(true);
  if (cg_enable) {
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_cuda_graph_enable(cg_enable);
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_cuda_graph_capture(true);
/*
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->add_output_names_with_cg("matchdoc");
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->add_output_names_with_cg("notFoundPk");
*/
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->add_output_names_with_cg("p4p_Main_Score_Network/hiddenlayer_4/hiddenlayer_4/LeakyRelu/output_0");

    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_subgraph_group_name("darvin");
    SubgraphDescription* subgraph = options.config.mutable_graph_options()
                                        ->mutable_optimizer_options()
                                        ->add_subgraph_descriptions();
    subgraph->add_cuda_graph_batch_sizes(2);
    subgraph->set_subgraph_name("main_darwin");
    subgraph->add_output_node_names("p4p_Main_Score_Network/hiddenlayer_4/hiddenlayer_4/LeakyRelu");   

    SubgraphInputTensor* input= subgraph->add_input_tensors();
    input->set_tensor_provider_name("p4p_Main_Score_Network/hiddenlayer_1/hiddenlayer_1/LeakyRelu");
    // input->set_tensor_provider_name("ph");
    input->set_tensor_provider_slot(0);
    input->set_ph_name("phi1");
    input->set_type(DataType::DT_FLOAT);
    input->add_shape(-1);
    input->add_shape(1336);
/*
    SubgraphInputTensor* input1 = subgraph->add_input_tensors();
    input1->set_tensor_provider_name("p4p_Main_Score_Network/hiddenlayer_0/column_extend/concat_2");
//    input1->set_tensor_provider_name("ph1");
    input1->set_tensor_provider_slot(0);
    input1->set_ph_name("ph1");
    input1->set_type(DataType::DT_FLOAT);
    input1->add_shape(-1);
    input1->add_shape(948);

    SubgraphInputTensor* input2 = subgraph->add_input_tensors();
    input2->set_tensor_provider_name("p4p_Main_Score_Network/hiddenlayer_0/column_extend_darwin/concat_1");
//    input2->set_tensor_provider_name("ph2");
    input2->set_tensor_provider_slot(0);
    input2->set_ph_name("ph2");
    input2->set_type(DataType::DT_FLOAT);
    input2->add_shape(1);
    input2->add_shape(512);

    SubgraphInputTensor* input3 = subgraph->add_input_tensors();
    input3->set_tensor_provider_name("p4p_Main_Score_Network/hiddenlayer_0/Sum");
//    input3->set_tensor_provider_name("ph3");
    input3->set_tensor_provider_slot(0);
    input3->set_ph_name("ph3");
    input3->set_type(DataType::DT_FLOAT);
    input3->add_shape(-1);
    input3->add_shape(1648);
*/
  }
}

Status Test(GraphDef & graph_def, 
            std::vector<std::string> & input_names,
            std::vector<std::string> & output_names,
            int batch_size,
            int num_infers_per_thread,
            int num_streams,
            int num_threads){
    graph::SetDefaultDevice("/device:GPU:0", &graph_def);

    // Creates a session.
    SessionOptions options;
    PrepareSessionOptionForDarvin(options, true); // for cuda graph
    // for cudagraph config
    std::unique_ptr<Session> session(NewSession(options));
    TF_CHECK_OK(session->Create(graph_def));
    // init host_allocator
    if (host_allocator == nullptr) {
      const DeviceMgr* device_manager;
      TF_CHECK_OK(session->LocalDeviceManager(&device_manager));
      std::vector<Device*> devices = device_manager->ListDevices();
      for (auto* d : devices) {
        if (d->attributes().device_type() == "CPU") {
          // todo: reuse this allocator
          host_allocator = dynamic_cast<ThreadPoolDevice*>(d)->GetAllocator(
              AllocatorAttributes());
        }
      }
    }

    SessionOptions options_tf;
    PrepareSessionOptionForDarvin(options_tf, false);
    std::unique_ptr<Session> session_tf(NewSession(options_tf));
    TF_CHECK_OK(session_tf->Create(graph_def));

    // Prepare inputs
    //
  for (int i = 0; i < 100; i++) {
     if(i % 10 == 0) {
      LOG(INFO) << "Round " << i;
   }
    std::vector<Tensor> input_tensors;
    GenerateInputs(graph_def, input_names, input_tensors, batch_size);
    InputsMap input_map; // input map for Normal TF run
    FillInputsMap(input_map, input_names, input_tensors);
    
    CudaGraphMgr& mgr = CudaGraphMgr::Singleton();

    std::vector<Tensor> output_tensors_cg;
    TF_CHECK_OK(session->Run(input_map, output_names, {}, &output_tensors_cg));

    std::vector<Tensor> output_tensors_tf;
    TF_CHECK_OK(session_tf->Run(input_map, output_names, {}, &output_tensors_tf));
/*
    LOG(INFO) << "CG results: ";
    tensor::PrintTensorData(output_tensors_cg[0]); 
    LOG(INFO) << "TF results: ";
    tensor::PrintTensorData(output_tensors_tf[0]); 
*/
    bool equal = tensor::CheckTensorEquality(output_tensors_cg[0], output_tensors_tf[0]);
    if (!equal) {
        LOG(INFO) << "check equality failed " << i;
    LOG(INFO) << "CG results: ";
    tensor::PrintTensorData(output_tensors_cg[0]);
    LOG(INFO) << "TF results: ";
    tensor::PrintTensorData(output_tensors_tf[0]);
    }
  }
    return Status();
}

Status CheckGraph(GraphDef& graph_def) {
    graph::SetDefaultDevice("/device:GPU:0", &graph_def);
    // Creates a session.
    SessionOptions options;
    PrepareSessionOptionForDarvin(options, true); // for cuda graph
    // for cudagraph config
    std::unique_ptr<Session> session(NewSession(options));
    TF_CHECK_OK(session->Create(graph_def));

    std::unordered_map<std::string, GraphDef>* graphs = session->GetCudaGraphRewriteDefs();
    for (auto iter = graphs->begin(); iter != graphs->end(); ++iter) {
        LOG(INFO) << "dump graphdef: " << iter->first; 
        DumpGraphDefToFile(iter->first, iter->second);
    }
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

//    example::CheckGraph(graph_def);
    return 0;
}
