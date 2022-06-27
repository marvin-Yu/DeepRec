#include <numeric>
#include <fstream>
#include "absl/base/call_once.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/jit/xla_activity_listener.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/compiler/jit/xla_argument_dumper.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"

namespace tensorflow {

XlaArgumentDumper::XlaArgumentDumper() {
  tensorflow::ReadStringFromEnvVar("TF_XLA_ARGUMENT_CACHE_DIR", "",
                                   &cache_dir_);
}

bool XlaArgumentDumper::AsProto(
    const std::vector<XlaCompiler::Argument>& args,
    const std::map<int, std::string>& args_indexs,
    XlaArgumensProto& protos) {
  for (int i = 0; i < args.size(); i++) {
    const auto& arg = args[i];
    auto iter = args_indexs.find(i);
    if (iter == args_indexs.end()) return false;

    const std::string& name = iter->second;
    XlaArgumentProto* proto = protos.add_xla_arguments();

    if (arg.kind == XlaCompiler::Argument::kConstant) {
      proto->set_kind(CacheKind::kConstant);
      // constant value
      TensorProto* constant_value = proto->mutable_constant_value(); 
      arg.constant_value.AsProtoTensorContent(constant_value);
      VLOG(1) << name << " is const";
    } else if (arg.kind == XlaCompiler::Argument::kParameter) {
      proto->set_kind(CacheKind::kParameter);
      VLOG(1) << name << " is arg";
    } else {
      LOG(ERROR) << "Not support xla shape cache! argument type=" << arg.kind;
      return false;
    }
    // name
    proto->set_name(name);
    // type
    proto->set_type(arg.type);

    // tensor_shape
    TensorShape tensor_shape;
    TensorShapeProto* tensor_shape_proto = proto->mutable_shape();
    if (absl::holds_alternative<xla::Shape>(arg.shape)) {
      xla::Shape xla_shape = absl::get<xla::Shape>(arg.shape);
      if (!XLAShapeToTensorShape(xla_shape, &tensor_shape).ok()) {
         VLOG(0) << "convert shape fail";
         return false;
      }
    } else {
      tensor_shape = absl::get<TensorShape>(arg.shape);
    }
    tensor_shape.AsProto(tensor_shape_proto);
  }
  return true;
}

Status XlaArgumentDumper::DumpXlaArguments(
    const std::vector<XlaCompiler::Argument>& args,
    const std::map<int, std::string>& args_indexs,
    const std::string graph_key,
    const string& uuid) {
  if(cache_dir_.empty()) return Status::OK();

  auto env = tensorflow::Env::Default();
 
  string dir = tensorflow::io::JoinPath(cache_dir_, graph_key);
  string file_path = tensorflow::io::JoinPath(dir, std::to_string(tensorflow::Hash64(uuid)) + ".shape");

  if (env->FileExists(file_path).ok()) {
    VLOG(0) << "File exist, return " << file_path;
    return Status::OK();
  }
  
  if (!env->IsDirectory(dir).ok()) {
    auto status = env->RecursivelyCreateDir(dir);
    if (!status.ok() && !env->IsDirectory(dir).ok()) {
      LOG(ERROR) << "Could not create directory " << dir
                 << " for dumping xla shapes: " << status;
      return errors::Unimplemented("Dump xla shape failed ", uuid);
    }
  }

  XlaArgumensProto proto;
  if (!AsProto(args, args_indexs, proto)) return errors::Unimplemented("Dump xla shape failed");

  VLOG(0) << "Dump " << uuid << "; cache dir " << file_path; 
  return WriteTextProto(env, file_path, proto);
}

Status XlaArgumentDumper::ParseFromFile(
    const std::shared_ptr<InputsShapeInfo>& base,
    const std::string graph_key,
    const std::map<std::string, int>& indexs_args,
    std::vector<std::vector<XlaCompiler::Argument>>& args_array,
    std::vector<std::shared_ptr<InputsShapeInfo>>& inputs_shape_info_array) {
  if(cache_dir_.empty()) return Status::OK();

  args_array.clear();
  inputs_shape_info_array.clear();
  auto env = tensorflow::Env::Default();
  string dir = tensorflow::io::JoinPath(cache_dir_, graph_key);
  std::vector<string> files;
  if(!env->GetChildren(dir, &files).ok()) return errors::Unimplemented("Get cache files failed", dir);

  for (auto file_path: files) {
    file_path = tensorflow::io::JoinPath(dir, file_path);
    VLOG(0) << "Get file " << file_path;
    XlaArgumensProto protos;
    auto s =tensorflow::ReadTextProto(env, file_path, &protos);
    if (!s.ok()) return s;

    std::vector<XlaCompiler::Argument> args;
    if(!FromProto(protos, indexs_args, args)) continue;
    inputs_shape_info_array.push_back(BuildInputsShapeInfo(base, args)); 
    args_array.push_back(args);
  }
  return Status::OK();
}

std::shared_ptr<InputsShapeInfo> XlaArgumentDumper::BuildInputsShapeInfo(
      const std::shared_ptr<InputsShapeInfo>& base, 
      const std::vector<XlaCompiler::Argument>& args) {
  if(base == nullptr) return nullptr;

  std::shared_ptr<InputsShapeInfo> inputs = std::make_shared<InputsShapeInfo>();
  inputs->is_cpu_device = base->is_cpu_device;
  inputs->input_names = base->input_names;
  inputs->output_names = base->output_names;
  inputs->var_indexs = base->var_indexs;
  inputs->const_indexs = base->const_indexs;
  inputs->input_shapes.reserve(args.size());
  inputs->input_tensors.reserve(args.size()); 
  for (int i = 0; i < args.size(); i++) {
    TensorShape shape = absl::get<TensorShape>(args[i].shape);
    inputs->input_shapes.push_back(shape);
    if(std::find(inputs->const_indexs.begin(), inputs->const_indexs.end(), i) != inputs->const_indexs.end()) {
      inputs->input_tensors.push_back(args[i].constant_value);
    } else {
      inputs->input_tensors.push_back(Tensor(args[i].type, shape));
    }
  }
  inputs->refresh_size();
  return inputs;
}

std::vector<std::string> split(const std::string& s, const std::string& delim,
                               const bool keep_empty = true) {
  using namespace std;
  vector<string> result;
  if (delim.empty()) {
    result.push_back(s);
    return result;
  }
  string::const_iterator substart = s.begin(), subend;
  while (true) {
    subend = search(substart, s.end(), delim.begin(), delim.end());
    if (keep_empty || substart != subend) result.emplace_back(substart, subend);
    if (subend == s.end()) break;
    substart = subend + delim.size();
  }
  return result;
}

std::string combine_new_name(const std::vector<std::string> names, int len) {
  if (names.size() < len) return "";
  std::string res = "";
  for (int i = 0; i < len; i++) {
    res += names[i] + "_";
  }
  VLOG(0) << "len " << len << " return name " << res;
  return res;
}

int startsWith(string s, string sub){
  return s.find(sub)==0?1:0;
}

int get_name_index(std::string name, const std::map<std::string, int>& indexs_args) {
    int index = -1;
    auto iter = indexs_args.find(name);
    if (iter == indexs_args.end()) {
      VLOG(0) << "Read Xla arg error. cannot find index of " << name;
      std::vector<std::string> names = split(name, "_");
      for(int idx = 1; idx < names.size() - 2; idx++) {
        std::string new_name = combine_new_name(names, names.size() - idx);
        for(auto it = indexs_args.begin(); it != indexs_args.end(); it++) {
          VLOG(1) << "Current args is " << it->first << " " << it->second;
          if (startsWith(it->first, new_name)) {
            index = it->second;
            VLOG(0) << "Jinsi find " << name << "(" << new_name  << ") with " 
		    << it->first << " " << it->second;
            break;
          }
        }
        if (index >= 0) break;
      }
      if (index < 0) return index;
    } else {
      index = iter->second;
    }
    return index;
}

bool XlaArgumentDumper::FromProto(
    const XlaArgumensProto& protos,
    const std::map<std::string, int>& indexs_args,
    std::vector<XlaCompiler::Argument>& args) {

  args.resize(protos.xla_arguments_size());
  for (int i = 0; i < protos.xla_arguments_size(); i++) {
    const auto& proto = protos.xla_arguments(i);
    XlaCompiler::Argument arg;
  
    // type
    arg.type = proto.type();
   
    // shape
    TensorShape tensor_shape(proto.shape());
    arg.shape = tensor_shape;

    // kind
    if (proto.kind() == CacheKind::kConstant) {
      arg.kind = XlaCompiler::Argument::kConstant;
      // constant_value
      arg.constant_value.FromProto(proto.constant_value());
    } else if (proto.kind() == CacheKind::kParameter) {
      arg.kind = XlaCompiler::Argument::kParameter;
    }

    // name
    std::string name = proto.name();
    int index = get_name_index(name, indexs_args);
    VLOG(1) << name << " index is " << index;
    if (index < 0) {
      VLOG(0) << "Parse xla args file fail, cannt find " << name;
      return false;
    }

    args[index] = arg;
  } 
  return true;
}

} // namespace tensorflow
