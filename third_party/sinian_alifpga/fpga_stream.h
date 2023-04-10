#ifndef THIRD_PARTY_SINIAN_ALIFPGA_FPGA_STREAM_H_
#define THIRD_PARTY_SINIAN_ALIFPGA_FPGA_STREAM_H_
#include <string>
#include <map>
#include <vector>
#include <set>
#include <functional>
class FPGA_code;
class FPGA_op;
class FPGA_op_list;
enum stream_kind {
  sk_none = 0,
  sk_head = 1,
  sk_body = 2,
  sk_tail = 3
};
namespace sinian_alifpga {
class SinianAliFPGAStream {
 public:
  SinianAliFPGAStream();
  ~SinianAliFPGAStream();
  void SetCode(FPGA_code *code);
  FPGA_code* GetCode();
  void PushInPut(FPGA_op* op);
  void PushOutPut(FPGA_op* op);
  void SetParameterIndex(FPGA_op* op, uint64_t index);
  bool HasParameterIndex(FPGA_op* op);
  uint64_t GetParameterIndex(FPGA_op* op);
  FPGA_op_list* GetInPutOpList();
  FPGA_op_list* GetOutPutOpList();
  static void FreeFPGACode(FPGA_code* code);
  static void FreeFPGAOpListDeeply(FPGA_op_list* list);
  static void FreeFPGAOpList(FPGA_op_list* list);
  static void DumpFPGAOperand(FPGA_op* op);
  void DumpFPGAOperands();
  std::string ToString();
  void DumpCodeToFile(std::string);
  void DumpCodeToFile();
  void SetNextStream(SinianAliFPGAStream *);
  SinianAliFPGAStream *GetNextStream();
  void SetStreamKind(stream_kind);
  stream_kind GetStreamKind();
  bool IsValid();
  void SetValid(bool);
  bool IsTupleStream();
  void SetStreamOutputCount(uint64_t);
  uint64_t GetStreamOutputCount();
  bool SetNloSeq(std::vector<const void*>& nlo_seq);
  bool RunNloSeq(int output_num, signed char **outputs, int input_num,
      signed char **inputs, void *sv, void *call_frame = nullptr);
  void AddCachedInput(uint64_t idx);
  std::set<uint64_t> CachedInputs();
 private:
  FPGA_code *code_;
  FPGA_op_list *inputs_;
  FPGA_op_list *outputs_;
  std::map<FPGA_op*, uint64_t> parameter_map_;
  SinianAliFPGAStream *next_stream;
  stream_kind skind;
  bool valid_;
  uint64_t stream_output_count_;
  std::vector<const void*> nlo_seq_;
  uint64_t tempbuff_size_;
  std::set<uint64_t> cached_inputs_;
  std::vector<std::function<void(signed char *, signed char **, signed char **, void *, void *)>> nlo_func_;
};
} // sinian_alifpga
#endif // THIRD_PARTY_SINIAN_ALIFPGA_FPGA_STREAM_H_
