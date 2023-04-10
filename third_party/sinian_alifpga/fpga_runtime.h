#ifndef THIRD_PARTY_SINIAN_ALIFPGA_FPGA_RUNTIME_H_
#define THIRD_PARTY_SINIAN_ALIFPGA_FPGA_RUNTIME_H_
// If compile release version, change this maroc to 1.
#define RELEASE_MODE 1
extern "C" {
extern void AliFPGARuntime(
    void *s,
    bool* run_status_ok,
    int output_num,
    signed char **outputs,
    int input_num,
    signed char **inputs,
    void *call_frame = nullptr);

extern void AliFPGARuntimeCopyToCpu(signed char *cpu_addr,
    signed char *fpga_addr, int dim1, int dim2);
extern void AliFPGARuntimeCopyToFPGA(signed char *fpga_addr,
    signed char *cpu_addr, int dim1, int dim2);
}  // extern "C"
#endif // THIRD_PARTY_SINIAN_ALIFPGA_FPGA_RUNTIME_H_

