/* Copyright 2018 The Sinian FPGA Compiler Authors. All Rights Reserved.*/
#include <string.h>
#include "fpga_runtime.h"
void AliFPGARuntime(
    void* /*s*/,
    bool* /*run_status_ok*/,
    int /*output_num*/,
    signed char** /*outputs*/,
    int /*input_num*/,
    signed char** /*inputs*/,
    void *) {
  return;
}

void AliFPGARuntimeCopyToCpu(signed char *cpu_addr,
    signed char *fpga_addr, int dim1, int dim2) {
  memcpy(cpu_addr, fpga_addr, dim1 * dim2 * 4);
}

void AliFPGARuntimeCopyToFPGA(signed char *fpga_addr,
    signed char *cpu_addr, int dim1, int dim2) {
  memcpy(fpga_addr, cpu_addr, dim1 * dim2 * 4);
}
