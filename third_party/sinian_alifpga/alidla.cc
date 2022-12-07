/*  */


#include "alidla.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>


AliDLAStatus_t AliDLAAllocMem(void **start, size_t size, int /*flag*/) {
  if (start == nullptr) {
    return ALIDLA_ST_FAIL;
  }
  *start = malloc(size);
  return ALIDLA_ST_SUCCESS; 
}

AliDLAStatus_t AliDLAFreeMem(void *start) {
  if (start) {
    free(start);
  }
  return ALIDLA_ST_SUCCESS;
}

static size_t
ComputeShapeSize(const std::vector<int64_t>& dim_size) {
  size_t all_size = 1;
  for (auto size: dim_size) {
    all_size *= size;
  }
  all_size *= sizeof(float);
  return all_size;
}

static AliDLAStatus_t
CopyHelper(void *dst, const void *src, const std::vector<int64_t>& dim_size) {
  if (dst == nullptr || src == nullptr) {
    return ALIDLA_ST_FAIL;
  }
  memcpy(dst, src, ComputeShapeSize(dim_size));
  return ALIDLA_ST_SUCCESS;
}

AliDLAStatus_t AliDLACopyHostToDevice(void *dst, const char *src, const std::vector<int64_t>& dim_size) {
  return CopyHelper(dst, src, dim_size);
}

AliDLAStatus_t AliDLACopyDeviceToHost(char* dst, void *src, const std::vector<int64_t>& dim_size) {
  return CopyHelper(dst, src, dim_size);
}

AliDLAStatus_t AliDLACopyHostToDevice(void *dst, const char *src, const AliDLAShape_t& src_shape) {
  if (dst == nullptr || src == nullptr) {
    return ALIDLA_ST_FAIL;
  }
  size_t size = src_shape.num_rows * src_shape.num_cols * sizeof(float);
  memcpy(dst, src, size);
  return ALIDLA_ST_SUCCESS;
}

AliDLAStatus_t AliDLACopyDeviceToHost(char* dst, void *src, const AliDLAShape_t& src_shape) {
  if (dst == nullptr || src == nullptr) {
    return ALIDLA_ST_FAIL;
  }
  size_t size = src_shape.num_rows * src_shape.num_cols * sizeof(float);
  memcpy(dst, src, size);
  return ALIDLA_ST_SUCCESS;
}

AliDLAStatus_t AliDLALoadWeightToDevice(void *weight_host, const AliDLAShape_t& weight_shape, void *weight_device) {
  if (weight_host == nullptr || weight_device == nullptr) {
    return ALIDLA_ST_FAIL;
  }
  size_t size = weight_shape.num_rows * weight_shape.num_cols * sizeof(float);
  memcpy(weight_device, weight_host, size);
  return ALIDLA_ST_SUCCESS;
}

AliDLAStatus_t AliDLALoadWeightToDevice(void *weight_host, const std::vector<int64_t>& dim_size, void *weight_device) {
  return CopyHelper(weight_device, weight_host, dim_size);
}

AliDLAStatus_t AliDLAInitDevice(const char* /*instrs_dir*/) {
  /*Unimplemented*/
  return ALIDLA_ST_SUCCESS;
}

void AliDLAReleaseDevice() {
  /*Unimplemented*/
  return;
}

int AliDLAGetAttrFromAddr(void* /*addr*/) {
  /*Unimplemented*/
  return 0x1;
}

void AliDLAAddrSetAttr(void *addr, int attr) {
  /*Unimplemented*/
  return;
}

int AliDLAGetDataBankInfo(int /*attr_value*/) {
  /*Unimplemented*/
  return 0x1;
}

int AliDLAGetDataBankInfo() {
  /*Unimplemented*/
  return 0x1;
}

AliDLAStatus_t AliDLAGetPlatformInfo(AliDLAInfoEntry_t /*info_name*/, size_t /*param_value_size*/,
                                     void* /*param_value*/, size_t* /*param_value_size_ret*/) {
  /*Unimplemented*/
  return ALIDLA_ST_SUCCESS;
}

int AliDLAGetPlatformComputeUnitNum() {
  return MAX_CU_NUM;
}

void* AlliDLAGetHostAddrFromDeviceAddr(const void* addr) {
  /*Unimplemented*/
  return (void *)addr;
}

void AliDLAVerifyResult(const void* /*fpga_addr*/, const void* /*cpu_addr*/, size_t /*num_bytes*/) {
  /*Unimplemented*/
  return;
}

bool CheckAliDLAVerifyConfig() {
  return false;
}

bool CheckAliDLAHealthy() {
  return true;
}

void SetConstWeightKey(const void *host_addr) {
}

void ClearConstWeightKey(const void *host_addr) {
}

