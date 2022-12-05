/*
 *
 */
#ifndef ALIDLA_H_
#define ALIDLA_H_

#include <vector>
#include <string>

#define MAX_CU_NUM 4

typedef enum _AliDLAStatus_ {
  ALIDLA_ST_SUCCESS       = 0,
  ALIDLA_ST_FAIL          = -1,
  ALIDLA_ST_INVALID_PARAM = -2,
  ALIDLA_ST_BAD_DATA      = -3,
  ALIDLA_ST_DEVICE_ERROR  = -4,
  ALIDLA_ST_MEM_ERROR     = -5,
  ALIDLA_ST_SW_TIMEOUT    = -6,
  ALIDLA_ST_HW_TIMEOUT    = -7,
  ALIDLA_ST_INTERRUPT     = -8
} AliDLAStatus_t;
typedef struct _AliDLAShape_ {
  int64_t num_rows;
  int64_t num_cols;
  std::vector<int64_t> dim_size;
} AliDLAShape_t;

typedef enum _AliDLAActiveMode_ {
  MODE_BYPASS  = 0x0,
  MODE_SIGMOID = 0x1,
  MODE_TANH    = 0x2,
  MODE_RELU    = 0x3,
  MODE_DOTADD  = 0x80,
  MODE_DOTMUL  = 0x90,
  MODE_DOTSUB  = 0xc0,
  MODE_DOT1SUB = 0xe0,
  MODE_LINEAR  = 0xb0,
  MODE_LEAKYRELU = 0x109,  //mode2
} AliDLAActiveMode_t;


//DEVICE INFO LIST
typedef enum _AliDLAInfoEntry {
  ALIDLA_CUS_LOGIC_VERSION,
  ALIDLA_CUS_COMPUTE_UTIL,
  ALIDLA_CUS_DDR_UTIL,
  ALIDLA_CUS_FAIL_COUNT,
  ALIDLA_CUS_BS_EFFICIENCY,
  ALIDLA_CUS_DDR_BANDWIDTH,
  ALIDLA_CUS_COMPUTE_FLOPS,
  ALIDLA_CUS_HEALTHY,
  ALIDLA_CUS_ALL,
} AliDLAInfoEntry_t;

// ALL INFO for ALIDLA_CUS_ALL
typedef struct _AliDLAInfoSummary {
  int logic_version[MAX_CU_NUM];
  int compute_util[MAX_CU_NUM];   //单位百分比
  int ddr_util[MAX_CU_NUM];       //单位百分比
  int fail_count[MAX_CU_NUM];
  int bs_efficiency[MAX_CU_NUM];
  int ddr_bandwidth[MAX_CU_NUM];
  int compute_flops[MAX_CU_NUM];
  bool healthy[MAX_CU_NUM];
} AliDLAInfoSummary_t;


/*
 * AliDLAAllocMem : alloc specified DDR memory segment in AliFPGA device.
 * return: error codes
 *
 * "start": start address pointer.
 * "size" : size of DDR segment.
 * "flag": ddr bank info, eg "BANK0", "BANK0|BANK1" ...
 *         indicates the DDR bank range you want to use.
 */
AliDLAStatus_t AliDLAAllocMem(void **start, size_t size, int flag);

/*
 * AliDLAFreeMem : free specified DDR memory segment in AliFPGA device.
 * return: error codes
 *
 * "start": start address pointer.
 */
AliDLAStatus_t AliDLAFreeMem(void *start);

/*
 * AliDLACopyHostToDevice: copy memory from host to device.
 * return: error codes
 *
 * "dst"  : destination physical address in device memory.
 * "src"  : source address in host memory.
 * "size" : size of data.
 */
AliDLAStatus_t AliDLACopyHostToDevice(void *dst, const char *src, const AliDLAShape_t& src_shape);

AliDLAStatus_t AliDLACopyHostToDevice(void *dst, const char *src, const std::vector<int64_t>& dim_size);

/*
 * AliDLACopyDeviceToHost: copy memory from device to host.
 * return: error codes
 *
 * "dst"  : destination address in host memory.
 * "src"  : source physical address in device memory.
 * "size" : size of data.
 */
AliDLAStatus_t AliDLACopyDeviceToHost(char* dst, void *src, const AliDLAShape_t& src_shape);

AliDLAStatus_t AliDLACopyDeviceToHost(char* dst, void *src, const std::vector<int64_t>& dim_size);

AliDLAStatus_t AliDLALoadWeightToDevice(void *weight_host, const AliDLAShape_t& weight_shape,
                                        void *weight_device);

AliDLAStatus_t AliDLALoadWeightToDevice(void *weight_host, const std::vector<int64_t>& dim_size,
                                        void *weight_device);

/*
 * AliDLAInitDevice: Init Device, Load Insts.
 */
AliDLAStatus_t AliDLAInitDevice(const char* instrs_dir);

void AliDLAReleaseDevice();

void AliDLAAddrSetAttr(void *addr, int attr);

int AliDLAGetAttrFromAddr(void *addr);

int AliDLAGetDataBankInfo();

int AliDLAGetDataBankInfo(int attr_value);

AliDLAStatus_t AliDLAGetPlatformInfo(AliDLAInfoEntry_t info_name, size_t param_value_size,
                                     void *param_value, size_t *param_value_size_ret);

int AliDLAGetPlatformComputeUnitNum();

void* AlliDLAGetHostAddrFromDeviceAddr(const void* addr);

void AliDLAVerifyResult(const void *fpga_addr, const void *cpu_addr, size_t num_bytes);

bool CheckAliDLAVerifyConfig();

bool CheckAliDLAHealthy();

void SetConstWeightKey(const void *host_addr);

void ClearConstWeightKey(const void *host_addr);

#endif
