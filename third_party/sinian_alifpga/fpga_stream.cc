/* Copyright 2018 The Sinian FPGA Compiler Authors. All Rights Reserved.*/
#include "third_party/sinian_alifpga/fpga_stream.h"
namespace sinian_alifpga {
SinianAliFPGAStream::SinianAliFPGAStream() :
  code_(nullptr), inputs_(nullptr), outputs_(nullptr), valid_(false) {
}

SinianAliFPGAStream::~SinianAliFPGAStream() {
}

/* static */
void SinianAliFPGAStream::FreeFPGACode(FPGA_code* /*code*/) {
}

/* static */
void SinianAliFPGAStream::FreeFPGAOpListDeeply(FPGA_op_list* /*list*/) {
}

/* static */
void SinianAliFPGAStream::FreeFPGAOpList(FPGA_op_list* /*list*/) {
}

void SinianAliFPGAStream::SetCode(FPGA_code* /*code*/) {
}

FPGA_code* SinianAliFPGAStream::GetCode() {
  return nullptr;
}

std::string SinianAliFPGAStream::ToString() {
  return "";
}

void SinianAliFPGAStream::PushInPut(FPGA_op* /*op*/) {
}

void SinianAliFPGAStream::PushOutPut(FPGA_op* /*op*/) {
}

FPGA_op_list *SinianAliFPGAStream::GetInPutOpList() {
  return nullptr;
}

FPGA_op_list *SinianAliFPGAStream::GetOutPutOpList() {
  return nullptr;
}

void SinianAliFPGAStream::SetParameterIndex(FPGA_op* /*op*/, uint64_t /*index*/) {
}

bool SinianAliFPGAStream::HasParameterIndex(FPGA_op* /*op*/) {
  return false;
}

uint64_t SinianAliFPGAStream::GetParameterIndex(FPGA_op* /*op*/) {
  return 0;
}

/* static */
void SinianAliFPGAStream::DumpFPGAOperand(FPGA_op* /*op*/) {
}

void SinianAliFPGAStream::DumpFPGAOperands() {
}

void SinianAliFPGAStream::DumpCodeToFile(std::string idx) {
}

bool SinianAliFPGAStream::IsValid() {
  return valid_;
}

void SinianAliFPGAStream::SetValid(bool b) {
  valid_ = b;
  return;
}

}  // namespace sinian_alifpga
