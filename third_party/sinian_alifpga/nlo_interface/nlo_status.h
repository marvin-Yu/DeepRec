#ifndef NLO_INTERFACE_NLO_STATUS_H_
#define NLO_INTERFACE_NLO_STATUS_H_
#include <cstdint>
namespace sinian {
class NloStatus {
 public:
  NloStatus() {}
  NloStatus(int32_t status) {
    status_ =  status;
  }
  ~NloStatus() {}
  static NloStatus OK() {
    return NloStatus(0);
  }
  static NloStatus Unimplemented() {
    return NloStatus(1);
  }

  static NloStatus FailedPrecondition() {
    return NloStatus(2);
  }

  static NloStatus UNKNOWN() {
    return NloStatus(3);
  }

  bool ok() {
    return status_ == 0;
  }

 private:
  int32_t status_;
};
} // namespace sinian
#endif // NLO_INTERFACE_NLO_STATUS_H_