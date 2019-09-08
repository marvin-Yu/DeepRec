#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_COMMON_DEFINES_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_COMMON_DEFINES_H_

#include <string>
#include <vector>
#include "tensorflow/core/platform/default/logging.h"

const std::string kEmptyString = "";

#define CONCATENATE_IMPL(s1, s2)  s1##s2
#define CONCATENATE(s1, s2) CONCATENATE_IMPL(s1, s2)
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __LINE__)

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_COMMON_DEFINES_H_