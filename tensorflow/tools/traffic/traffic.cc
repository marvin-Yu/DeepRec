#include "traffic.h"

#include <pthread.h>
#include <thread>
#include <mutex>
#include <chrono>

const std::string kDefaultTraffic = "Default";

#define CHECK_IF_TURNOFF(obj) { \
  if ((obj)->ctrl_flag_ == 0) { \
    return; \
  } \
}

#define CHECK_IF_TURNOFF_RETURN(obj, returnval) { \
  if ((obj)->ctrl_flag_ == 0) { \
    return (returnval); \
  } \
}

std::mutex g_mtx_;
Traffic* Traffic::instance_ = NULL;

void* Traffic::ReportFunc (void* data) {
  static std::ofstream ofstrm("/tmp/traffic.log");

  Traffic* this_ptr = (Traffic*)data;
  volatile int step = 0;
  while (this_ptr->stop_ == false) {
    CHECK_IF_TURNOFF_RETURN(this_ptr, NULL);
    
    ofstrm.clear();
    std::this_thread::sleep_for(std::chrono::microseconds(100000));
    ++step;
    if (step >= 300) {
      for (auto it = this_ptr->datas_[0].begin(), ite = this_ptr->datas_[0].end(); it != ite; ++it) {
        std::string outstr = this_ptr->Report(it->first);
        if (this_ptr->traffic_func_ == NULL) {
          ofstrm << outstr << std::endl;
        } else {
          this_ptr->traffic_func_(outstr.c_str());
        }
      }
      this_ptr->StandBy();
      step = 0;
    }
  }
  this_ptr->stop_ = true;

  return this_ptr;
}

Traffic* Traffic::Instance() {
  if (Traffic::instance_ == NULL) {
    Traffic::instance_ = new Traffic;
  }
  
  return Traffic::instance_;
}

void Traffic::RegistRecord(const std::string& dscstr, RecordType type,
    const std::string& traffic) {
  CHECK_IF_TURNOFF(this);

  g_mtx_.lock();
  if (datas_[0][traffic].find(dscstr) == datas_[0][traffic].end()) {
    datas_[0][traffic][dscstr] = Record_t(type, 0, 0.0f);
  }
  if (datas_[1][traffic].find(dscstr) == datas_[1][traffic].end()) {
    datas_[1][traffic][dscstr] = Record_t(type, 0, 0.0f);
  }
  g_mtx_.unlock();
}

void Traffic::Record(const std::string& key, float val,
    const std::string& traffic) {
  CHECK_IF_TURNOFF(this);

  const auto index = index_;
  auto it_traffic = datas_[index].find(traffic);
  if (it_traffic == datas_[index].end()) {
    return;
  }

  auto it = it_traffic->second.find(key);
  if (it != it_traffic->second.end()) {
    if (it->second.type_ == STAT) {
      it->second.count_ += 1;
      it->second.value_ += val;
    }
  }
}

void Traffic::RecordRate(const std::string& key, bool val,
    const std::string& traffic) {

  RecordRate(key, (val ? 1 : 0), 1, traffic);
}

void Traffic::RecordRate(const std::string& key,
                         long numerator, long denominator,
                         const std::string& traffic) {
  CHECK_IF_TURNOFF(this);

  const auto index = index_;
  auto it_traffic = datas_[index].find(traffic);
  if (it_traffic == datas_[index].end()) {
    return;
  }

  auto it = it_traffic->second.find(key);
  if (it != it_traffic->second.end()) {
    if (it->second.type_ == RATE) {
      it->second.count_ += denominator;
      it->second.value_ += numerator * 1.0f;
    }
  }
}

void Traffic::Record(const std::string& key, const timeval& s, const timeval& e,
    const std::string& traffic) {
  CHECK_IF_TURNOFF(this);

  const auto index = index_;
  auto it_traffic = datas_[index].find(traffic);
  if (it_traffic == datas_[index].end()) {
    return;
  }

  auto it = it_traffic->second.find(key);
  float tmcost = (e.tv_sec - s.tv_sec) * 1000.0f + (e.tv_usec - s.tv_usec) / 1000.0f;
  if (it != it_traffic->second.end()) {
    if (it->second.type_ == STAT) {
      it->second.value_ += tmcost;
      it->second.count_ += 1;
    }
  }
}

void Traffic::Record(const std::string& key,
    const std::string& traffic) {
  CHECK_IF_TURNOFF(this);

  const auto index = index_;
  auto it_traffic = datas_[index].find(traffic);
  if (it_traffic == datas_[index].end()) {
    return;
  }

  auto it = it_traffic->second.find(key);
  if (it != it_traffic->second.end()) {
    if (it->second.type_ == COUNT) {
      it->second.count_ += 1;
    }
  }
}

void Traffic::RecordTm(const std::string& key, float val,
    const std::string& traffic /* = kDefaultTraffic */) {
  CHECK_IF_TURNOFF(this);

  const auto index = index_;
  auto it_traffic = datas_[index].find(traffic);
  if (it_traffic == datas_[index].end()) {
    return;
  }

  auto it = it_traffic->second.find(key);
  if (it != it_traffic->second.end()) {
    if (it->second.type_ == TMRT) {
      it->second.count_ += 1;
      if (val <= 5.0f) {
        it->second.count_by_time_[0] += 1;
      }
      if (val <= 10.0f) {
        it->second.count_by_time_[1] += 1;
      }
      if (val <= 20.0f) {
        it->second.count_by_time_[2] += 1;
      }
      if (val <= 30.0f) {
        it->second.count_by_time_[3] += 1;
      }
      if (val <= 40.0f) {
        it->second.count_by_time_[4] += 1;
      }
      if (val <= 50.0f) {
        it->second.count_by_time_[5] += 1;
      }
      if (val <= 60.0f) {
        it->second.count_by_time_[6] += 1;
      }
      if (val <= 80.0f) {
        it->second.count_by_time_[7] += 1;
      }
      if (val <= 100.0f) {
        it->second.count_by_time_[8] += 1;
      }
      if (val > 100.0f) {
        it->second.count_by_time_[9] += 1;
      }
    }
  }
}

std::string Traffic::Report(const std::string& traffic) {
  // for report:
  std::string outstr = "Traffic(" + traffic + "):[";
  const auto index = index_;
  for (auto it = datas_[index][traffic].begin(), ite = datas_[index][traffic].end(); it != ite; ++it) {
    // for value:
    if (it->second.type_ == COUNT) {
      outstr += it->first + ":" + std::to_string(it->second.count_ / 30.0) + " ";
    } else if (it->second.type_ == STAT || it->second.type_ == RATE) {
      outstr += it->first + ":" + std::to_string(it->second.value_ / (it->second.count_ + 0.000001f)) + " ";
    } else if (it->second.type_ == TMRT) {
      outstr += it->first + "-ng5:" +   std::to_string(it->second.count_by_time_[0] / (it->second.count_ + 0.000001f)) + " ";
      outstr += it->first + "-ng10:" +  std::to_string(it->second.count_by_time_[1] / (it->second.count_ + 0.000001f)) + " ";
      outstr += it->first + "-ng20:" +  std::to_string(it->second.count_by_time_[2] / (it->second.count_ + 0.000001f)) + " ";
      outstr += it->first + "-ng30:" +  std::to_string(it->second.count_by_time_[3] / (it->second.count_ + 0.000001f)) + " ";
      outstr += it->first + "-ng40:" +  std::to_string(it->second.count_by_time_[4] / (it->second.count_ + 0.000001f)) + " ";
      outstr += it->first + "-ng50:" +  std::to_string(it->second.count_by_time_[5] / (it->second.count_ + 0.000001f)) + " ";
      outstr += it->first + "-ng60:" +  std::to_string(it->second.count_by_time_[6] / (it->second.count_ + 0.000001f)) + " ";
      outstr += it->first + "-ng80:" +  std::to_string(it->second.count_by_time_[7] / (it->second.count_ + 0.000001f)) + " ";
      outstr += it->first + "-ng100:" + std::to_string(it->second.count_by_time_[8] / (it->second.count_ + 0.000001f)) + " ";
      outstr += it->first + "-gt100:" + std::to_string(it->second.count_by_time_[9] / (it->second.count_ + 0.000001f)) + " ";
    }
  }
  outstr += "]";
  return outstr;
}

void Traffic::StandBy() {
  int index = index_;
  index_ = index_ ^ 1;
  for (auto it = datas_[index].begin(), ite = datas_[index].end(); it != ite; ++it) {
    for (auto it_traffic = it->second.begin(), ite_traffic = it->second.end(); it_traffic != ite_traffic; ++it_traffic) {
      it_traffic->second.count_ = 0;
      it_traffic->second.value_ = 0.0f;
      memset(&(it_traffic->second.count_by_time_[0]), 0, sizeof(it_traffic->second.count_by_time_));
    }
  }
}

bool Traffic::Start(TraffFunc_t* traffic_func) {
  CHECK_IF_TURNOFF_RETURN(this, true);

  g_mtx_.lock();
  if (is_runing_) {
    g_mtx_.unlock();
    return false;
  }
  is_runing_ = true;
  traffic_func_ = traffic_func;
  g_mtx_.unlock();

  pthread_t thd;
  pthread_create(&thd, NULL, Traffic::ReportFunc, this);

  return true;
}

void Traffic::Stop() {
  g_mtx_.lock();
  stop_ = true;
  g_mtx_.unlock();
  sleep(1);
}

void Traffic::SetCtrlFlag(long ctrl_flag) {
  ctrl_flag_ = ctrl_flag;
}

