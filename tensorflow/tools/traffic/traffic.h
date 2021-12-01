#ifndef __MY_TRAFFIC_STAT_H
#define __MY_TRAFFIC_STAT_H

#include <string>
#include <string.h>
#include <map>
#include <fstream>
#include <unistd.h>
#include <sys/time.h>

extern const std::string kDefaultTraffic;

class Traffic {
  Traffic(const Traffic&);
  Traffic operator = (const Traffic&);
  Traffic() : index_(0), is_runing_(false), stop_(false), traffic_func_(NULL), ctrl_flag_(1) {}
public:
  typedef void(TraffFunc_t)(const char*);
  enum RecordType {
    COUNT,
    STAT,
    TMRT,
    RATE
  };
private:
  struct Record_t {
    Record_t(RecordType type, long count, float val) 
      : type_(type), count_(count), value_(val) {
        memset(&count_by_time_[0], 0, sizeof(count_by_time_));
    }
    Record_t(): type_(COUNT), count_(0), value_(0.0f) {
    }
    RecordType type_;
    long count_;
    float value_;

    // 0-5, 0-10, 0-20, 0-30, 0-40, 0-50, 0-60, 0-80, 0-100, 100+t 
    float count_by_time_[10];
  };
  typedef std::map<std::string, Record_t> RecordData_t;

  static void* ReportFunc (void* data);
public:
  void SetCtrlFlag(long ctrl_flag);

  static Traffic* Instance();

  // @brief: 注册特定的Traffic
  // @param dscstr: Traffic 描述字符串
  // @param type: 计数还是统计数值
  // @param traffic: 特定的频道[Default]
  void RegistRecord(const std::string& dscstr, RecordType type,
      const std::string& traffic = kDefaultTraffic);

  // @brief: 注册特定的Traffic
  // @param key: Traffic 描述字符串
  // @param val: 记录的数值
  // @param traffic: 特定的频道[Default]
  void Record(const std::string& key, float val,
      const std::string& traffic = kDefaultTraffic);

  // @brief: 注册特定的Traffic
  // @param key: Traffic 描述字符串
  // @param val: 成功失败标记
  // @param traffic: 特定的频道[Default]
  void RecordRate(const std::string& key, bool val,
      const std::string& traffic = kDefaultTraffic);

  // @brief: 注册特定的Traffic
  // @param key: Traffic 描述字符串
  // @param numerator: 计算百分比的分子
  // @param denominator: 计算百分比的分母
  // @param traffic: 特定的频道[Default]
  void RecordRate(const std::string& key,
      long numerator, long denominator,
      const std::string& traffic = kDefaultTraffic);

  // @brief: 注册特定的Traffic
  // @param key: Traffic 描述字符串
  // @param s, e: 用于记录时间，开始时间点和结束时间点
  // @param traffic: 特定的频道[Default]
  void Record(const std::string& key, const timeval& s, const timeval& e,
      const std::string& traffic = kDefaultTraffic);

  // @brief: 注册特定的Traffic
  // @param key: Traffic 描述字符串, 不需要val，调用一次，计数增加1
  // @param traffic: 特定的频道[Default]
  void Record(const std::string& key, const std::string& traffic = kDefaultTraffic);

  // @brief: 注册特定的Traffic
  // @param key: Traffic 描述字符串
  // @param val: 耗时时间, 用于统计各个区间的成功率
  // @param traffic: 特定的频道[Default]
  void RecordTm(const std::string& key, float val,
      const std::string& traffic = kDefaultTraffic);

  bool Start(TraffFunc_t* traffic_func = NULL);

  void Stop();

private:
  void StandBy();
  std::string Report(const std::string& traffic = kDefaultTraffic);

  std::map<std::string, RecordData_t> datas_[2];
  int index_;
  bool is_runing_;
  volatile bool stop_;
  TraffFunc_t* traffic_func_;

  static Traffic* instance_;
  volatile long ctrl_flag_;
};

#endif // __MY_TRAFFIC_STAT_H
