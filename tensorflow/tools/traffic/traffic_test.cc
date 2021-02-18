#include <iostream>
#include <stdlib.h>
#include <thread>
#include "traffic.h"

int main(int nargc, char* pargv[]) {
  long ctrlflag = 1;
  if (nargc == 2) {
    ctrlflag = atol(pargv[1]);
  }
  ::Traffic::Instance()->SetCtrlFlag(ctrlflag);

  Traffic::Instance()->RegistRecord("One", Traffic::COUNT);
  Traffic::Instance()->RegistRecord("Two", Traffic::COUNT);
  Traffic::Instance()->RegistRecord("Thr", Traffic::COUNT);
  Traffic::Instance()->RegistRecord("Fur", Traffic::COUNT);
  Traffic::Instance()->RegistRecord("Fve", Traffic::STAT);
  Traffic::Instance()->RegistRecord("Six", Traffic::STAT);

  Traffic::Instance()->RegistRecord("One", Traffic::COUNT, "Mem");
  Traffic::Instance()->RegistRecord("Time", Traffic::TMRT, "Mem");
  Traffic::Instance()->RegistRecord("Succ", Traffic::RATE, "Mem");

  Traffic::Instance()->RegistRecord("One", Traffic::COUNT, "Nfm");
  Traffic::Instance()->RegistRecord("Two", Traffic::COUNT, "Nfm");
  Traffic::Instance()->RegistRecord("Thr", Traffic::COUNT, "Nfm");
  Traffic::Instance()->RegistRecord("Fur", Traffic::COUNT, "Nfm");
  Traffic::Instance()->RegistRecord("Fve", Traffic::STAT, "Nfm");
  Traffic::Instance()->RegistRecord("Six", Traffic::STAT, "Nfm");

  auto log_cb = [] (const char* loginfo) {
    std::cout << loginfo << std::endl;
  };

  Traffic::Instance()->Start(log_cb);
  volatile bool stop = false;

  auto thd1_fuc = [&stop] {
    while (!stop) {
      float seed = rand() * 1.0 / RAND_MAX;
      if (seed < 0.1) {
        Traffic::Instance()->Record("One", "Nfm");
        Traffic::Instance()->Record("One", "Mem");
      } else if (seed >= 0.1 && seed < 0.3) {
        Traffic::Instance()->Record("Two", "Nfm");
      } else if (seed >= 0.3 && seed < 0.35) {
        Traffic::Instance()->Record("Thr", "Nfm");
      } else if (seed >= 0.35 && seed < 0.38) {
        Traffic::Instance()->Record("Fur", "Nfm");
      } else if (seed >= 0.38 && seed < 0.68 ) {
        Traffic::Instance()->Record("Fve", seed, "Nfm");
      } else {
        Traffic::Instance()->Record("Six", seed * 2.5, "Nfm");
      }

      Traffic::Instance()->RecordTm("Time", rand() % 105, "Mem");

      usleep(rand() % 100000 + 100);
    }
  };

  for (int i = 0; i < 64; ++i) {
    std::thread thd1(thd1_fuc);
    thd1.detach();
  }

  auto thd2_fuc = [&stop] {
    while (!stop) {
      float seed = rand() * 1.0 / RAND_MAX;
      if (seed < 0.1) {
        Traffic::Instance()->Record("One");
      } else if (seed >= 0.1 && seed < 0.2) {
        Traffic::Instance()->Record("Two");
      } else if (seed >= 0.2 && seed < 0.35) {
        Traffic::Instance()->Record("Thr");
      } else if (seed >= 0.35 && seed < 0.48) {
        Traffic::Instance()->Record("Fur");
      } else if (seed >= 0.48 && seed < 0.98 ) {
        Traffic::Instance()->Record("Fve", seed);
      } else {
        Traffic::Instance()->Record("Six", seed * 2.5);
        if (rand() % 999 == 0) {
          Traffic::Instance()->RecordRate("Succ", false, "Mem");
        } else {
          Traffic::Instance()->RecordRate("Succ", true, "Mem");
        }
      }
      usleep(rand() % 100000 + 100);
    }
  };

  for (int i = 0; i < 64; ++i) {
    std::thread thd2(thd2_fuc);
    thd2.detach();
  }

  while(1) {
    sleep(600);
    stop = true;
    sleep(1);
    Traffic::Instance()->Stop();
  }

  return 0;
}
