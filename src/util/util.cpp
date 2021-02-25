#include <unistd.h>
#include <ios>
#include <fstream>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <sstream>
#include <array>
#include <memory>
#include "util.h"

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

unsigned int getNumThreads() {
    pid_t pid = getpid();
    std::string command = "cat /proc/" + std::to_string(pid) + "/status | grep Threads";
    auto retStr = exec(command.c_str());

    auto numThreads = retStr.substr(9);
    std::stringstream ss(numThreads);

    int nThreads;
    ss >> nThreads;

    return nThreads;
}

double getProcessMemUsage() {
    using std::ios_base;
    using std::ifstream;
    using std::string;

//        double vmUsage     = 0.0;
    double residentSet = 0.0;

    // 'file' stat seems to give the most reliable results
    ifstream stat_stream("/proc/self/stat", ios_base::in);

    // dummy vars for leading entries in stat that we don't care about
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;

    // the two fields we want
    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
                >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
                >> utime >> stime >> cutime >> cstime >> priority >> nice
                >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

    stat_stream.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
//        vmUsage     = vsize / 1024.0;
    residentSet = rss * page_size_kb;

    return residentSet;
}
