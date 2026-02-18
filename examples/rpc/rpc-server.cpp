#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#include "ggml-rpc.h"
#ifdef GGML_USE_RDMA
#include "ggml-rdma.h"
#endif
#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#include <locale>
#include <windows.h>
#else
#define DIRECTORY_SEPARATOR '/'
#include <sys/stat.h>
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#endif
#include <algorithm>
#include <array>
#include <codecvt>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdio.h>
#include <string>
#include <thread>

namespace fs = std::filesystem;

// NOTE: this is copied from common.cpp to avoid linking with libcommon
// returns true if successful, false otherwise

#ifdef _WIN32
static std::wstring utf8_to_wstring(const std::string &str) {
  if (str.empty()) {
    return std::wstring();
  }

  int size =
      MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

  if (size <= 0) {
    return std::wstring();
  }

  std::wstring wstr(size, 0);
  MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

  return wstr;
}
#endif

static bool fs_create_directory_with_parents(const std::string &path) {
#ifdef _WIN32
  std::wstring wpath = utf8_to_wstring(path);

  // if the path already exists, check whether it's a directory
  const DWORD attributes = GetFileAttributesW(wpath.c_str());
  if ((attributes != INVALID_FILE_ATTRIBUTES) &&
      (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
    return true;
  }

  size_t pos_slash = 0;

  // process path from front to back, procedurally creating directories
  while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
    const std::wstring subpath = wpath.substr(0, pos_slash);
    const wchar_t *test = subpath.c_str();

    const bool success = CreateDirectoryW(test, NULL);
    if (!success) {
      const DWORD error = GetLastError();

      // if the path already exists, ensure that it's a directory
      if (error == ERROR_ALREADY_EXISTS) {
        const DWORD attributes = GetFileAttributesW(subpath.c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES ||
            !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
          return false;
        }
      } else {
        return false;
      }
    }

    pos_slash += 1;
  }

  return true;
#else
  // if the path already exists, check whether it's a directory
  struct stat info;
  if (stat(path.c_str(), &info) == 0) {
    return S_ISDIR(info.st_mode);
  }

  size_t pos_slash = 1; // skip leading slashes for directory creation

  // process path from front to back, procedurally creating directories
  while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
    const std::string subpath = path.substr(0, pos_slash);
    struct stat info;

    // if the path already exists, ensure that it's a directory
    if (stat(subpath.c_str(), &info) == 0) {
      if (!S_ISDIR(info.st_mode)) {
        return false;
      }
    } else {
      // create parent directories
      const int ret = mkdir(subpath.c_str(), 0755);
      if (ret != 0) {
        return false;
      }
    }

    pos_slash += 1;
  }

  return true;
#endif // _WIN32
}

// NOTE: this is copied from common.cpp to avoid linking with libcommon
static std::string fs_get_cache_directory() {
  std::string cache_directory = "";
  auto ensure_trailing_slash = [](std::string p) {
    // Make sure to add trailing slash
    if (p.back() != DIRECTORY_SEPARATOR) {
      p += DIRECTORY_SEPARATOR;
    }
    return p;
  };
  if (getenv("LLAMA_CACHE")) {
    cache_directory = std::getenv("LLAMA_CACHE");
  } else {
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX)
    if (std::getenv("XDG_CACHE_HOME")) {
      cache_directory = std::getenv("XDG_CACHE_HOME");
    } else {
      cache_directory = std::getenv("HOME") + std::string("/.cache/");
    }
#elif defined(__APPLE__)
    cache_directory = std::getenv("HOME") + std::string("/Library/Caches/");
#elif defined(_WIN32)
    cache_directory = std::getenv("LOCALAPPDATA");
#else
#error Unknown architecture
#endif
    cache_directory = ensure_trailing_slash(cache_directory);
    cache_directory += "llama.cpp";
  }
  return ensure_trailing_slash(cache_directory);
}

// Thunderbolt/RDMA interface detection — adapted from
// src/llama-tb-transport.cpp Self-contained: uses only POSIX APIs, no libllama
// dependency.
#ifdef __APPLE__
namespace rdma_detect {

static std::string trim_copy(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

static std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return s;
}

static void append_unique(std::vector<std::string> &out,
                          const std::string &value) {
  if (value.empty()) {
    return;
  }
  if (std::find(out.begin(), out.end(), value) == out.end()) {
    out.push_back(value);
  }
}

static bool run_command_capture(const char *cmd, std::string &out) {
  out.clear();
  FILE *pipe = ::popen(cmd, "r");
  if (!pipe) {
    return false;
  }
  std::array<char, 512> buf = {};
  while (std::fgets(buf.data(), (int)buf.size(), pipe) != nullptr) {
    out.append(buf.data());
  }
  return ::pclose(pipe) == 0;
}

static std::vector<std::string> detect_macos_thunderbolt_devices() {
  std::vector<std::string> devices;
  std::string out;
  if (!run_command_capture("networksetup -listallhardwareports", out)) {
    return devices;
  }
  std::stringstream ss(out);
  std::string line;
  std::string current_port;
  while (std::getline(ss, line)) {
    const std::string trimmed = trim_copy(line);
    if (trimmed.rfind("Hardware Port:", 0) == 0) {
      current_port = trim_copy(trimmed.substr(strlen("Hardware Port:")));
      continue;
    }
    if (trimmed.rfind("Device:", 0) == 0) {
      const std::string dev = trim_copy(trimmed.substr(strlen("Device:")));
      const std::string port = lower_copy(current_port);
      if (!dev.empty() && port.find("thunderbolt") != std::string::npos &&
          port.find("bridge") == std::string::npos) {
        append_unique(devices, dev);
      }
      current_port.clear();
    }
  }
  return devices;
}

struct iface_ip {
  std::string iface;
  std::string ip;
};

static std::vector<iface_ip>
collect_interface_ips(const std::vector<std::string> &ifnames) {
  std::vector<iface_ip> out;
  if (ifnames.empty()) {
    return out;
  }

  struct ifaddrs *ifaddr = nullptr;
  if (::getifaddrs(&ifaddr) != 0 || !ifaddr) {
    return out;
  }

  // Collect IPv4 first (preferred — ggml-rpc only supports AF_INET), then IPv6.
  for (int pass = 0; pass < 2; pass++) {
    const sa_family_t want = (pass == 0) ? AF_INET : AF_INET6;
    for (struct ifaddrs *cur = ifaddr; cur != nullptr; cur = cur->ifa_next) {
      if (!cur->ifa_name || !cur->ifa_addr) {
        continue;
      }
      if (!(cur->ifa_flags & IFF_UP)) {
        continue;
      }
      if (cur->ifa_addr->sa_family != want) {
        continue;
      }
      if (std::find(ifnames.begin(), ifnames.end(), cur->ifa_name) ==
          ifnames.end()) {
        continue;
      }

      char buf[INET6_ADDRSTRLEN] = {};
      if (want == AF_INET) {
        const struct sockaddr_in *sin =
            (const struct sockaddr_in *)cur->ifa_addr;
        if (::inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf)) == nullptr) {
          continue;
        }
      } else {
        const struct sockaddr_in6 *sin6 =
            (const struct sockaddr_in6 *)cur->ifa_addr;
        if (::inet_ntop(AF_INET6, &sin6->sin6_addr, buf, sizeof(buf)) ==
            nullptr) {
          continue;
        }
      }

      const std::string ip(buf);
      if (ip.empty() || ip == "127.0.0.1" || ip == "::1") {
        continue;
      }
      // Deduplicate by IP
      bool dup = false;
      for (const auto &entry : out) {
        if (entry.ip == ip) {
          dup = true;
          break;
        }
      }
      if (!dup) {
        out.push_back({std::string(cur->ifa_name), ip});
      }
    }
  }
  ::freeifaddrs(ifaddr);
  return out;
}

// Returns Thunderbolt/RDMA interface IPs with interface names for diagnostics.
static std::vector<iface_ip> detect_rdma_interfaces() {
  std::vector<std::string> ifnames;

  // Prefer explicit rdma_* interfaces when available.
  {
    struct ifaddrs *ifaddr = nullptr;
    if (::getifaddrs(&ifaddr) == 0 && ifaddr) {
      for (struct ifaddrs *cur = ifaddr; cur != nullptr; cur = cur->ifa_next) {
        if (!cur->ifa_name) {
          continue;
        }
        const std::string name(cur->ifa_name);
        if (name.rfind("rdma_", 0) == 0) {
          append_unique(ifnames, name);
        }
      }
      ::freeifaddrs(ifaddr);
    }
  }

  // Map Thunderbolt hardware ports to rdma_<enX> names.
  const std::vector<std::string> tb_devices =
      detect_macos_thunderbolt_devices();
  for (const std::string &dev : tb_devices) {
    append_unique(ifnames, "rdma_" + dev);
    append_unique(ifnames, dev);
  }

  // Also check bridge0 — Thunderbolt IP networking often appears here.
  append_unique(ifnames, "bridge0");

  return collect_interface_ips(ifnames);
}

// Collect all non-loopback IPv4 addresses for the diagnostic banner.
static std::vector<iface_ip> collect_all_ipv4_interfaces() {
  std::vector<iface_ip> out;
  struct ifaddrs *ifaddr = nullptr;
  if (::getifaddrs(&ifaddr) != 0 || !ifaddr) {
    return out;
  }

  for (struct ifaddrs *cur = ifaddr; cur != nullptr; cur = cur->ifa_next) {
    if (!cur->ifa_name || !cur->ifa_addr) {
      continue;
    }
    if (!(cur->ifa_flags & IFF_UP)) {
      continue;
    }
    if (cur->ifa_addr->sa_family != AF_INET) {
      continue;
    }

    char buf[INET_ADDRSTRLEN] = {};
    const struct sockaddr_in *sin = (const struct sockaddr_in *)cur->ifa_addr;
    if (::inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf)) == nullptr) {
      continue;
    }

    const std::string ip(buf);
    if (ip == "127.0.0.1") {
      continue;
    }
    out.push_back({std::string(cur->ifa_name), ip});
  }
  ::freeifaddrs(ifaddr);
  return out;
}

} // namespace rdma_detect
#endif // __APPLE__

struct rpc_server_params {
  std::string host = "127.0.0.1";
  int port = 50052;
  bool use_cache = false;
  bool use_cpu = false;
  int n_threads = std::max(1U, std::thread::hardware_concurrency() / 2);
  std::vector<std::string> devices;
  // RDMA / Thunderbolt transport options
  bool rdma = false;
  std::string rdma_backend_str = "auto"; // auto, jaccl, ibverbs, tcp
  std::string bind_addr;
  int socket_send_buf = 0; // effective value (includes --rdma auto-defaults)
  int socket_recv_buf = 0; // effective value (includes --rdma auto-defaults)
  int explicit_socket_send_buf = 0; // only set if user passed --socket-send-buf
  int explicit_socket_recv_buf = 0; // only set if user passed --socket-recv-buf
};

static void print_usage(int /*argc*/, char **argv, rpc_server_params params) {
  fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -h, --help                          show this help "
                  "message and exit\n");
  fprintf(stderr,
          "  -t, --threads N                     number of threads for the CPU "
          "device (default: %d)\n",
          params.n_threads);
  fprintf(stderr, "  -d, -dev, --device <dev1,dev2,...>  comma-separated list "
                  "of devices\n");
  fprintf(stderr, "  -cpu                                enable cpu backend\n");
  fprintf(
      stderr,
      "  -h, -H, --host, --Host HOST         host to bind to (default: %s)\n",
      params.host.c_str());
  fprintf(
      stderr,
      "  -p, -P, --port, --Port PORT         port to bind to (default: %d)\n",
      params.port);
  fprintf(stderr,
          "  -c, --cache                         enable local file cache\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "RDMA / Thunderbolt transport:\n");
  fprintf(stderr, "  --rdma                              auto-detect "
                  "Thunderbolt interfaces (macOS); sets\n");
  fprintf(stderr, "                                      host to 0.0.0.0. KV "
                  "handoff gets large socket\n");
  fprintf(stderr, "                                      buffers; RPC decode "
                  "uses system defaults for\n");
  fprintf(stderr, "                                      minimum latency\n");
  fprintf(stderr, "  --rdma-backend BACKEND              RDMA backend: auto, "
                  "jaccl, ibverbs, tcp (default: auto)\n");
  fprintf(stderr, "  --bind-addr ADDR                    explicit bind address "
                  "(overrides --host)\n");
  fprintf(stderr, "  --socket-send-buf N                 override SO_SNDBUF "
                  "for RPC sockets (0 = system default)\n");
  fprintf(stderr, "  --socket-recv-buf N                 override SO_RCVBUF "
                  "for RPC sockets (0 = system default)\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "KV handoff socket buffers are configured separately via:\n");
  fprintf(stderr, "  LLAMA_PREFILL_TB_SOCKET_SEND_BUF    SO_SNDBUF for TBP "
                  "streams (default: 4 MiB with --rdma)\n");
  fprintf(stderr, "  LLAMA_PREFILL_TB_SOCKET_RECV_BUF    SO_RCVBUF for TBP "
                  "streams (default: 4 MiB with --rdma)\n");
  fprintf(stderr, "\n");
}

static bool rpc_server_params_parse(int argc, char **argv,
                                    rpc_server_params &params) {
  std::string arg;
  for (int i = 1; i < argc; i++) {
    arg = argv[i];
    if (arg == "-H" || arg == "-h" || arg == "--host" || arg == "--Host") {
      if (++i >= argc) {
        return false;
      }
      params.host = argv[i];
    } else if (arg == "-t" || arg == "--threads") {
      if (++i >= argc) {
        return false;
      }
      params.n_threads = std::stoi(argv[i]);
      if (params.n_threads <= 0) {
        fprintf(stderr, "error: invalid number of threads: %d\n",
                params.n_threads);
        return false;
      }
    } else if (arg == "-d" || arg == "-dev" || arg == "--device") {
      if (++i >= argc) {
        return false;
      }
      const std::regex regex{R"([,/]+)"};
      std::string dev_str = argv[i];
      std::sregex_token_iterator iter(dev_str.begin(), dev_str.end(), regex,
                                      -1);
      std::sregex_token_iterator end;
      for (; iter != end; ++iter) {
        try {
          params.devices.push_back(*iter);
        } catch (const std::exception &) {
          fprintf(stderr, "error: invalid device: %s\n", iter->str().c_str());
          return false;
        }
      }
    } else if (arg == "-p" || arg == "-P" || arg == "--port" ||
               arg == "--Port") {
      if (++i >= argc) {
        return false;
      }
      params.port = std::stoi(argv[i]);
      if (params.port <= 0 || params.port > 65535) {
        return false;
      }
    } else if (arg == "-c" || arg == "--cache") {
      params.use_cache = true;
    } else if (arg == "-cpu") {
      params.use_cpu = true;
    } else if (arg == "--rdma") {
      params.rdma = true;
    } else if (arg == "--rdma-backend") {
      if (++i >= argc) {
        return false;
      }
      params.rdma_backend_str = argv[i];
      params.rdma = true; // --rdma-backend implies --rdma
    } else if (arg == "--bind-addr") {
      if (++i >= argc) {
        return false;
      }
      params.bind_addr = argv[i];
    } else if (arg == "--socket-send-buf") {
      if (++i >= argc) {
        return false;
      }
      params.socket_send_buf = std::stoi(argv[i]);
      params.explicit_socket_send_buf = params.socket_send_buf;
    } else if (arg == "--socket-recv-buf") {
      if (++i >= argc) {
        return false;
      }
      params.socket_recv_buf = std::stoi(argv[i]);
      params.explicit_socket_recv_buf = params.socket_recv_buf;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argc, argv, params);
      exit(0);
    } else {
      fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
      print_usage(argc, argv, params);
      exit(0);
    }
  }
  return true;
}

static ggml_backend_t create_cpu_backend(const rpc_server_params &params) {
  fprintf(stderr, "%s: using CPU backend\n", __func__);
  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_backend_cpu_set_n_threads(backend, params.n_threads);
  return backend;
}

static ggml_backend_t create_gpu_backend(const rpc_server_params &params,
                                         uint32_t device) {
  ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
  fprintf(stderr, "%s: using CUDA backend: CUDA%d\n", __func__, device);
  backend = ggml_backend_cuda_init(device, nullptr); // init device
  if (!backend) {
    fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
  }
#elif GGML_USE_METAL
  fprintf(stderr, "%s: using Metal backend\n", __func__);
  backend = ggml_backend_metal_init();
  if (!backend) {
    fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
  }
#elif GGML_USE_VULKAN
  fprintf(stderr, "%s: using Vulkan backend\n", __func__);
  backend = ggml_backend_vk_init(device); // init device 0
  if (!backend) {
    fprintf(stderr, "%s: ggml_backend_vulkan_init() failed\n", __func__);
  }
#elif GGML_USE_SYCL
  fprintf(stderr, "%s: using SYCL backend\n", __func__);
  backend = ggml_backend_sycl_init(device); // init device 0
  if (!backend) {
    fprintf(stderr, "%s: ggml_backend_sycl_init() failed\n", __func__);
  }
#endif
  // if there aren't GPU Backends fallback to CPU backend
  // if (!backend) {
  //    fprintf(stderr, "%s: using CPU backend\n", __func__);
  //    backend = ggml_backend_cpu_init();
  //    ggml_backend_cpu_set_n_threads(backend, params.n_threads);
  //}
  return backend;
}

static int32_t find_device_idx(const std::string &str) {
  std::regex pattern(R"((\d+)$)"); // Match digits at the end
  std::smatch matches;
  int number = -1;
  if (std::regex_search(str, matches, pattern)) {
    number = std::stoi(matches[1]);
  }
  return number;
}

static size_t get_gpu_backend_count(const rpc_server_params &params) {
  size_t count = 0;
#if defined(GGML_USE_CUDA)
  count = ggml_backend_cuda_get_device_count();
#elif defined(GGML_USE_SYCL)
  count = ggml_backend_sycl_get_device_count();
#elif defined(GGML_USE_VULKAN)
  count = ggml_backend_vk_get_device_count();
#elif defined(GGML_USE_CANN)
  return ggml_backend_cann_get_device_count();
#endif
  return count;
}

static std::vector<ggml_backend_t>
get_devices(const rpc_server_params &params) {
  std::vector<ggml_backend_t> devices;
  if (!params.devices.empty()) {
    for (auto device : params.devices) {
      int32_t device_id;
      ggml_backend_t dev;
      if (params.use_cpu && device == "CPU") {
        dev = create_cpu_backend(params);
      } else {
        device_id = find_device_idx(device);
        if (device_id < 0) {
          fprintf(stderr, "error: unknown device: %s\n", device.c_str());
          continue;
        }
        dev = create_gpu_backend(params, device_id);
      }
      if (dev) {
        devices.push_back(dev);
      } else {
        fprintf(stderr, "error: unknown device: %s\n", device.c_str());
      }
    }
  } else {
    for (size_t i = 0; i < get_gpu_backend_count(params); i++) {
      ggml_backend_t dev = create_gpu_backend(params, i);
      if (dev) {
        devices.push_back(dev);
      }
    }
    // cpu backend at last
    if (params.use_cpu || devices.empty()) {
      ggml_backend_t dev = create_cpu_backend(params);
      if (dev) {
        devices.push_back(dev);
      }
    }
  }
  return devices;
}

static void get_cpu_backend_memory(size_t *free_mem, size_t *total_mem) {
#ifdef _WIN32
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  *total_mem = status.ullTotalPhys;
  *free_mem = status.ullAvailPhys;
#else
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  *total_mem = pages * page_size;
  *free_mem = *total_mem;
#endif
}

static void get_backend_memory(uint32_t device, size_t *free_mem,
                               size_t *total_mem) {
#ifdef GGML_USE_CUDA
  ggml_backend_cuda_get_device_memory(device, free_mem, total_mem);
#elif GGML_USE_VULKAN
  ggml_backend_vk_get_device_memory(device, free_mem, total_mem);
#elif GGML_USE_SYCL
  ggml_backend_sycl_get_device_memory(device, free_mem, total_mem);
#else
#ifdef _WIN32
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  *total_mem = status.ullTotalPhys;
  *free_mem = status.ullAvailPhys;
#else
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  *total_mem = pages * page_size;
  *free_mem = *total_mem;
#endif
#endif
}

int main(int argc, char *argv[]) {
  rpc_server_params params;
  if (!rpc_server_params_parse(argc, argv, params)) {
    fprintf(stderr, "Invalid parameters\n");
    return 1;
  }

  // ---- RDMA / Thunderbolt transport resolution ----
  std::string
      rdma_ip; // Thunderbolt IP for the diagnostic banner (empty if none)
  std::string rdma_iface; // interface name for the TB IP

  if (!params.bind_addr.empty()) {
    // Explicit bind address overrides everything.
    params.host = params.bind_addr;
    fprintf(stderr, "[rdma] bind-addr override: %s\n", params.host.c_str());
  } else if (params.rdma) {
    // Auto-detect Thunderbolt interfaces; bind to 0.0.0.0 so both
    // RDMA (Thunderbolt) and TCP (Ethernet/Wi-Fi) clients can connect.
    params.host = "0.0.0.0";

#ifdef __APPLE__
    auto tb_ifaces = rdma_detect::detect_rdma_interfaces();
    if (!tb_ifaces.empty()) {
      rdma_ip = tb_ifaces[0].ip;
      rdma_iface = tb_ifaces[0].iface;
      fprintf(stderr, "[rdma] auto-detected Thunderbolt interface: %s (%s)\n",
              rdma_ip.c_str(), rdma_iface.c_str());
      for (size_t i = 1; i < tb_ifaces.size(); i++) {
        fprintf(stderr, "[rdma]   other candidate: %s (%s)\n",
                tb_ifaces[i].ip.c_str(), tb_ifaces[i].iface.c_str());
      }
    } else {
      fprintf(stderr, "[rdma] WARNING: --rdma specified but no Thunderbolt "
                      "interface found\n");
      fprintf(stderr, "[rdma]   checked: rdma_* interfaces, Thunderbolt "
                      "hardware ports, bridge0\n");
      fprintf(
          stderr,
          "[rdma]   falling back to 0.0.0.0 (plain TCP on all interfaces)\n");
    }
#else
    fprintf(stderr,
            "[rdma] WARNING: --rdma is only supported on macOS, ignoring\n");
#endif
    // Default socket buffers for Thunderbolt: 4 MiB (benefits high-BW,
    // low-latency links).
    if (params.socket_send_buf == 0) {
      params.socket_send_buf = 4 * 1024 * 1024;
    }
    if (params.socket_recv_buf == 0) {
      params.socket_recv_buf = 4 * 1024 * 1024;
    }
  }

  if (params.host != "127.0.0.1") {
    fprintf(stderr, "\n");
    fprintf(stderr,
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    fprintf(stderr, "WARNING: Host ('%s') is != '127.0.0.1'\n",
            params.host.c_str());
    fprintf(stderr,
            "         Never expose the RPC server to an open network!\n");
    fprintf(stderr,
            "         This is an experimental feature and is not secure!\n");
    fprintf(stderr,
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    fprintf(stderr, "\n");
  }

  auto devices = get_devices(params);
  if (devices.empty()) {
    fprintf(stderr, "No backend found\n");
    return 1;
  }

  std::string endpoint = params.host + ":" + std::to_string(params.port);
  std::vector<size_t> free_mem, total_mem;
  for (size_t i = 0; i < devices.size(); i++) {
    size_t free, total;
    const char *name = ggml_backend_name(devices[i]);
    if (std::string(name) == "CPU") {
      get_cpu_backend_memory(&free, &total);
    } else {
      int32_t idx = find_device_idx(name);
      get_backend_memory((uint32_t)idx, &free, &total);
    }
    free_mem.push_back(free);
    total_mem.push_back(total);
  }

  const char *cache_dir = nullptr;
  std::string cache_dir_str;
  if (params.use_cache) {
    cache_dir_str = fs_get_cache_directory() + "rpc/";
    if (!fs_create_directory_with_parents(cache_dir_str)) {
      fprintf(stderr, "Failed to create cache directory: %s\n",
              cache_dir_str.c_str());
      return 1;
    }
    cache_dir = cache_dir_str.c_str();
  }

  // Print RDMA diagnostic banner (printed after
  // ggml_backend_rpc_start_server_ex prints its own "Starting RPC server"
  // header, so we print ours first).
  if (params.rdma) {
    fprintf(stderr, "[rdma] transport      : rdma (with tcp fallback)\n");
    if (!rdma_ip.empty()) {
      fprintf(stderr, "[rdma] thunderbolt    : %s:%d (%s)\n", rdma_ip.c_str(),
              params.port, rdma_iface.c_str());
    }
#ifdef __APPLE__
    auto all_ips = rdma_detect::collect_all_ipv4_interfaces();
    for (const auto &entry : all_ips) {
      if (entry.ip == rdma_ip) {
        continue;
      }
      fprintf(stderr, "[rdma] tcp fallback   : %s:%d (%s)\n", entry.ip.c_str(),
              params.port, entry.iface.c_str());
    }
#endif
    if (params.socket_send_buf > 0 || params.socket_recv_buf > 0) {
      fprintf(stderr,
              "[rdma] kv buffers     : send=%d recv=%d (TBP throughput path)\n",
              params.socket_send_buf, params.socket_recv_buf);
    }
    fprintf(stderr,
            "[rdma] rpc buffers    : send=%d recv=%d (decode latency path)\n",
            params.explicit_socket_send_buf, params.explicit_socket_recv_buf);
  }

  // Only pass socket buffer overrides to the RPC server if the user
  // explicitly specified them via --socket-send-buf / --socket-recv-buf.
  // The --rdma auto-default of 4 MiB buffers is designed for KV handoff
  // throughput (TBP transport), NOT the latency-sensitive RPC decode path.
  // Large buffers inflate the TCP receive window, which interacts badly
  // with macOS delayed ACKs (~40ms) and causes bufferbloat on small
  // decode messages (graph_recompute = 13 bytes).
  // ---- RDMA kernel transport init ----
#ifdef GGML_USE_RDMA
  if (params.rdma) {
    // Parse --rdma-backend string to enum
    int rdma_backend_val = -1; // auto
    if (params.rdma_backend_str == "tcp") {
      rdma_backend_val = GGML_RDMA_TCP;
    } else if (params.rdma_backend_str == "jaccl") {
      rdma_backend_val = GGML_RDMA_JACCL;
    } else if (params.rdma_backend_str == "ibverbs") {
      rdma_backend_val = GGML_RDMA_IBVERBS;
    } else if (params.rdma_backend_str == "auto") {
      rdma_backend_val = -1;
    } else {
      fprintf(stderr,
              "[rdma] unknown backend: %s (use: auto, jaccl, ibverbs, tcp)\n",
              params.rdma_backend_str.c_str());
      return 1;
    }

    struct ggml_rdma_config rdma_cfg = {};
    if (rdma_backend_val >= 0) {
      rdma_cfg.backend = (enum ggml_rdma_backend)rdma_backend_val;
    } else {
      rdma_cfg.backend = ggml_rdma_best_available();
    }

    if (rdma_cfg.backend != GGML_RDMA_TCP) {
      if (ggml_rdma_init(&rdma_cfg)) {
        fprintf(stderr, "[rdma] kernel RDMA active: %s\n",
                ggml_rdma_backend_name(rdma_cfg.backend));
      } else {
        fprintf(stderr, "[rdma] kernel RDMA init failed, using TCP fallback\n");
      }
    } else {
      fprintf(stderr, "[rdma] no kernel RDMA backend available, using TCP\n");
    }
  }
#endif

  ggml_rpc_server_config config = {};
  config.socket_send_buf = params.explicit_socket_send_buf;
  config.socket_recv_buf = params.explicit_socket_recv_buf;
#ifdef GGML_USE_RDMA
  if (params.rdma) {
    // -1 = auto detect
    if (params.rdma_backend_str == "auto") {
      config.rdma_backend = -1;
    } else if (params.rdma_backend_str == "jaccl") {
      config.rdma_backend = GGML_RDMA_JACCL;
    } else if (params.rdma_backend_str == "ibverbs") {
      config.rdma_backend = GGML_RDMA_IBVERBS;
    }
  }
#endif
  ggml_backend_rpc_start_server_ex(endpoint.c_str(), cache_dir, devices.size(),
                                   devices.data(), free_mem.data(),
                                   total_mem.data(), &config);

#ifdef GGML_USE_RDMA
  ggml_rdma_shutdown();
#endif
  return 0;
}
