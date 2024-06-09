#include <Eigen/Dense>
#include <digilent/waveforms/dwf.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

constexpr char const *help_text = "Usage: DEVICE BAUD";

constexpr std::size_t size_128bit = 16u;
// samples_count 8000 hz_rate 65e6 traces_count 50000 repetitions_count 20
constexpr unsigned samples_count = 8000u;
constexpr double hz_rate = 65e6;
constexpr std::size_t traces_count = 50000u;
constexpr std::size_t repetitions_count = 20u;

template <std::size_t N>
static std::string_view arr_to_sv(std::array<char, N> const &arr) {
  return std::string_view(arr.data(), arr.size());
}

static void binary_to_hex(std::span<std::uint8_t const> src,
                          std::span<char> dst) {
  auto f = [](std::uint8_t b) -> char {
    if (b <= 9) {
      return b + '0';
    } else {
      return b - 10 + 'A';
    }
  };

  auto it{dst.begin()};
  for (std::uint8_t b : src) {
    assert(it != dst.end() || it + 1 != dst.end());
    *it = f(b >> 4);
    ++it;
    *it = f(b & 0xF);
    ++it;
  }
}

static auto set_uart(int fd, unsigned long baud_rate) -> bool {
  struct termios termios;
  if (tcgetattr(fd, &termios) == -1) {
    return false;
  }

  cfmakeraw(&termios);

  termios.c_cc[VTIME] = 0;
  termios.c_cc[VMIN] = 32;

  return cfsetspeed(&termios, baud_rate) != -1 &&
         tcsetattr(fd, TCSANOW, &termios) != -1;
}

static auto cipher_uart_comm(int fd, std::string_view const plaintext,
                             std::string_view const key,
                             std::span<char> ciphertext) -> bool {
  return write(fd, plaintext.data(), plaintext.size()) != -1 &&
         write(fd, key.data(), key.size()) != -1 &&
         read(fd, ciphertext.data(), ciphertext.size()) != -1;
}

class FileDescriptor {
public:
  explicit FileDescriptor(int fd) : fd_{fd} {}

  FileDescriptor(const FileDescriptor &) = delete;

  ~FileDescriptor() {
    if (fd_ != -1) {
      close(fd_);
    }
  }

  FileDescriptor &operator=(const FileDescriptor &) = delete;

  auto get() -> int { return fd_; }

  void leak() { fd_ = -1; }

private:
  int fd_;
};

auto main(int argc, char **argv) -> int {
  char const *device_path;
  unsigned long baud_rate;
  if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
    std::cout << help_text << '\n';
    return EXIT_SUCCESS;
  } else if (argc == 3) {
    device_path = argv[1];
    char const *first = argv[2];
    char const *last = first + std::strlen(first);
    auto [ptr, ec] = std::from_chars(first, last, baud_rate);
    if (ec != std::errc() || ptr != last) {
      std::cerr << "malformed baud rate\n";
      return EXIT_FAILURE;
    }
  } else {
    std::cout << help_text << '\n';
    return EXIT_FAILURE;
  }

  FileDescriptor fd{open(device_path, O_RDWR)};
  if (fd.get() == -1) {
    std::cerr << "open: " << std::strerror(errno) << '\n';
    return EXIT_FAILURE;
  }

  if (!set_uart(fd.get(), baud_rate)) {
    std::cerr << "set_uart: " << std::strerror(errno) << '\n';
    return EXIT_FAILURE;
  }

  HDWF hdwf;
  std::array<char, 512> err_msg;
  if (!FDwfDeviceOpen(-1, &hdwf)) {
    FDwfGetLastErrorMsg(err_msg.data());
    std::cerr << "Device open failed: " << err_msg.data();
    return EXIT_FAILURE;
  }

  FDwfDeviceAutoConfigureSet(hdwf, 0);
  FDwfAnalogInFrequencySet(hdwf, hz_rate);
  FDwfAnalogInBufferSizeSet(hdwf, samples_count);
  FDwfAnalogInChannelEnableSet(hdwf, 0, true);
  FDwfAnalogInChannelRangeSet(hdwf, 0, 12.0);

  FDwfAnalogInTriggerAutoTimeoutSet(hdwf, 0.0);
  FDwfAnalogInTriggerPositionSet(hdwf, 0.5 * samples_count / hz_rate);

  FDwfAnalogInTriggerSourceSet(hdwf, trigsrcExternal1);
  FDwfAnalogInTriggerConditionSet(hdwf, DwfTriggerSlopeRise);

  FDwfAnalogInConfigure(hdwf, true, false);

  std::this_thread::sleep_for(std::chrono::seconds(2));

  FDwfAnalogInConfigure(hdwf, false, true);

  std::random_device rd;
  std::uniform_int_distribution<> dist(0, 255);

  std::fstream key_fs("key.txt", std::ios::out | std::ios::trunc);
  std::fstream traces_fs("traces.csv", std::ios::out | std::ios::trunc);
  std::fstream inputs_fs("inputs.csv", std::ios::out | std::ios::trunc);

  std::array<std::uint8_t, size_128bit> key;
  std::generate(key.begin(), key.end(), [&]() { return dist(rd); });

  std::array<char, size_128bit * 2> key_text;
  binary_to_hex(key, key_text);

  std::cout << "key:    " << arr_to_sv(key_text) << '\n';

  key_fs << arr_to_sv(key_text) << std::endl;

  Eigen::VectorXd samples(samples_count);
  Eigen::VectorXd trace(samples_count);
  for (std::size_t i{0}; i != traces_count; ++i) {
    std::array<std::uint8_t, size_128bit> plain;
    std::generate(plain.begin(), plain.end(), [&]() { return dist(rd); });

    inputs_fs << +plain[0] << std::endl;

    std::array<char, size_128bit * 2> plain_text;
    binary_to_hex(plain, plain_text);

    samples.fill(0);
    for (std::size_t j{0}; j != repetitions_count; ++j) {
      std::array<char, size_128bit * 2> cipher_text;
      if (!cipher_uart_comm(fd.get(), arr_to_sv(plain_text),
                            arr_to_sv(key_text), cipher_text)) {
        FDwfDeviceCloseAll();
        return EXIT_FAILURE;
      }

      std::cout << "plain:  " << arr_to_sv(plain_text)
                << "\ncipher: " << arr_to_sv(cipher_text) << '\n';

      for (;;) {
        DwfState sts;
        FDwfAnalogInStatus(hdwf, true, &sts);
        if (sts == DwfStateDone) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      FDwfAnalogInStatusData(hdwf, 0, trace.data(), trace.size());
      samples += trace;
    }

    samples /= repetitions_count;

    auto it{samples.begin()};
    for (; it + 1 != samples.end(); ++it) {
      traces_fs << *it << ',';
    }
    traces_fs << *it << std::endl;
  }

  FDwfDeviceCloseAll();
}
