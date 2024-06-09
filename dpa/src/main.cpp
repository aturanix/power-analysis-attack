#include <Eigen/Dense>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

static auto read_inputs(char const *file, std::vector<std::uint8_t> &vec)
    -> bool {
  std::fstream fs(file);
  if (!fs.is_open()) {
    return false;
  }

  int input;
  for (;;) {
    fs >> input;
    if (fs.eof()) {
      vec.shrink_to_fit();
      return true;
    } else if (!fs.good()) {
      return false;
    }
    vec.push_back(input);
  }
  return true;
}

template <typename T>
static auto parse_line(char const *first, char const *last, std::vector<T> &vec)
    -> bool {
  vec.clear();

  if (first == last) {
    return true;
  }

  for (;;) {
    T val;
    std::from_chars_result res = std::from_chars(first, last, val);
    if (res.ec != std::errc()) {
      return false;
    }

    vec.push_back(val);

    if (res.ptr == last) {
      vec.shrink_to_fit();
      return true;
    }

    first = res.ptr + 1;
  }
  return true;
}

static auto read_traces(char const *file, std::vector<std::vector<double>> &vec)
    -> bool {
  std::fstream fs(file);
  if (!fs.is_open()) {
    return false;
  }

  std::string line;
  while (std::getline(fs, line)) {
    std::vector<double> trace;
    if (!parse_line(line.data(), line.data() + line.size(), trace)) {
      return false;
    }
    vec.push_back(trace);
  }
  vec.shrink_to_fit();
  return true;
}

static auto read_subbytes(char const *file, std::vector<std::uint8_t> &vec)
    -> bool {
  std::fstream fs(file);
  if (!fs.is_open()) {
    return false;
  }

  std::string line;
  return std::getline(fs, line) &&
         parse_line(line.data(), line.data() + line.size(), vec);
}

static auto read_bytehammingweight(char const *file,
                                   std::vector<std::uint8_t> &vec) -> bool {
  std::fstream fs(file);
  if (!fs.is_open()) {
    return false;
  }

  std::string line;
  return std::getline(fs, line) &&
         parse_line(line.data(), line.data() + line.size(), vec);
}

static auto write_key_traces(char const *file, Eigen::MatrixXd const &vec)
    -> bool {
  std::fstream fs(file, std::ios_base::out | std::ios_base::trunc);
  if (!fs.is_open()) {
    return false;
  }

  for (auto const &key_trace : vec.rowwise()) {
    auto it{key_trace.begin()};
    for (; it != key_trace.end() - 1; ++it) {
      fs << *it << ',';
    }
    fs << *it << '\n';
  }

  return true;
}

static auto kocher(Eigen::VectorX<std::uint8_t> const &inputs,
                   Eigen::MatrixXd const &traces,
                   Eigen::VectorX<std::uint8_t> const &subbytes)
    -> Eigen::MatrixXd {
  Eigen::MatrixXd key_traces(256, traces.cols());
  for (Eigen::Index i{0}; i != inputs.size(); ++i) {
    for (std::uint8_t b{0};; ++b) {
      std::uint8_t sbox_res = subbytes[inputs[i] ^ b];
      if (sbox_res & 0b1) {
        key_traces(b, Eigen::all) += traces(i, Eigen::all);
      } else {
        key_traces(b, Eigen::all) -= traces(i, Eigen::all);
      }

      if (b == 255) {
        break;
      }
    }
  }
  return key_traces;
}

static auto calculate_correlation_matrix(Eigen::MatrixXd const &in)
    -> Eigen::MatrixXd {
  Eigen::MatrixXd x_minus_mean = in.rowwise() - in.colwise().mean();
  Eigen::VectorXd s_j =
      (x_minus_mean.array() * x_minus_mean.array()).colwise().sum().sqrt();
  Eigen::MatrixXd mat(in.cols(), in.cols());
  for (Eigen::Index i{0}; i != mat.rows(); ++i) {
    for (Eigen::Index j{0}; j != i; ++j) {
      double s_jk = (x_minus_mean(Eigen::all, i).array() *
                     x_minus_mean(Eigen::all, j).array())
                        .sum();
      double res = s_jk / s_j(i) / s_j(j);
      mat(i, j) = res;
      mat(j, i) = res;
    }
    mat(i, i) = 1.0;
  }
  return mat;
}

static auto calculate_cmatrix(
    Eigen::VectorX<std::uint8_t> const &inputs, Eigen::MatrixXd const &traces,
    Eigen::VectorX<std::uint8_t> const &subbytes,
    Eigen::VectorX<std::uint8_t> const &bytehammingweight, std::uint8_t byte,
    std::size_t chunk_start, std::size_t chunk_size) -> Eigen::MatrixXd {
  Eigen::MatrixXd chunk(traces.rows(), chunk_size + 1);
  chunk(Eigen::all, Eigen::seqN(0, chunk_size)) =
      traces(Eigen::all, Eigen::seqN(chunk_start, chunk_size));
  for (Eigen::Index i{0}; i != chunk.rows(); ++i) {
    chunk(i, chunk.cols() - 1) = bytehammingweight(subbytes(inputs(i) ^ byte));
  }
  return calculate_correlation_matrix(chunk);
}

static auto
correlation_byte(Eigen::VectorX<std::uint8_t> const &inputs,
                 Eigen::MatrixXd const &traces,
                 Eigen::VectorX<std::uint8_t> const &subbytes,
                 Eigen::VectorX<std::uint8_t> const &bytehammingweight,
                 std::uint8_t byte) -> Eigen::VectorXd {
  std::cout << +byte << '\n';
  Eigen::VectorXd vector(traces.cols());
  std::size_t chunk_size = 50;
  std::size_t chunks = traces.cols() / 50;
  for (std::size_t i{1}; i <= chunks; ++i) {
    std::size_t chunk_start = (i - 1) * chunk_size;
    Eigen::MatrixXd cmatrix =
        calculate_cmatrix(inputs, traces, subbytes, bytehammingweight, byte,
                          chunk_start, chunk_size);
    vector(Eigen::seqN(chunk_start, chunk_size)) =
        cmatrix(chunk_size, Eigen::seqN(0, chunk_size));
  }
  return vector;
}

static auto correlation(Eigen::VectorX<std::uint8_t> const &inputs,
                        Eigen::MatrixXd const &traces,
                        Eigen::VectorX<std::uint8_t> const &subbytes,
                        Eigen::VectorX<std::uint8_t> const &bytehammingweight)
    -> Eigen::MatrixXd {
  Eigen::MatrixXd key_traces(256, traces.cols());

  for (std::uint16_t b{0}; b < 256; b += 4) {
    // key_traces(b, Eigen::all) =
    auto future0 = std::async(std::launch::async, &correlation_byte, inputs,
                              traces, subbytes, bytehammingweight, b);
    auto future1 = std::async(std::launch::async, &correlation_byte, inputs,
                              traces, subbytes, bytehammingweight, b + 1);
    auto future2 = std::async(std::launch::async, &correlation_byte, inputs,
                              traces, subbytes, bytehammingweight, b + 2);
    auto future3 = std::async(std::launch::async, &correlation_byte, inputs,
                              traces, subbytes, bytehammingweight, b + 3);
    key_traces(b, Eigen::all) = future0.get();
    key_traces(b + 1, Eigen::all) = future1.get();
    key_traces(b + 2, Eigen::all) = future2.get();
    key_traces(b + 3, Eigen::all) = future3.get();
  }
  return key_traces;
}

auto main() -> int {
  Eigen::VectorX<std::uint8_t> inputs;
  {
    std::vector<std::uint8_t> tmp;
    if (!read_inputs("inputs.csv", tmp)) {
      std::cerr << "error reading inputs.csv\n";
      return EXIT_FAILURE;
    }

    inputs.resize(tmp.size());
    std::copy(tmp.begin(), tmp.end(), inputs.begin());
  }

  Eigen::MatrixXd traces;
  {
    std::vector<std::vector<double>> tmp;
    if (!read_traces("traces.csv", tmp)) {
      std::cerr << "error reading traces.csv\n";
      return EXIT_FAILURE;
    }

    traces.resize(tmp.size(), tmp[0].size());
    for (Eigen::Index i{0}; i != traces.rows(); ++i) {
      std::copy(tmp[i].begin(), tmp[i].end(), traces.row(i).begin());
    }
  }

  Eigen::VectorX<std::uint8_t> subbytes;
  {
    std::vector<std::uint8_t> tmp;
    if (!read_subbytes("subbytes.csv", tmp)) {
      std::cerr << "error reading subbytes.csv\n";
      return EXIT_FAILURE;
    }

    subbytes.resize(tmp.size());
    std::copy(tmp.begin(), tmp.end(), subbytes.begin());
  }

  if (true) {
    Eigen::VectorX<std::uint8_t> bytehammingweight;
    {
      std::vector<std::uint8_t> tmp;
      if (!read_bytehammingweight("bytehammingweight.csv", tmp)) {
        std::cerr << "error reading bytehammingweight.csv\n";
        return EXIT_FAILURE;
      }

      bytehammingweight.resize(tmp.size());
      std::copy(tmp.begin(), tmp.end(), bytehammingweight.begin());
    }

    Eigen::MatrixXd keys =
        correlation(inputs, traces, subbytes, bytehammingweight);
    if (!write_key_traces("key-traces.csv", keys)) {
      std::cerr << "error writing key traces\n";
      return EXIT_FAILURE;
    }
  } else {
    Eigen::MatrixXd keys = kocher(inputs, traces, subbytes);
    if (!write_key_traces("key-traces.csv", keys)) {
      std::cerr << "error writing key traces\n";
      return EXIT_FAILURE;
    }
  }
}
