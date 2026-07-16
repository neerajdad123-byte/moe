// MoEx clean-room. Read-only memory-mapped file (Windows). Lets us parse the
// GGUF header/tensor table of an 18 GB blob while touching only the pages we
// actually read. Non-throwing: open() reports failure through an out-param so
// the loader can classify the error.
#pragma once

#include <cstdint>
#include <string>

namespace moex {

class MmapFile {
 public:
  MmapFile() = default;
  ~MmapFile();

  MmapFile(const MmapFile&) = delete;
  MmapFile& operator=(const MmapFile&) = delete;
  MmapFile(MmapFile&& o) noexcept;
  MmapFile& operator=(MmapFile&& o) noexcept;

  // Map `path` read-only. Returns false and fills *err on failure.
  bool open(const std::string& path, std::string* err);

  const uint8_t* data() const { return data_; }
  uint64_t size() const { return size_; }
  bool is_open() const { return data_ != nullptr; }

 private:
  void close_();
  void* file_ = nullptr;     // HANDLE
  void* mapping_ = nullptr;  // HANDLE
  const uint8_t* data_ = nullptr;
  uint64_t size_ = 0;
};

}  // namespace moex
