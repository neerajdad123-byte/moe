// MoEx clean-room. Bounded little-endian byte cursor for parsing untrusted files.
//
// BinReader is a read-only, bounds-checked cursor over a byte span it does not
// own. Every accessor returns false on a short/out-of-bounds read instead of
// throwing, so the GGUF parser can turn any malformed input into a classified
// error rather than a crash. GGUF is little-endian; the x86-64 target is too,
// so scalars are a straight memcpy.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

namespace moex {

class BinReader {
 public:
  BinReader(const uint8_t* data, uint64_t size) : data_(data), size_(size) {}

  uint64_t pos() const { return pos_; }
  uint64_t size() const { return size_; }
  uint64_t remaining() const { return size_ - pos_; }

  // Move the cursor to an absolute offset. Returns false if past end.
  bool seek(uint64_t abs) {
    if (abs > size_) return false;
    pos_ = abs;
    return true;
  }

  // Borrow `n` bytes at the cursor without copying; advances past them.
  // On success sets *out to a pointer into the mapped buffer.
  bool take(uint64_t n, const uint8_t** out) {
    if (n > size_ - pos_) return false;
    *out = data_ + pos_;
    pos_ += n;
    return true;
  }

  bool skip(uint64_t n) {
    if (n > size_ - pos_) return false;
    pos_ += n;
    return true;
  }

  template <typename T>
  bool scalar(T* out) {
    static_assert(std::is_trivially_copyable_v<T>, "scalar must be trivially copyable");
    if (sizeof(T) > size_ - pos_) return false;
    std::memcpy(out, data_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return true;
  }

  bool u8(uint8_t* v) { return scalar(v); }
  bool i8(int8_t* v) { return scalar(v); }
  bool u16(uint16_t* v) { return scalar(v); }
  bool i16(int16_t* v) { return scalar(v); }
  bool u32(uint32_t* v) { return scalar(v); }
  bool i32(int32_t* v) { return scalar(v); }
  bool u64(uint64_t* v) { return scalar(v); }
  bool i64(int64_t* v) { return scalar(v); }
  bool f32(float* v) { return scalar(v); }
  bool f64(double* v) { return scalar(v); }

 private:
  const uint8_t* data_ = nullptr;
  uint64_t size_ = 0;
  uint64_t pos_ = 0;
};

}  // namespace moex
