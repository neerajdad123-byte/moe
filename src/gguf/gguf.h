// MoEx clean-room GGUF v3 reader.
//
// Reimplements the public GGUF container format from its published
// specification. No ggml/llama.cpp source, headers, or code are used.
// Format reference (studied, not copied): the GGUF format is a header of
//   magic "GGUF" | u32 version | u64 tensor_count | u64 metadata_kv_count
// followed by metadata key/value pairs, then a tensor-info table, then a
// padded blob of tensor data.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "support/bytes.h"
#include "support/mmap_file.h"

namespace moex::gguf {

using moex::BinReader;
using moex::MmapFile;

// GGUF metadata value types (wire-format enum, GGUF v3).
enum class Type : uint32_t {
  U8 = 0,
  I8 = 1,
  U16 = 2,
  I16 = 3,
  U32 = 4,
  I32 = 5,
  F32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  U64 = 10,
  I64 = 11,
  F64 = 12,
};

// GGML tensor element types we need to recognize for Qwen3 Q4_K_M.
// Values match the published GGUF type enumeration. We only *decode* the
// subset the target model actually uses; anything else is rejected at load.
enum class GgmlType : uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  UNKNOWN = 0xffffffff,
};

const char* ggml_type_name(GgmlType t);

// Block byte size and element-per-block for a GGML type. Returns false for
// types MoEx does not support decoding.
bool ggml_type_block(GgmlType t, uint64_t* block_bytes, uint64_t* block_elems);

// A single metadata value. Scalars live in their typed field; strings in
// `str`; arrays keep their element type plus raw counts so callers can pull
// typed elements without re-parsing.
struct Value {
  Type type = Type::U32;
  // Scalars (only the field matching `type` is meaningful).
  uint64_t u = 0;   // holds U8/U16/U32/U64/BOOL
  int64_t i = 0;    // holds I8/I16/I32/I64
  double f = 0.0;   // holds F32/F64
  std::string str;  // holds STRING

  // Arrays.
  Type array_type = Type::U32;
  uint64_t array_len = 0;
  // For arrays of strings we keep the decoded strings; for numeric arrays we
  // keep the raw little-endian bytes plus per-element stride.
  std::vector<std::string> array_strings;
  std::vector<uint8_t> array_raw;
  uint64_t array_elem_stride = 0;

  bool is_int() const {
    return type == Type::U8 || type == Type::U16 || type == Type::U32 ||
           type == Type::U64 || type == Type::I8 || type == Type::I16 ||
           type == Type::I32 || type == Type::I64 || type == Type::BOOL;
  }
  // Best-effort unsigned view of any integer scalar.
  uint64_t as_u64() const;
  int64_t as_i64() const;
  double as_f64() const;
};

// One tensor's location and shape within the file.
struct TensorInfo {
  std::string name;
  std::vector<uint64_t> dims;  // GGUF stores fastest-varying dim first
  GgmlType type = GgmlType::UNKNOWN;
  uint64_t rel_offset = 0;   // offset from start of the data blob
  uint64_t file_offset = 0;  // absolute offset in the file (rel + data_start)
  uint64_t nbytes = 0;       // computed byte length of the tensor payload

  uint64_t elem_count() const;
};

// A parsed GGUF file. Owns nothing large: metadata + a tensor table plus the
// absolute offset where the data blob starts. Tensor payloads stay on disk.
class Model {
 public:
  // Parse the header/metadata/tensor-table region of `path`. The data blob is
  // NOT read. On failure returns false and fills `err`.
  bool open(const std::string& path, std::string* err);

  uint32_t version() const { return version_; }
  uint64_t tensor_count() const { return tensor_count_; }
  uint64_t data_start() const { return data_start_; }
  const std::string& path() const { return path_; }
  uint64_t file_size() const { return file_.size(); }

  // Base pointer of the mapped file. Add a tensor's file_offset to read its
  // raw (still-quantized) payload directly from the mapping.
  const uint8_t* base() const { return file_.data(); }

  const std::map<std::string, Value>& metadata() const { return kv_; }
  const std::vector<TensorInfo>& tensors() const { return tensors_; }

  // Metadata lookups. Return nullptr / default when absent.
  const Value* find(const std::string& key) const;
  bool has(const std::string& key) const { return find(key) != nullptr; }
  uint64_t get_u64(const std::string& key, uint64_t def) const;
  int64_t get_i64(const std::string& key, int64_t def) const;
  double get_f64(const std::string& key, double def) const;
  std::string get_str(const std::string& key, const std::string& def) const;

  // Tensor lookup by exact name.
  const TensorInfo* tensor(const std::string& name) const;

 private:
  bool parse_metadata(BinReader& r, std::string* err);
  bool parse_tensor_table(BinReader& r, std::string* err);
  bool read_value(BinReader& r, Value* v, std::string* err, int depth);

  std::string path_;
  MmapFile file_;
  uint32_t version_ = 0;
  uint64_t tensor_count_ = 0;
  uint64_t kv_count_ = 0;
  uint64_t data_start_ = 0;
  uint32_t alignment_ = 32;  // default GGUF alignment; overridden by metadata

  std::map<std::string, Value> kv_;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, size_t> tensor_index_;
};

}  // namespace moex::gguf
