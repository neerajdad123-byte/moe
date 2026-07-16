#include "gguf/gguf.h"

#include <cstring>

namespace moex::gguf {

const char* ggml_type_name(GgmlType t) {
  switch (t) {
    case GgmlType::F32: return "F32";
    case GgmlType::F16: return "F16";
    case GgmlType::Q4_0: return "Q4_0";
    case GgmlType::Q4_1: return "Q4_1";
    case GgmlType::Q5_0: return "Q5_0";
    case GgmlType::Q5_1: return "Q5_1";
    case GgmlType::Q8_0: return "Q8_0";
    case GgmlType::Q8_1: return "Q8_1";
    case GgmlType::Q2_K: return "Q2_K";
    case GgmlType::Q3_K: return "Q3_K";
    case GgmlType::Q4_K: return "Q4_K";
    case GgmlType::Q5_K: return "Q5_K";
    case GgmlType::Q6_K: return "Q6_K";
    case GgmlType::Q8_K: return "Q8_K";
    default: return "UNKNOWN";
  }
}

bool ggml_type_block(GgmlType t, uint64_t* block_bytes, uint64_t* block_elems) {
  // K-quant blocks all span 256 elements (a "super-block"). Byte sizes below
  // are the on-disk block sizes fixed by the GGUF quant layout. These are the
  // published constants for the format; the decode math lives in the kernel
  // layer, gated by the NumericalConformanceContract.
  uint64_t bb = 0, be = 0;
  switch (t) {
    case GgmlType::F32:  bb = 4;   be = 1;   break;
    case GgmlType::F16:  bb = 2;   be = 1;   break;
    case GgmlType::Q8_0: bb = 34;  be = 32;  break;  // 32*i8 + f16 scale
    case GgmlType::Q4_0: bb = 18;  be = 32;  break;
    case GgmlType::Q4_1: bb = 20;  be = 32;  break;
    case GgmlType::Q5_0: bb = 22;  be = 32;  break;
    case GgmlType::Q5_1: bb = 24;  be = 32;  break;
    case GgmlType::Q2_K: bb = 84;  be = 256; break;
    case GgmlType::Q3_K: bb = 110; be = 256; break;
    case GgmlType::Q4_K: bb = 144; be = 256; break;
    case GgmlType::Q5_K: bb = 176; be = 256; break;
    case GgmlType::Q6_K: bb = 210; be = 256; break;
    case GgmlType::Q8_K: bb = 292; be = 256; break;
    default: return false;
  }
  if (block_bytes) *block_bytes = bb;
  if (block_elems) *block_elems = be;
  return true;
}

uint64_t Value::as_u64() const {
  switch (type) {
    case Type::I8: case Type::I16: case Type::I32: case Type::I64:
      return static_cast<uint64_t>(i);
    case Type::F32: case Type::F64:
      return static_cast<uint64_t>(f);
    default:
      return u;
  }
}

int64_t Value::as_i64() const {
  switch (type) {
    case Type::U8: case Type::U16: case Type::U32: case Type::U64:
    case Type::BOOL:
      return static_cast<int64_t>(u);
    case Type::F32: case Type::F64:
      return static_cast<int64_t>(f);
    default:
      return i;
  }
}

double Value::as_f64() const {
  switch (type) {
    case Type::F32: case Type::F64:
      return f;
    case Type::U8: case Type::U16: case Type::U32: case Type::U64:
    case Type::BOOL:
      return static_cast<double>(u);
    default:
      return static_cast<double>(i);
  }
}

uint64_t TensorInfo::elem_count() const {
  uint64_t n = 1;
  for (uint64_t d : dims) n *= d;
  return n;
}

// --- scalar byte width for GGUF metadata primitive types ------------------
static bool scalar_width(Type t, uint64_t* w) {
  switch (t) {
    case Type::U8: case Type::I8: case Type::BOOL: *w = 1; return true;
    case Type::U16: case Type::I16: *w = 2; return true;
    case Type::U32: case Type::I32: case Type::F32: *w = 4; return true;
    case Type::U64: case Type::I64: case Type::F64: *w = 8; return true;
    default: return false;  // STRING/ARRAY handled separately
  }
}

// Read a GGUF string: u64 length + raw bytes (no NUL terminator on disk).
static bool read_gguf_string(BinReader& r, std::string* out, std::string* err) {
  uint64_t len;
  if (!r.u64(&len)) { if (err) *err = "truncated string length"; return false; }
  // Guard against absurd lengths from a corrupt/hostile file.
  if (len > (1ull << 30)) { if (err) *err = "string length too large"; return false; }
  const uint8_t* p;
  if (!r.take(len, &p)) { if (err) *err = "truncated string body"; return false; }
  out->assign(reinterpret_cast<const char*>(p), len);
  return true;
}

bool Model::read_value(BinReader& r, Value* v, std::string* err, int depth) {
  if (depth > 2) { if (err) *err = "metadata nesting too deep"; return false; }
  uint32_t raw_type;
  if (!r.u32(&raw_type)) { if (err) *err = "truncated value type"; return false; }
  v->type = static_cast<Type>(raw_type);

  switch (v->type) {
    case Type::U8:  { uint8_t x;  if (!r.u8(&x))  return false; v->u = x; return true; }
    case Type::I8:  { int8_t x;   if (!r.i8(&x))  return false; v->i = x; return true; }
    case Type::U16: { uint16_t x; if (!r.u16(&x)) return false; v->u = x; return true; }
    case Type::I16: { int16_t x;  if (!r.i16(&x)) return false; v->i = x; return true; }
    case Type::U32: { uint32_t x; if (!r.u32(&x)) return false; v->u = x; return true; }
    case Type::I32: { int32_t x;  if (!r.i32(&x)) return false; v->i = x; return true; }
    case Type::F32: { float x;    if (!r.f32(&x)) return false; v->f = x; return true; }
    case Type::BOOL:{ uint8_t x;  if (!r.u8(&x))  return false; v->u = x ? 1 : 0; return true; }
    case Type::U64: { uint64_t x; if (!r.u64(&x)) return false; v->u = x; return true; }
    case Type::I64: { int64_t x;  if (!r.i64(&x)) return false; v->i = x; return true; }
    case Type::F64: { double x;   if (!r.f64(&x)) return false; v->f = x; return true; }
    case Type::STRING:
      return read_gguf_string(r, &v->str, err);
    case Type::ARRAY: {
      uint32_t at;
      if (!r.u32(&at)) { if (err) *err = "truncated array type"; return false; }
      v->array_type = static_cast<Type>(at);
      if (!r.u64(&v->array_len)) { if (err) *err = "truncated array len"; return false; }
      if (v->array_type == Type::STRING) {
        v->array_strings.reserve(v->array_len < 4096 ? v->array_len : 4096);
        for (uint64_t k = 0; k < v->array_len; ++k) {
          std::string s;
          if (!read_gguf_string(r, &s, err)) return false;
          v->array_strings.push_back(std::move(s));
        }
        return true;
      }
      if (v->array_type == Type::ARRAY) {
        if (err) *err = "nested arrays unsupported";
        return false;
      }
      uint64_t w;
      if (!scalar_width(v->array_type, &w)) {
        if (err) *err = "bad array element type";
        return false;
      }
      v->array_elem_stride = w;
      uint64_t total = w * v->array_len;
      const uint8_t* p;
      if (!r.take(total, &p)) { if (err) *err = "truncated array body"; return false; }
      v->array_raw.assign(p, p + total);
      return true;
    }
    default:
      if (err) *err = "unknown metadata value type " + std::to_string(raw_type);
      return false;
  }
}

bool Model::parse_metadata(BinReader& r, std::string* err) {
  for (uint64_t k = 0; k < kv_count_; ++k) {
    std::string key;
    if (!read_gguf_string(r, &key, err)) return false;
    Value v;
    if (!read_value(r, &v, err, 0)) {
      if (err) *err = "kv '" + key + "': " + *err;
      return false;
    }
    kv_.emplace(std::move(key), std::move(v));
  }
  // GGUF may declare a custom alignment; honor it for the data-blob offset.
  auto it = kv_.find("general.alignment");
  if (it != kv_.end() && it->second.is_int()) {
    uint64_t a = it->second.as_u64();
    if (a >= 1 && (a & (a - 1)) == 0 && a <= 65536) alignment_ = static_cast<uint32_t>(a);
  }
  return true;
}

bool Model::parse_tensor_table(BinReader& r, std::string* err) {
  tensors_.reserve(tensor_count_ < 100000 ? tensor_count_ : 100000);
  uint64_t max_rel_end = 0;
  for (uint64_t t = 0; t < tensor_count_; ++t) {
    TensorInfo ti;
    if (!read_gguf_string(r, &ti.name, err)) return false;
    uint32_t ndim;
    if (!r.u32(&ndim)) { if (err) *err = "truncated tensor ndim"; return false; }
    if (ndim > 4) { if (err) *err = "tensor '" + ti.name + "' has ndim>4"; return false; }
    ti.dims.resize(ndim);
    for (uint32_t d = 0; d < ndim; ++d) {
      if (!r.u64(&ti.dims[d])) { if (err) *err = "truncated tensor dim"; return false; }
    }
    uint32_t raw_tt;
    if (!r.u32(&raw_tt)) { if (err) *err = "truncated tensor type"; return false; }
    ti.type = static_cast<GgmlType>(raw_tt);
    if (!r.u64(&ti.rel_offset)) { if (err) *err = "truncated tensor offset"; return false; }

    // Compute byte length from the quant block geometry when we know the type.
    uint64_t bb = 0, be = 0;
    if (ggml_type_block(ti.type, &bb, &be)) {
      uint64_t elems = ti.elem_count();
      if (be == 0 || (elems % be) != 0) {
        // Non-block-aligned element count: only valid for be==1 types.
        if (be != 1) {
          if (err) *err = "tensor '" + ti.name + "' element count not block-aligned";
          return false;
        }
      }
      ti.nbytes = (elems / be) * bb;
    } else {
      // Unknown type: record 0 and let the manifest layer flag it. We still
      // parse so diagnostics can report exactly which type is unsupported.
      ti.nbytes = 0;
    }
    uint64_t rel_end = ti.rel_offset + ti.nbytes;
    if (rel_end > max_rel_end) max_rel_end = rel_end;
    tensor_index_[ti.name] = tensors_.size();
    tensors_.push_back(std::move(ti));
  }

  // The data blob starts at the current cursor, padded up to `alignment_`.
  uint64_t here = r.pos();
  uint64_t pad = (alignment_ - (here % alignment_)) % alignment_;
  data_start_ = here + pad;

  // Fill absolute offsets and sanity-check the blob fits in the file.
  for (auto& ti : tensors_) ti.file_offset = data_start_ + ti.rel_offset;
  if (data_start_ + max_rel_end > file_.size()) {
    if (err) {
      *err = "tensor data blob exceeds file size (need " +
             std::to_string(data_start_ + max_rel_end) + " have " +
             std::to_string(file_.size()) + ")";
    }
    return false;
  }
  return true;
}

bool Model::open(const std::string& path, std::string* err) {
  path_ = path;
  if (!file_.open(path, err)) return false;

  BinReader r(file_.data(), file_.size());
  const uint8_t* magic;
  if (!r.take(4, &magic)) { if (err) *err = "file too small for magic"; return false; }
  if (std::memcmp(magic, "GGUF", 4) != 0) {
    if (err) *err = "not a GGUF file (bad magic)";
    return false;
  }
  if (!r.u32(&version_)) { if (err) *err = "truncated version"; return false; }
  if (version_ != 2 && version_ != 3) {
    if (err) *err = "unsupported GGUF version " + std::to_string(version_);
    return false;
  }
  if (!r.u64(&tensor_count_)) { if (err) *err = "truncated tensor count"; return false; }
  if (!r.u64(&kv_count_)) { if (err) *err = "truncated kv count"; return false; }
  if (!parse_metadata(r, err)) return false;
  if (!parse_tensor_table(r, err)) return false;
  return true;
}

const Value* Model::find(const std::string& key) const {
  auto it = kv_.find(key);
  return it == kv_.end() ? nullptr : &it->second;
}

uint64_t Model::get_u64(const std::string& key, uint64_t def) const {
  auto* v = find(key);
  return v && v->is_int() ? v->as_u64() : def;
}
int64_t Model::get_i64(const std::string& key, int64_t def) const {
  auto* v = find(key);
  return v && v->is_int() ? v->as_i64() : def;
}
double Model::get_f64(const std::string& key, double def) const {
  auto* v = find(key);
  return v && (v->is_int() || v->type == Type::F32 || v->type == Type::F64)
             ? v->as_f64() : def;
}
std::string Model::get_str(const std::string& key, const std::string& def) const {
  auto* v = find(key);
  return v && v->type == Type::STRING ? v->str : def;
}

const TensorInfo* Model::tensor(const std::string& name) const {
  auto it = tensor_index_.find(name);
  return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

}  // namespace moex::gguf
