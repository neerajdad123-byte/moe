// MoEx tool: parse a GGUF file with the clean-room reader and print a summary.
// This is the first end-to-end proof that the reader works on the real blob.
#include <cstdio>
#include <map>
#include <string>

#include "gguf/gguf.h"

using namespace moex;

static void print_scalar(const gguf::Value& v) {
  switch (v.type) {
    case gguf::Type::STRING: std::printf("\"%s\"", v.str.c_str()); break;
    case gguf::Type::F32:
    case gguf::Type::F64:    std::printf("%g", v.as_f64()); break;
    case gguf::Type::BOOL:   std::printf("%s", v.as_u64() ? "true" : "false"); break;
    case gguf::Type::ARRAY:
      std::printf("[array len=%llu]", (unsigned long long)v.array_len);
      break;
    default:
      if (v.is_int()) std::printf("%lld", (long long)v.as_i64());
      else std::printf("<?>");
      break;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: gguf_dump <file.gguf> [--tensors]\n");
    return 2;
  }
  const std::string path = argv[1];
  bool show_tensors = argc > 2 && std::string(argv[2]) == "--tensors";

  gguf::Model m;
  std::string err;
  if (!m.open(path, &err)) {
    std::fprintf(stderr, "parse failed: %s\n", err.c_str());
    return 1;
  }

  std::printf("file        : %s\n", path.c_str());
  std::printf("size        : %.2f GiB\n", m.file_size() / (1024.0 * 1024 * 1024));
  std::printf("gguf version: %u\n", m.version());
  std::printf("tensors     : %llu\n", (unsigned long long)m.tensor_count());
  std::printf("kv pairs    : %llu\n", (unsigned long long)m.metadata().size());
  std::printf("data start  : %llu\n", (unsigned long long)m.data_start());

  std::printf("\n--- metadata ---\n");
  for (const auto& [k, v] : m.metadata()) {
    std::printf("  %-40s = ", k.c_str());
    print_scalar(v);
    std::printf("\n");
  }

  // Type histogram across all tensors.
  std::printf("\n--- tensor quant histogram ---\n");
  std::map<std::string, int> hist;
  for (const auto& t : m.tensors()) hist[gguf::ggml_type_name(t.type)]++;
  for (const auto& [name, n] : hist) std::printf("  %-8s %d\n", name.c_str(), n);

  if (show_tensors) {
    std::printf("\n--- tensors ---\n");
    for (const auto& t : m.tensors()) {
      std::printf("  %-34s [", t.name.c_str());
      for (size_t i = 0; i < t.dims.size(); ++i)
        std::printf("%s%llu", i ? "," : "", (unsigned long long)t.dims[i]);
      std::printf("] %-6s off=%llu bytes=%llu\n", gguf::ggml_type_name(t.type),
                  (unsigned long long)t.file_offset, (unsigned long long)t.nbytes);
    }
  }
  return 0;
}
