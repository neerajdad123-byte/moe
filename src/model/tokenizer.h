// MoEx clean-room. GPT-2 byte-level BPE tokenizer for Qwen3.
//
// Decode (id -> text): vocab strings come straight from the GGUF
// `tokenizer.ggml.tokens` array. GPT-2 byte-level BPE maps each raw byte to a
// printable Unicode codepoint so the vocab has no control bytes; detokenizing
// reverses that map. This remains the correctness-gate path.
//
// Encode (text -> ids): `Encoder` below is a benchmarking/data-collection
// utility, not part of the numerical conformance gate. It implements
// standard GPT-2 byte-level BPE (pretokenize, byte->unicode map, greedy
// merge by rank from `tokenizer.ggml.merges`, vocab lookup). The
// pretokenizer regex is ASCII-scoped (`[A-Za-z]+`/`[0-9]+` in place of
// `\p{L}+`/`\p{N}+`, since std::regex has no Unicode property classes
// without ICU) — correct for plain ASCII prompt text, not general UTF-8.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf/gguf.h"

namespace moex {

class Detokenizer {
 public:
  // Load vocab strings from the parsed GGUF. Returns false if the tokens array
  // is missing.
  bool load(const gguf::Model& m, std::string* err);

  // Decode a single token id to its raw byte string (may be a partial UTF-8
  // sequence; concatenation across ids yields valid UTF-8).
  std::string decode_id(int id) const;

  // Decode a sequence of ids to a UTF-8 string.
  std::string decode(const std::vector<int>& ids) const;

  size_t vocab_size() const { return tokens_.size(); }
  int eos_id() const { return eos_id_; }
  int bos_id() const { return bos_id_; }

 private:
  std::vector<std::string> tokens_;        // id -> byte-level-BPE string
  std::unordered_map<uint32_t, uint8_t> u2b_;  // unicode codepoint -> raw byte
  int eos_id_ = -1;
  int bos_id_ = -1;
};

// Text -> token ids (GPT-2 byte-level BPE encode). See file header for scope.
class Encoder {
 public:
  // Loads vocab (tokenizer.ggml.tokens) and merge ranks (tokenizer.ggml.merges).
  bool load(const gguf::Model& m, std::string* err);

  // Encode UTF-8 text (ASCII-scoped pretokenizer, see file header) to ids.
  // On an out-of-vocab symbol (should not happen against a matching model),
  // the symbol is skipped and `err` is set if provided; encoding continues.
  std::vector<int> encode(const std::string& text, std::string* err = nullptr) const;

 private:
  std::unordered_map<std::string, int> tok2id_;
  std::unordered_map<uint8_t, uint32_t> b2u_;       // raw byte -> unicode codepoint
  std::unordered_map<std::string, int> merge_rank_;  // "left\0right" -> priority
};

}  // namespace moex
