// MoEx clean-room. GPT-2 byte-level BPE detokenizer for Qwen3.
//
// v1 scope: DECODE (id -> text). The vocab strings come straight from the GGUF
// `tokenizer.ggml.tokens` array. GPT-2 byte-level BPE maps each raw byte to a
// printable Unicode codepoint so the vocab has no control bytes; detokenizing
// reverses that map. Encoding (text -> ids) is out of scope for the correctness
// gate — prompt ids are provided externally.
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

}  // namespace moex
