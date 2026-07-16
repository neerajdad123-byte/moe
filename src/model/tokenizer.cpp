// MoEx clean-room. GPT-2 byte-level BPE detokenizer implementation.
#include "model/tokenizer.h"

namespace moex {

// GPT-2 byte-level BPE byte<->unicode table. The "printable" bytes map to
// themselves; the rest are remapped to codepoints 256+ so no control/space
// bytes appear literally in the vocab. This reproduces the standard
// bytes_to_unicode() mapping used by GPT-2 / Qwen tokenizers.
static void build_byte_unicode(std::unordered_map<uint32_t, uint8_t>* u2b) {
  auto in_printable = [](int b) {
    return (b >= 33 && b <= 126) ||    // '!'..'~'
           (b >= 161 && b <= 172) ||   // ¡..¬
           (b >= 174 && b <= 255);     // ®..ÿ
  };
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (in_printable(b)) {
      (*u2b)[(uint32_t)b] = (uint8_t)b;  // codepoint == byte
    } else {
      (*u2b)[(uint32_t)(256 + n)] = (uint8_t)b;  // remapped codepoint
      ++n;
    }
  }
}

// Decode one UTF-8 codepoint from s at pos; advance pos. Returns codepoint.
static uint32_t next_cp(const std::string& s, size_t* pos) {
  uint8_t c = (uint8_t)s[*pos];
  uint32_t cp;
  int extra;
  if (c < 0x80) { cp = c; extra = 0; }
  else if ((c >> 5) == 0x6) { cp = c & 0x1f; extra = 1; }
  else if ((c >> 4) == 0xe) { cp = c & 0x0f; extra = 2; }
  else if ((c >> 3) == 0x1e) { cp = c & 0x07; extra = 3; }
  else { cp = c; extra = 0; }  // invalid lead: pass through
  ++(*pos);
  for (int k = 0; k < extra && *pos < s.size(); ++k) {
    cp = (cp << 6) | ((uint8_t)s[*pos] & 0x3f);
    ++(*pos);
  }
  return cp;
}

bool Detokenizer::load(const gguf::Model& m, std::string* err) {
  const gguf::Value* toks = m.find("tokenizer.ggml.tokens");
  if (!toks || toks->type != gguf::Type::ARRAY ||
      toks->array_type != gguf::Type::STRING) {
    if (err) *err = "tokenizer.ggml.tokens missing or not a string array";
    return false;
  }
  tokens_ = toks->array_strings;
  build_byte_unicode(&u2b_);
  bos_id_ = (int)m.get_u64("tokenizer.ggml.bos_token_id", 151643);
  eos_id_ = (int)m.get_u64("tokenizer.ggml.eos_token_id", 151645);
  return true;
}

std::string Detokenizer::decode_id(int id) const {
  if (id < 0 || id >= (int)tokens_.size()) return "";
  const std::string& t = tokens_[id];
  // Reverse the byte-level map: each unicode codepoint -> one raw byte.
  std::string out;
  size_t pos = 0;
  while (pos < t.size()) {
    uint32_t cp = next_cp(t, &pos);
    auto it = u2b_.find(cp);
    if (it != u2b_.end()) out.push_back((char)it->second);
    else out.push_back('?');  // shouldn't happen for well-formed vocab
  }
  return out;
}

std::string Detokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  for (int id : ids) out += decode_id(id);
  return out;
}

}  // namespace moex
