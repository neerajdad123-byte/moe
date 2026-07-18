// MoEx clean-room. GPT-2 byte-level BPE tokenizer implementation.
#include "model/tokenizer.h"

#include <algorithm>
#include <regex>

namespace moex {

// GPT-2 byte-level BPE byte<->unicode table. The "printable" bytes map to
// themselves; the rest are remapped to codepoints 256+ so no control/space
// bytes appear literally in the vocab. This reproduces the standard
// bytes_to_unicode() mapping used by GPT-2 / Qwen tokenizers. Fills both
// directions so decode (codepoint->byte) and encode (byte->codepoint) share
// one definition.
static void build_byte_unicode(std::unordered_map<uint32_t, uint8_t>* u2b,
                               std::unordered_map<uint8_t, uint32_t>* b2u) {
  auto in_printable = [](int b) {
    return (b >= 33 && b <= 126) ||    // '!'..'~'
           (b >= 161 && b <= 172) ||   // ¡..¬
           (b >= 174 && b <= 255);     // ®..ÿ
  };
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    uint32_t cp;
    if (in_printable(b)) {
      cp = (uint32_t)b;  // codepoint == byte
    } else {
      cp = (uint32_t)(256 + n);  // remapped codepoint
      ++n;
    }
    if (u2b) (*u2b)[cp] = (uint8_t)b;
    if (b2u) (*b2u)[(uint8_t)b] = cp;
  }
}

// UTF-8 encode one codepoint (only needs the 1- and 2-byte forms: the
// byte-level-BPE alphabet only ever produces codepoints 0..255+~68, well
// under the 0x800 2-byte ceiling).
static std::string utf8_encode_cp(uint32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out.push_back((char)cp);
  } else {
    out.push_back((char)(0xC0 | (cp >> 6)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  }
  return out;
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
  build_byte_unicode(&u2b_, nullptr);
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

bool Encoder::load(const gguf::Model& m, std::string* err) {
  const gguf::Value* toks = m.find("tokenizer.ggml.tokens");
  if (!toks || toks->type != gguf::Type::ARRAY ||
      toks->array_type != gguf::Type::STRING) {
    if (err) *err = "tokenizer.ggml.tokens missing or not a string array";
    return false;
  }
  tok2id_.reserve(toks->array_strings.size());
  for (size_t i = 0; i < toks->array_strings.size(); ++i)
    tok2id_[toks->array_strings[i]] = (int)i;

  const gguf::Value* merges = m.find("tokenizer.ggml.merges");
  if (!merges || merges->type != gguf::Type::ARRAY ||
      merges->array_type != gguf::Type::STRING) {
    if (err) *err = "tokenizer.ggml.merges missing or not a string array";
    return false;
  }
  merge_rank_.reserve(merges->array_strings.size());
  for (size_t i = 0; i < merges->array_strings.size(); ++i) {
    const std::string& pair = merges->array_strings[i];
    size_t sp = pair.find(' ');
    if (sp == std::string::npos) continue;  // malformed entry, skip
    std::string key = pair.substr(0, sp) + '\0' + pair.substr(sp + 1);
    merge_rank_[key] = (int)i;  // lower index = higher merge priority
  }

  build_byte_unicode(nullptr, &b2u_);
  return true;
}

// GPT-2 pretokenizer regex, ASCII-scoped (see tokenizer.h file header):
//   contractions | ' '?letters | ' '?digits | ' '?other-non-space |
//   run-of-space-not-followed-by-nonspace | run-of-space
static const std::regex& gpt2_pretokenize_re() {
  static const std::regex re(
      "'s|'t|'re|'ve|'m|'ll|'d"
      "| ?[A-Za-z]+"
      "| ?[0-9]+"
      "| ?[^ \\t\\n\\r\\fA-Za-z0-9]+"
      "|[ \\t\\n\\r\\f]+(?![^ \\t\\n\\r\\f])"
      "|[ \\t\\n\\r\\f]+");
  return re;
}

std::vector<int> Encoder::encode(const std::string& text, std::string* err) const {
  std::vector<int> ids;
  auto begin = std::sregex_iterator(text.begin(), text.end(), gpt2_pretokenize_re());
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    const std::string chunk = it->str();
    if (chunk.empty()) continue;

    // Byte-level map: each raw byte -> its UTF-8-encoded unicode codepoint.
    std::vector<std::string> symbols;
    symbols.reserve(chunk.size());
    for (unsigned char c : chunk) symbols.push_back(utf8_encode_cp(b2u_.at(c)));

    // Greedy BPE: repeatedly merge the adjacent pair with the lowest rank.
    while (symbols.size() > 1) {
      int best_rank = -1;
      size_t best_i = 0;
      for (size_t i = 0; i + 1 < symbols.size(); ++i) {
        auto k = symbols[i] + '\0' + symbols[i + 1];
        auto found = merge_rank_.find(k);
        if (found != merge_rank_.end() &&
            (best_rank < 0 || found->second < best_rank)) {
          best_rank = found->second;
          best_i = i;
        }
      }
      if (best_rank < 0) break;  // no mergeable pair left
      symbols[best_i] += symbols[best_i + 1];
      symbols.erase(symbols.begin() + (best_i + 1));
    }

    for (const auto& s : symbols) {
      auto found = tok2id_.find(s);
      if (found == tok2id_.end()) {
        if (err) *err = "out-of-vocab symbol during encode: " + s;
        continue;
      }
      ids.push_back(found->second);
    }
  }
  return ids;
}

}  // namespace moex
