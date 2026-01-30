#include "include/tokenizer.h"

#include <fstream>

namespace nemo {

// SentencePiece uses ▁ (U+2581) as word start marker
// In UTF-8, this is 0xE2 0x96 0x81
static const char SPIECE_UNDERLINE[] = "\xE2\x96\x81";
static const size_t SPIECE_UNDERLINE_LEN = 3;

bool Tokenizer::load(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        return false;
    }

    vocab_.clear();
    std::string line;
    while (std::getline(file, line)) {
        vocab_.push_back(line);
    }

    return !vocab_.empty();
}

std::string Tokenizer::decode_token(int token_id) const {
    if (token_id == BLANK_TOKEN) {
        return "";  // Blank token
    }
    if (token_id < 0 || (size_t)token_id >= vocab_.size()) {
        return "<unk>";
    }
    return vocab_[token_id];
}

bool Tokenizer::is_word_start(int token_id) const {
    if (token_id < 0 || (size_t)token_id >= vocab_.size()) {
        return false;
    }
    const std::string& token = vocab_[token_id];
    // Check if token starts with ▁ (SentencePiece word start marker)
    return token.size() >= SPIECE_UNDERLINE_LEN &&
           token.compare(0, SPIECE_UNDERLINE_LEN, SPIECE_UNDERLINE) == 0;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;

    for (int token_id : tokens) {
        if (token_id == BLANK_TOKEN) {
            continue;  // Skip blank tokens
        }

        std::string token = decode_token(token_id);

        if (is_word_start(token_id)) {
            // Token starts with ▁ - add space and remove the marker
            if (!result.empty()) {
                result += ' ';
            }
            result += token.substr(SPIECE_UNDERLINE_LEN);
        } else {
            // Continuation token - append directly (no space)
            result += token;
        }
    }

    return result;
}

}  // namespace nemo
