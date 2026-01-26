#include "tokenizer.h"

#include <cstdio>

using namespace nemo;

void test_load() {
    printf("\n=== Testing Tokenizer Load ===\n");

    Tokenizer tokenizer;
    if (!tokenizer.load("vocab.txt")) {
        printf("FAIL: Could not load vocab.txt\n");
        return;
    }

    printf("Loaded %zu tokens\n", tokenizer.size());

    // Print first 10 tokens
    printf("First 10 tokens:\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d: '%s'\n", i, tokenizer.decode_token(i).c_str());
    }

    printf("OK: Tokenizer load\n");
}

void test_decode_single() {
    printf("\n=== Testing Single Token Decode ===\n");

    Tokenizer tokenizer;
    if (!tokenizer.load("vocab.txt")) {
        printf("FAIL: Could not load vocab.txt\n");
        return;
    }

    // Test blank token
    std::string blank = tokenizer.decode_token(Tokenizer::BLANK_TOKEN);
    if (!blank.empty()) {
        printf("FAIL: Blank token should decode to empty string\n");
        return;
    }
    printf("Blank token (1024): '%s' OK\n", blank.c_str());

    // Test some known tokens (based on vocab file)
    printf("Token 0: '%s'\n", tokenizer.decode_token(0).c_str());  // 't'
    printf("Token 1: '%s'\n", tokenizer.decode_token(1).c_str());  // 'th'
    printf("Token 4: '%s'\n", tokenizer.decode_token(4).c_str());  // 'the'
    printf("Token 32: '%s'\n", tokenizer.decode_token(32).c_str()); // 'and'

    // Test out of range
    std::string unk = tokenizer.decode_token(9999);
    printf("Token 9999 (out of range): '%s'\n", unk.c_str());

    printf("OK: Single token decode\n");
}

void test_decode_sequence() {
    printf("\n=== Testing Sequence Decode ===\n");

    Tokenizer tokenizer;
    if (!tokenizer.load("vocab.txt")) {
        printf("FAIL: Could not load vocab.txt\n");
        return;
    }

    // Find some useful token indices by looking at vocab
    // These are approximate based on the vocab file content
    // Token 4 = 'the', Token 0 = 't'

    // Test simple sequence (with continuation tokens)
    // "the" = token 4
    std::vector<int> tokens1 = {4};  // "the"
    std::string result1 = tokenizer.decode(tokens1);
    printf("Tokens [4]: '%s'\n", result1.c_str());

    // Test with blank tokens interspersed
    std::vector<int> tokens2 = {1024, 4, 1024, 1024};  // blanks + "the"
    std::string result2 = tokenizer.decode(tokens2);
    printf("Tokens [blank, 4, blank, blank]: '%s'\n", result2.c_str());

    // Test continuation handling: "t" + "##h" should be "th"
    // Token 0 = 't', but we need to find ##h...
    // Looking at vocab: ##in, ##re, ##er, ##at, ##ou, ##nd, ##it, ##is, etc.
    // Let's test with tokens that we know exist

    // Create a longer example with multiple words
    // 4 = "the", 32 = "and", 28 = "of"
    std::vector<int> tokens3 = {4, 32, 28};
    std::string result3 = tokenizer.decode(tokens3);
    printf("Tokens [4, 32, 28] ('the', 'and', 'of'): '%s'\n", result3.c_str());

    printf("OK: Sequence decode\n");
}

void test_word_start() {
    printf("\n=== Testing Word Start Detection ===\n");

    Tokenizer tokenizer;
    if (!tokenizer.load("vocab.txt")) {
        printf("FAIL: Could not load vocab.txt\n");
        return;
    }

    // Token 5 should be "▁the" (word start)
    bool ws5 = tokenizer.is_word_start(5);
    printf("Token 5 ('%s') is_word_start: %s\n",
           tokenizer.decode_token(5).c_str(),
           ws5 ? "true" : "false");

    // Token 4 should be "in" (not word start)
    bool ws4 = tokenizer.is_word_start(4);
    printf("Token 4 ('%s') is_word_start: %s\n",
           tokenizer.decode_token(4).c_str(),
           ws4 ? "true" : "false");

    printf("OK: Word start detection\n");
}

int main() {
    printf("=== Testing Tokenizer ===\n");

    test_load();
    test_decode_single();
    test_decode_sequence();
    test_word_start();

    printf("\n=== All Tokenizer tests complete ===\n");
    return 0;
}
