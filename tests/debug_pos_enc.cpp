#include <cstdio>
#include "conformer_modules.h"

using namespace nemo;

int main() {
    RelPositionalEncoding pos_enc;
    pos_enc.init();
    
    TensorF pos_emb;
    size_t seq_len = 251;
    pos_enc.get_pos_emb(seq_len, pos_emb);
    
    size_t out_len = 2 * seq_len - 1;  // 501
    
    printf("C++ pos_emb shape: [%zu, %zu]\n", pos_emb.shape[0], pos_emb.shape[1]);
    
    // pos_emb[0, :5] should be position 250
    printf("\npos_emb[0, :5] (pos %d):\n  ", (int)seq_len-1);
    for (int i = 0; i < 5; i++) printf("%.6f ", pos_emb(0, i));
    printf("\n");
    
    // pos_emb[250, :5] should be position 0
    printf("\npos_emb[250, :5] (pos 0):\n  ");
    for (int i = 0; i < 5; i++) printf("%.6f ", pos_emb(250, i));
    printf("\n");
    
    // pos_emb[500, :5] should be position -250
    printf("\npos_emb[500, :5] (pos -%d):\n  ", (int)seq_len-1);
    for (int i = 0; i < 5; i++) printf("%.6f ", pos_emb(500, i));
    printf("\n");
    
    return 0;
}
