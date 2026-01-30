#include "include/rnnt_joint.h"

#include <algorithm>

namespace nemo {

void RNNTJoint::load_weights(const ModelWeights& weights) {
    enc_weight = weights.require("joint.enc.weight").data.data();
    enc_bias = weights.require("joint.enc.bias").data.data();
    pred_weight = weights.require("joint.pred.weight").data.data();
    pred_bias = weights.require("joint.pred.bias").data.data();
    out_weight = weights.require("joint.joint_net.2.weight").data.data();
    out_bias = weights.require("joint.joint_net.2.bias").data.data();
}

void RNNTJoint::forward(const TensorF& enc_out, const TensorF& dec_out, TensorF& logits) {
    // enc_out: [batch, enc_dim] or [batch, 1, enc_dim]
    // dec_out: [batch, dec_dim]

    size_t batch = dec_out.shape[0];

    // Handle both 2D and 3D encoder output
    TensorF enc_2d;
    if (enc_out.ndim() == 3) {
        // [batch, 1, enc_dim] -> [batch, enc_dim]
        enc_2d.resize({batch, ENCODER_DIM});
        for (size_t b = 0; b < batch; b++) {
            for (size_t d = 0; d < ENCODER_DIM; d++) {
                enc_2d(b, d) = enc_out(b, 0, d);
            }
        }
    } else {
        enc_2d = enc_out;
    }

    // Project encoder: [batch, 1024] -> [batch, 640]
    linear(enc_2d, enc_weight, JOINT_DIM, ENCODER_DIM, enc_bias, enc_proj_);

    // Project decoder: [batch, 640] -> [batch, 640]
    linear(dec_out, pred_weight, JOINT_DIM, DECODER_DIM, pred_bias, dec_proj_);

    // Add and ReLU: joint = ReLU(enc_proj + dec_proj)
    joint_.resize({batch, JOINT_DIM});
    for (size_t b = 0; b < batch; b++) {
        for (size_t d = 0; d < JOINT_DIM; d++) {
            float val = enc_proj_(b, d) + dec_proj_(b, d);
            joint_(b, d) = std::max(0.0f, val);  // ReLU
        }
    }

    // Output projection: [batch, 640] -> [batch, 1025]
    linear(joint_, out_weight, VOCAB_SIZE, JOINT_DIM, out_bias, logits);
}

void RNNTJoint::forward_full(const TensorF& enc_out, const TensorF& dec_out, TensorF& logits) {
    // enc_out: [batch, time, enc_dim]
    // dec_out: [batch, dec_dim]

    size_t batch = enc_out.shape[0];
    size_t time = enc_out.shape[1];

    // Project encoder: [batch, time, 1024] -> [batch, time, 640]
    linear(enc_out, enc_weight, JOINT_DIM, ENCODER_DIM, enc_bias, enc_proj_);

    // Project decoder: [batch, 640] -> [batch, 640]
    linear(dec_out, pred_weight, JOINT_DIM, DECODER_DIM, pred_bias, dec_proj_);

    // Broadcast add and ReLU
    // enc_proj_: [batch, time, 640]
    // dec_proj_: [batch, 640]
    // joint: [batch, time, 640]
    joint_.resize({batch, time, JOINT_DIM});
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = 0; t < time; t++) {
            for (size_t d = 0; d < JOINT_DIM; d++) {
                float val = enc_proj_(b, t, d) + dec_proj_(b, d);
                joint_(b, t, d) = std::max(0.0f, val);  // ReLU
            }
        }
    }

    // Output projection: [batch, time, 640] -> [batch, time, 1025]
    linear(joint_, out_weight, VOCAB_SIZE, JOINT_DIM, out_bias, logits);
}

}  // namespace nemo
