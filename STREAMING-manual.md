Example streaming implemented in `scripts/my_streaming.py`.
Run using `uv run scripts/my_streaming.py`.

Model config audio:
sample_rate: 16000
window_size: 0.025sec / 400samples
window_stride: 0.01sec / 160samples

Audio chunking is defined by `att_context_size = (left_chunks_num, chunk_size)`

`cache_last_channel, cache_last_time, cache_last_channel_len`
is the streaming state preserved between the runs.

The sizes of these tensors are defined by
n_layers: 24
d_model: 1024
conv_kernel_size: 9
conv_context_size: 'causal' -> [(conv_kernel_size-1), 0] -> [8, 0]

cache_last_channel = create_tensor(
    (
        model_config.n_layers, # 24
        batch_size,            # 1
        att_context_size[0],   # 70 self.streaming_cfg.last_channel_cache_size
        model_config.d_model,  # 1024
    )
)
cache_last_time = create_tensor(
    (
        model_config.n_layers,             # 24
        batch_size,                        # 1
        d_model,                           # 1024
        # last_time_cache_size
        model_config.conv_context_size[0], # 8
    ),
    device=device,
    dtype=dtype,
)
cache_last_channel_len = torch.zeros(batch_size) # 1

