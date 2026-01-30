from socket import close
import torch
import torch.nn.functional as F
import struct, traceback
from omegaconf import OmegaConf
from nemo.utils import logging
from nemo.collections.asr.parts.utils.transcribe_utils import get_inference_device, get_inference_dtype, setup_model
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv2D

logging.setLevel(logging.ERROR)

# Patch CausalConv2D to dump padded input (will be initialized after dump_first_only is defined)
_causal_conv2d_call_count = [0]  # mutable counter
_dump_padded_fn = [None]  # will hold dump_first_only reference
def _patched_causal_conv2d_forward(self, x):
    padded = F.pad(x, pad=(self._left_padding, self._right_padding, self._left_padding, self._right_padding))
    _causal_conv2d_call_count[0] += 1
    # Only dump first call (conv0 of first chunk)
    if _causal_conv2d_call_count[0] == 1 and _dump_padded_fn[0]:
        _dump_padded_fn[0](padded, 'my_bin/nemo_conv0_padded.bin')
    result = torch.nn.Conv2d.forward(self, padded)
    return result
CausalConv2D.forward = _patched_causal_conv2d_forward
model_cfg = "../nemotron-speech-streaming-en-0.6b/model_config.yaml"
model_path = "../nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo"
cfg = OmegaConf.load(model_cfg)
cfg.model_path = model_path
cfg.pad_and_drop_preencoded = True
cfg.amp = False
cfg.audio_file = "../test/HFTKzy5xRM-cut.wav"


chunk_counter = [0]  # Use list to allow modification in nested function

def mk_hook(module_name):
    def hook(module, args, kwargs, output):
        if module_name == 'ASRModel.encoder.pre_encode':
            dump_append_data(kwargs["x"], 'my_bin/nemo_subsampling_input.bin')
            dump_append_data(output[0], 'my_bin/nemo_subsampling_output.bin')
        # Dump encoder output (after all conformer layers)
        if module_name == 'ASRModel.encoder.cache_aware_stream_step':
            # output is (encoded, encoded_len, cache_last_channel, cache_last_time, cache_last_channel_len)
            dump_append_data(output[0], 'my_bin/nemo_encoder_output.bin')
            # Dump cache state for chunks 0 and 1 - cache_last_channel is [n_layers, batch, cache_len, d_model]
            # This is the OUTPUT cache (will be INPUT for next chunk)
            chunk = chunk_counter[0]
            if chunk < 3:
                # Dump layer 0 K cache: shape [batch=1, cache_len=70, d_model=1024]
                dump_append_data(output[2][0], f'my_bin/nemo_k_cache_L0_after_chunk{chunk}.bin')
                # Dump conv cache: shape [batch=1, d_model=1024, conv_cache=8]
                dump_append_data(output[3][0], f'my_bin/nemo_conv_cache_L0_after_chunk{chunk}.bin')
                # Dump cache_last_channel_len
                print(f"[NeMo] Chunk {chunk}: cache_last_channel_len = {output[4]}")
            chunk_counter[0] += 1
        # Dump decoder joint logits
        if module_name == 'ASRModel.decoding.rnnt_decoder_predictions_tensor':
            # This is the main decoding function, can inspect args
            pass
        # Dump all joint values from the SAME forward pass
        # Track enc/pred/joint_net all at once on first call
        if module_name == 'ASRModel.joint.enc':
            dump_first_only(output, 'my_bin/nemo_joint_enc.bin')
        if module_name == 'ASRModel.joint.pred':
            dump_first_only(output, 'my_bin/nemo_joint_pred.bin')
        if module_name == 'ASRModel.joint.joint_net':
            dump_first_only(output, 'my_bin/nemo_joint_logits.bin')
        if module_name == 'ASRModel.encoder._create_masks':
            dump_first_only(output[0], 'my_bin/nemo_pad_mask_out.bin')
            dump_first_only(output[1], 'my_bin/nemo_attn_mask_out.bin')
        # Dump L1 attention input (query/key/value) and output to debug divergence
        if module_name == 'ASRModel.encoder.layers.1.self_attn':
            q = kwargs.get('query') if 'query' in kwargs else args[0]
            dump_append_data(q, 'my_bin/nemo_L1_attn_input.bin')
            attn_out = output[0] if isinstance(output, tuple) else output
            dump_append_data(attn_out, 'my_bin/nemo_L1_attn_output.bin')
        # Dump L1 depthwise conv input/output to debug conv cache issue
        if module_name == 'ASRModel.encoder.layers.1.conv.depthwise_conv':
            # Input is after GLU, shape [batch, d_model, seq_len]
            conv_in = args[0] if args else kwargs.get('x')
            dump_append_data(conv_in, 'my_bin/nemo_L1_dwconv_input.bin')
            # Output is [batch, d_model, seq_len] (or tuple with cache)
            conv_out = output[0] if isinstance(output, tuple) else output
            dump_append_data(conv_out, 'my_bin/nemo_L1_dwconv_output.bin')
        # Dump L2 attention input and output
        if module_name == 'ASRModel.encoder.layers.2.self_attn':
            q = kwargs.get('query') if 'query' in kwargs else args[0]
            dump_append_data(q, 'my_bin/nemo_L2_attn_input.bin')
            attn_out = output[0] if isinstance(output, tuple) else output
            dump_append_data(attn_out, 'my_bin/nemo_L2_attn_output.bin')
        # Dump conformer layer outputs
        if module_name == 'ASRModel.encoder.layers.0':
            out_tensor = output[0] if isinstance(output, tuple) else output
            dump_append_data(out_tensor, 'my_bin/nemo_conformer_layer0.bin')
        if module_name == 'ASRModel.encoder.layers.1':
            out_tensor = output[0] if isinstance(output, tuple) else output
            dump_append_data(out_tensor, 'my_bin/nemo_conformer_layer1.bin')
        if module_name == 'ASRModel.encoder.layers.2':
            out_tensor = output[0] if isinstance(output, tuple) else output
            dump_append_data(out_tensor, 'my_bin/nemo_conformer_layer2.bin')
        if module_name == 'ASRModel.encoder.layers.3':
            out_tensor = output[0] if isinstance(output, tuple) else output
            dump_append_data(out_tensor, 'my_bin/nemo_conformer_layer3.bin')
        if module_name == 'ASRModel.encoder.layers.4':
            out_tensor = output[0] if isinstance(output, tuple) else output
            dump_append_data(out_tensor, 'my_bin/nemo_conformer_layer4.bin')
        if module_name == 'ASRModel.encoder.layers.5':
            out_tensor = output[0] if isinstance(output, tuple) else output
            dump_append_data(out_tensor, 'my_bin/nemo_conformer_layer5.bin')
        if module_name == 'ASRModel.encoder.layers.6':
            out_tensor = output[0] if isinstance(output, tuple) else output
            dump_append_data(out_tensor, 'my_bin/nemo_conformer_layer6.bin')
        if module_name == 'ASRModel.encoder.layers.12':
            out_tensor = output[0] if isinstance(output, tuple) else output
            dump_append_data(out_tensor, 'my_bin/nemo_conformer_layer12.bin')
        if module_name == 'ASRModel.encoder.streaming_post_process':
            # encoded, encoded_len, cache_last_channel, cache_last_time, cache_last_channel_len
            # ((Tensor[1, 1024, 1], Tensor[1], Tensor[24, 1, 70, 1024], Tensor[24, 1, 1024, 8], Tensor[1]), {'keep_all_outputs': 'bool'})
            # (Tensor[1, 1024, 1], Tensor[1], Tensor[24, 1, 70, 1024], Tensor[24, 1, 1024, 8], Tensor[1])
            dump_append_data(output[0], 'my_bin/encoder_streaming_postproc.bin')
        # Dump each conv layer in subsampling by class name
        prefix = 'ASRModel.encoder.pre_encode.conv'
        if module_name.startswith(prefix):
            (mn, idx) = module_name.rsplit('.', 1)
            if mn == prefix:
                shape = '_'.join(str(s) for s in output.shape)
                dump_append_data(output, f'my_bin/nemo_conv_layer{idx}_{shape}.bin')
        if kwargs:
            args = (*args, kwargs)
        ty = get_ty((args, output))
        append_name(module_name, module, ty)
        print(f"Inside {module_name} {module.__class__.__name__} {ty}")
    return hook


def extract_transcriptions(hyps):
    """
    The transcribed_texts returned by CTC and RNNT models are different.
    This method would extract and return the text section of the hypothesis.
    """
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    # for the first step there is no need to drop any tokens after the downsampling as no caching is being used
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded

from collections import OrderedDict
import inspect

all_names = OrderedDict()
def append_name(name, mod, inout=None):
    chunks = name.split('.')
    cur = all_names
    for c in chunks:
        if c not in cur:
            cur[c] = {}
        cur = cur[c]
    cur['##cls'] = mod.__class__.__name__
    fn = mod.forward if hasattr(mod, 'forward') else mod
    cur['##sig'] = inspect.signature(fn)
    if inout is not None:
        cur['##inout'] = inout

# Flat pretty print
def pprint_names_flat(d, prefix=''):
    if prefix != '':
        prefix = prefix + '.'
    for k, v in d.items():
        if k.startswith('##'): continue
        if k != '':
            inout = v.get('##inout', '')
            if inout != '':
                i, o = inout
                inout = f' | {i} -> {o}'
            sig = v.get('##sig', '')
            if sig != '':
                inout = str(sig) + inout
            print(f'{prefix}{k}: {v.get("##cls", "unknown")}{inout}')
        pprint_names_flat(v, prefix=prefix + k)

# Hierarchical pretty print
def pprint_names_tree(d, indent=0):
    for k, v in d.items():
        if k.startswith('##'): continue
        if k != '':
            inout = v.get('##inout', '')
            if inout != '':
                i, o = inout
                inout = f' | {i} -> {o}'
            sig = v.get('##sig', '')
            if sig != '':
                inout = str(sig) + inout
            print('  ' * indent + f'{k}: {v.get("##cls", "unknown")}{inout}')
        pprint_names_tree(v, indent + 1)

class TensorShape:
    def __init__(self, tensor):
        self.shape = list(tensor.shape)
    def __str__(self):
        return f'Tensor{self.shape}'
    def __repr__(self):
        return self.__str__()

def get_ty(obj):
    if isinstance(obj, tuple):
        return tuple(get_ty(o) for o in obj)
    elif isinstance(obj, list):
        return [get_ty(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: get_ty(v) for k, v in obj.items()}
    elif torch.is_tensor(obj):
        return TensorShape(obj)
    else:
        return type(obj).__name__

output_files = {}
dumped_once = set()  # Track files that should only be dumped once

def dump_first_only(tensor, filename):
    """Dump tensor only on first call (for varying shapes)"""
    if filename in dumped_once:
        return
    dumped_once.add(filename)
    dump_append_data(tensor, filename)

# Initialize the CausalConv2D patch function
_dump_padded_fn[0] = dump_first_only

def dump_append_data(tensor, filename):
    if filename not in output_files:
        # traceback.print_stack()
        print(f'Creating file {filename} tensor shape {TensorShape(tensor)}')
        shape = list(tensor.shape)
        shape.reverse()
        while len(shape) < 4: shape.append(1)
        struct.pack_into('4q', b := bytearray(32), 0, *shape)
        # print(f'Header bytes: {b.hex()}')
        file = open(filename, "wb")
        assert(len(b) == 32)
        file.write(b)
        output_files[filename] = (file, list(tensor.shape))
    file, shape = output_files[filename]
    # if list(tensor.shape) != shape:
    #     traceback.print_stack()

    assert list(tensor.shape) == shape, \
        f"Shape mismatch for {filename}: expected {shape}, got {list(tensor.shape)}"
    file.write(tensor.detach().cpu().numpy().tobytes())

def close_output_files():
    for _filename, (file, _shape) in output_files.items():
        file.close()

def hook_patch_object_method(obj, name, prefix, force=False):
    if not hasattr(obj, name):
        assert not force, f'Object {obj} has no method {name}'
        return
    orig_method = getattr(obj, name)
    hook_fn = mk_hook(prefix + name)
    def wrapped(*args, **kwargs):
        rv = orig_method(*args, **kwargs)
        hook_fn(orig_method, args, kwargs, rv)
        return rv
    setattr(obj, name, wrapped)

visited = set()
def instrument_everything(model, prefix=''):
    if prefix == 'ASRModel':
        hook_patch_object_method(model.decoding.decoding, 'forward', prefix + '.decoding.decoding.', force=True)
        hook_patch_object_method(model.decoding.decoding, '_greedy_decode_blank_as_pad_loop_labels', prefix + '.decoding.decoding.', force=True)
    # print(f'{model.__class__.__name__} instrument_everything called for prefix="{prefix}"')
    append_name(prefix, model)
    if prefix != '':
        prefix = prefix + '.'

    hook_patch_object_method(model, '_create_masks', prefix)
    hook_patch_object_method(model, 'conformer_stream_step', prefix)
    hook_patch_object_method(model, 'cache_aware_stream_step', prefix)
    hook_patch_object_method(model, 'rnnt_decoder_predictions_tensor', prefix)
    hook_patch_object_method(model, 'streaming_post_process', prefix)
    if model in visited: return
    visited.add(model)
    for name, module in model.named_modules():
        if module in visited: continue
        visited.add(module)
        # print(f"Visiting module: {prefix + name} of type {module.__class__.__name__}")
        if name == '': continue
        module.register_forward_hook(mk_hook(prefix + name), with_kwargs=True)
        instrument_everything(module, prefix=prefix + name)

def perform_streaming(
    asr_model,
    streaming_buffer,
    compute_dtype: torch.dtype,
    debug_mode=False,
    pad_and_drop_preencoded=False,
):
    instrument_everything(asr_model, prefix='ASRModel')
    batch_size = len(streaming_buffer.streams_length)
    
    final_offline_tran = None

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        # print(f"Processing chunk {step_num}, shape: {chunk_audio.shape} len={chunk_lengths}")
        # actual_chunk = chunk_audio[0, :, :]
        # print('actual mel chunk:', actual_chunk)
        # print('actual mel chunk row:', actual_chunk[0, :])
        with torch.inference_mode():
            # keep_all_outputs needs to be True for the last step of streaming when model is trained with att_context_style=regular
            # otherwise the last outputs would get dropped
            chunk_audio = chunk_audio.to(compute_dtype)
            with torch.no_grad():
                # if pred_out_stream:
                #     print('pred_out_stream=', pred_out_stream)
                # print(f'cache_last_channel {cache_last_channel.shape} = {cache_last_channel}')
                # print(f'cache_last_time {cache_last_time.shape} = {cache_last_time}')
                # print(f'cache_last_channel_len = {cache_last_channel_len}')
                # print('previous_hypotheses=', previous_hypotheses.shape)
                # tran = actual_chunk.transpose(0, 1).unsqueeze(0)
                # pre_encoded, lengths = asr_model.encoder.pre_encode(x=tran, lengths=chunk_lengths)
                #                 # Dump pre-encoded audio signal for debugging
                # with open("my_bin/nemo_pre_encoded_raw.bin", "ab") as f:
                #     f.write(pre_encoded.detach().cpu().numpy().tobytes())
                # with open("my_bin/nemo_mel_data.bin", "ab") as f:
                #     f.write(tran.detach().cpu().numpy().tobytes())

                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = asr_model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=streaming_buffer.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded),
                    return_transcription=True,
                )
                # print('in progress:', extract_transcriptions(transcribed_texts))

        # assert step_num < 10, "only looking at 10 steps now"
        # pprint_names_flat(all_names)
        # pprint_names_tree(all_names)
        assert False, "one step"
        if debug_mode:
            logging.info(f"Streaming transcriptions: {extract_transcriptions(transcribed_texts)}")

    final_streaming_tran = extract_transcriptions(transcribed_texts)
    logging.info(f"Final streaming transcriptions: {final_streaming_tran}")

    return final_streaming_tran, final_offline_tran


def main():
    compute_dtype = torch.float32
    device = torch.device('cuda')
    amp_dtype = torch.float16
    asr_model, model_name = setup_model(cfg=cfg, map_location=device)
    asr_model = asr_model.to(device=device, dtype=compute_dtype)
    asr_model.eval()
    # asr_model.encoder.setup_streaming_params(att_context_size=[70,1])
    # possible values: [[70, 0], [70, 1], [70, 6], [70, 13]]
    # [(x+2)*8+1 for x in [0,1,6,13]]
    # [17, 25, 65, 121]
    asr_model.encoder.set_default_att_context_size(att_context_size=[70, 0])
    # print(model_name, asr_model)
    # print(asr_model.encoder.streaming_cfg)
    import inspect
    # print(inspect.getfile(asr_model.decoder.forward))

    online_normalization = False # not in live
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    # print('Streaming preprocessor:', streaming_buffer.preprocessor)
    with torch.amp.autocast('cuda' if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=cfg.amp):
        # print('SINGLE FILE')
        # stream a single audio file
        processed_signal, processed_signal_length, _ = streaming_buffer.append_audio_file(cfg.audio_file, stream_id=-1)
        # print('processed_signal shape:', processed_signal.shape, 'processed_signal_length:', processed_signal_length)
        # processed_signal.cpu().numpy().transpose().tofile('my_bin/nemo_mel_data.bin')

        # print(streaming_buffer.streaming_cfg.chunk_size, streaming_buffer.streaming_cfg.shift_size)
        streaming_tran, _ = perform_streaming(
            debug_mode=True,
            asr_model=asr_model,
            streaming_buffer=streaming_buffer,
            compute_dtype=compute_dtype,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
        )
        print('transcription:', streaming_tran)
if __name__ == "__main__":
    try:
        main()
    finally:
        close_output_files()