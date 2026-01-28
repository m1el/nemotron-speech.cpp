import torch
import yaml
from omegaconf import OmegaConf
from nemo.utils import logging
from nemo.collections.asr.parts.utils.transcribe_utils import get_inference_device, get_inference_dtype, setup_model
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

logging.setLevel(logging.ERROR)
model_cfg = "../nemotron-speech-streaming-en-0.6b/model_config.yaml"
model_path = "../nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo"
cfg = OmegaConf.load(model_cfg)
cfg.model_path = model_path
cfg.pad_and_drop_preencoded = True
cfg.amp = False
cfg.audio_file = "../test/HFTKzy5xRM-cut.wav"

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

def perform_streaming(
    asr_model,
    streaming_buffer,
    compute_dtype: torch.dtype,
    debug_mode=False,
    pad_and_drop_preencoded=False,
):
    batch_size = len(streaming_buffer.streams_length)
    
    final_offline_tran = None

    cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        print(f"Processing chunk {step_num}, shape: {chunk_audio.shape} len={chunk_lengths}")
        actual_chunk = chunk_audio[0, :, :]
        print('actual mel chunk:', actual_chunk)
        print('actual mel chunk row:', actual_chunk[0, :])
        with torch.inference_mode():
            # keep_all_outputs needs to be True for the last step of streaming when model is trained with att_context_style=regular
            # otherwise the last outputs would get dropped
            chunk_audio = chunk_audio.to(compute_dtype)
            with torch.no_grad():
                if pred_out_stream:
                    print('pred_out_stream=', pred_out_stream)
                print(f'cache_last_channel {cache_last_channel.shape} = {cache_last_channel}')
                print(f'cache_last_time {cache_last_time.shape} = {cache_last_time}')
                print(f'cache_last_channel_len = {cache_last_channel_len}')
                # print('previous_hypotheses=', previous_hypotheses.shape)
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
                print('in progress:', extract_transcriptions(transcribed_texts))

        assert step_num < 10, "only looking at 10 steps now"
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
    asr_model.encoder.set_default_att_context_size(att_context_size=[70, 0])
    # print(model_name, asr_model)
    print(asr_model.encoder.streaming_cfg)

    online_normalization = False # not in live
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    print('Streaming preprocessor:', streaming_buffer.preprocessor)
    with torch.amp.autocast('cuda' if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=cfg.amp):
        print('SINGLE FILE')
        # stream a single audio file
        processed_signal, processed_signal_length, _ = streaming_buffer.append_audio_file(cfg.audio_file, stream_id=-1)
        print('processed_signal shape:', processed_signal.shape, 'processed_signal_length:', processed_signal_length)
        processed_signal.cpu().numpy().transpose().tofile('my_bin/nemo_mel_data.bin')

        print(streaming_buffer.streaming_cfg.chunk_size, streaming_buffer.streaming_cfg.shift_size)
        streaming_tran, _ = perform_streaming(
            debug_mode=True,
            asr_model=asr_model,
            streaming_buffer=streaming_buffer,
            compute_dtype=compute_dtype,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
        )
        print('transcription:', streaming_tran)
if __name__ == "__main__":
    main()