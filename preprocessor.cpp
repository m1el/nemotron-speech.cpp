#include <cassert>
#include <cmath>
#include <span>
#include <vector>
#include "stdint.h"
#include "stdio.h"

typedef enum {
    WINDOW_HANN,
    WINDOW_HAMMING,
    WINDOW_BLACKMAN,
    WINDOW_BARTLETT
} WindowType;

typedef enum {
    NORMALIZE_NONE,
    NORMALIZE_PER_FEATURE,
    NORMALIZE_ALL_FEATURES
} NormalizeType;

typedef enum {
    LOG_ZERO_GUARD_ADD,
    LOG_ZERO_GUARD_CLAMP
} LogZeroGuardType;

struct Shape {
    size_t n_dims;
    size_t dims[6];
    size_t size() const {
        size_t s = 1;
        for (size_t i = 0; i < n_dims; i++) {
            s *= dims[i];
        }
        return s;
    }
};
struct Tensor {
    Shape shape;
    std::vector<float> data;
    Tensor() {}
    Tensor(Shape shape) {
        resize(shape);
    }
    void fill(float value) {
        std::fill(data.begin(), data.end(), value);
    }
    void resize(Shape new_shape) {
        shape = new_shape;
        data.resize(shape.size());
    }
    void resize(size_t n_rows, size_t n_cols) {
        Shape shape = { .n_dims = 2, .dims = { n_rows, n_cols } };
        resize(shape);
    }
    float& operator()(size_t i, size_t j) {
        assert(shape.n_dims == 2 && i < shape.dims[0] && j < shape.dims[1]);
        return data[i * shape.dims[1] + j];
    }
};

typedef struct {
    // Sample rate of the input audio data.
    // Defaults to 16000
    int sample_rate;
    // Size of window for fft in seconds
    // Defaults to sample_rate * 0.02
    size_t n_window_size;
    // Stride of window for fft in seconds
    // Defaults to sample_rate * 0.01
    size_t n_window_stride;
    // Windowing function for fft. can be one of WINDOW_HANN,
    // WINDOW_HAMMING, WINDOW_BLACKMAN, WINDOW_BARTLETT
    // Defaults to WINDOW_HANN
    WindowType window;
    // Can be one of NORMALIZE_PER_FEATURE, NORMALIZE_ALL_FEATURES; all
    // other options disable feature normalization.
    // NORMALIZE_ALL_FEATURES normalizes the entire spectrogram to be mean 0 with std 1.
    // NORMALIZE_PER_FEATURE normalizes per channel / freq instead.
    // Defaults to NORMALIZE_PER_FEATURE
    NormalizeType normalize;
    // Length of FT window.
    // If 0, it uses the smallest power of 2 that is larger than n_window_size.
    // Defaults to 0
    size_t n_fft;
    // Amount of pre emphasis to add to audio. Can be disabled by passing 0.0.
    // Defaults to 0.97     
    float preemph;
    // Number of mel spectrogram freq bins to output.
    // Defaults to 64
    size_t features;
    // Lower bound on mel basis in Hz.
    // Defaults to 0
    size_t lowfreq;
    // Lower bound on mel basis in Hz.
    // Defaults to None
    size_t highfreq;
    // Log features.
    // Defaults to True
    bool log;
    // Need to avoid taking the log of zero. There
    // are two options: LOG_ZERO_GUARD_ADD or LOG_ZERO_GUARD_CLAMP.
    // Defaults to LOG_ZERO_GUARD_ADD.
    LogZeroGuardType log_zero_guard_type;
    // Add or clamp requires the number to add with or clamp to.
    // log_zero_guard_value is a float.
    // Defaults to 2**-24.
    float log_zero_guard_value;
    // Amount of white-noise dithering.
    // Defaults to 1e-5
    float dither;
    // Ensures that the output size of the time dimension
    // is a multiple of pad_to.
    // Defaults to 16
    size_t pad_to;
    // Defaults to 1
    size_t frame_splicing;
    // If True, sets stft center to False and adds padding,
    // such that num_frames = audio_length // hop_length.
    // Defaults to False.
    bool exact_pad;
    // The value that shorter mels are padded with.
    // Defaults to 0.0
    float pad_value;
    // The power that the linear spectrogram is raised to
    // prior to multiplication with mel basis.
    // Defaults to 2 for a power spec
    float mag_power;
    // Probability with which narrowband augmentation would be applied to
    // samples in the batch.
    // Defaults to 0.0
    float nb_augmentation_prob;
    // Frequency above which all frequencies will be masked for narrowband augmentation.
    // Defaults to 4000
    size_t nb_max_freq;
    // Normalization used for mel filterbank weights.
    // Defaults to 0.0 (area normalization, or "slaney")
    float mel_norm;
} PreprocessorConfig;

// nemotron-speech-streaming-en-0.6b preprocessor config
PreprocessorConfig nemotron_speech_preprocessor = {
    .sample_rate = 16000,
    .n_window_size = 400,
    .n_window_stride = 160,
    .window = WINDOW_HANN,
    .normalize = NORMALIZE_NONE,
    .n_fft = 512,
    .preemph = 0.97,
    .features = 128,
    .lowfreq = 0,
    .highfreq = 0,
    .log = true,
    .log_zero_guard_type = LOG_ZERO_GUARD_ADD,
    .log_zero_guard_value = 5.960464477539063e-8, // 2**-24
    .dither = 1.0e-05,
    .pad_to = 0,
    .frame_splicing = 1,
    .exact_pad = false,
    .pad_value = 0.0,
    .mag_power = 2.0,
    .nb_augmentation_prob = 0.0,
    .nb_max_freq = 4000,
    .mel_norm = 0.0
};

std::vector<float> fft_frequencies(float sr, size_t n_fft) {
    size_t cols = 1 + n_fft / 2;
    std::vector<float> frequencies(cols);
    float freq_step = sr / n_fft;
    for (size_t i = 0; i < cols; i++) {
        frequencies[i] = i * freq_step;
    }
    return frequencies;
}

float hz_to_mel(float frequency) {
    // Slaney-style mel scale (htk=false)
    const float f_min = 0.0f;
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp;
    const float logstep = logf(6.4f) / 27.0f;

    float mel = (frequency - f_min) / f_sp;
    if (frequency >= min_log_hz) {
        mel = min_log_mel + logf(frequency / min_log_hz) / logstep;
    }
    return mel;
}

float mel_to_hz(float mel) {
    // Slaney-style mel scale (htk=false)
    const float f_min = 0.0f;
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp;
    const float logstep = logf(6.4f) / 27.0f;

    float freq = f_min + f_sp * mel;
    if (mel >= min_log_mel) {
        freq = min_log_hz * expf(logstep * (mel - min_log_mel));
    }
    return freq;
}

std::vector<float> mel_frequencies(size_t n_mels, float fmin, float fmax) {
    assert(n_mels > 2);
    float min_mel = hz_to_mel(fmin);
    float max_mel = hz_to_mel(fmax);

    std::vector<float> hz(n_mels);
    for (size_t i = 0; i < n_mels; i++) {
        float mel = min_mel + (max_mel - min_mel) * i / (n_mels - 1);
        hz[i] = mel_to_hz(mel);
    }
    return hz;
}

Tensor generate_filterbank(PreprocessorConfig& m_cfg) {
    size_t cols = 1 + m_cfg.n_fft / 2;
    Tensor weights({ .n_dims = 2, .dims = {m_cfg.features, cols} });
    std::vector<float> fftfreqs = fft_frequencies(m_cfg.sample_rate, m_cfg.n_fft);
    std::vector<float> mel_f = mel_frequencies(m_cfg.features + 2, m_cfg.lowfreq, m_cfg.highfreq);
    // fdiff = np.diff(mel_f)
    std::vector<float> fdiff(mel_f.size() - 1);
    for (size_t i = 0; i < fdiff.size(); i++) {
        fdiff[i] = mel_f[i + 1] - mel_f[i];
    }
    // ramps = np.subtract.outer(mel_f, fftfreqs)
    Tensor ramps({ .n_dims = 2, .dims = {mel_f.size(), fftfreqs.size()} });
    for (size_t i = 0; i < mel_f.size(); i++) {
        for (size_t j = 0; j < fftfreqs.size(); j++) {
            ramps(i, j) = mel_f[i] - fftfreqs[j];
        }
    }
    // 
    for (size_t i = 0; i < m_cfg.features; i++) {
        for (size_t j = 0; j < fftfreqs.size(); j++) {
            float lower = -ramps(i, j) / fdiff[i];
            float upper = ramps(i + 2, j) / fdiff[i + 1];
            weights(i, j) = std::max(0.0f, std::min(lower, upper));
        }
    }
    float norm = m_cfg.mel_norm;
    if (norm == 0.0) { // "slaney"
        for (size_t i = 0; i < m_cfg.features; i++) {
            float enorm = 2.0f / (mel_f[i + 2] - mel_f[i]);
            for (size_t j = 0; j < fftfreqs.size(); j++) {
                weights(i, j) *= enorm;
            }
        }
    } else {
        // weights = util.normalize(weights, norm=norm, axis=-1)
        for (size_t i = 0; i < m_cfg.features; i++) {
            float norm_factor = 0.0f;
            for (size_t j = 0; j < fftfreqs.size(); j++) {
                norm_factor += pow(weights(i, j), norm);
            }
            norm_factor = pow(norm_factor, 1.0f / norm);
            if (norm_factor > 0.0f) {
                for (size_t j = 0; j < fftfreqs.size(); j++) {
                    weights(i, j) /= norm_factor;
                }
            }
        }
    }

    return weights;
}

struct Preprocessor {
    PreprocessorConfig m_cfg;
    std::vector<float> m_window;
    Tensor m_filterbank;
    std::vector<float> m_sin_vals;
    std::vector<float> m_cos_vals;

    // buffers for forward pass
    std::vector<float> m_frame;
    std::vector<float> m_real_out;
    std::vector<float> m_imag_out;
    std::vector<std::vector<float>> m_spectrogram;
    std::vector<float> m_audio_buf;

    Preprocessor(PreprocessorConfig config) {
        m_cfg = config;
        if (m_cfg.highfreq == 0.0) {
            m_cfg.highfreq = m_cfg.sample_rate / 2.0;
        }
        fill_sin_cos_table(m_cfg.n_fft);
        // fill_window(m_cfg.n_window_size, false);
        // m_filterbank = generate_filterbank(m_cfg);
        
        // Pre-allocate FFT buffers
        size_t n_bins = 1 + m_cfg.n_fft / 2;
        m_frame.resize(m_cfg.n_fft);
        m_real_out.resize(n_bins);
        m_imag_out.resize(n_bins);
        m_spectrogram.resize(n_bins);
    }
    int load_weights(const char* window, const char* fb) {
        // Load window and filterbank weights from binary files
        FILE* fwin = fopen(window, "rb");
        if (!fwin) {
            return -1;
        }
        m_window.resize(m_cfg.n_window_size);
        size_t read = fread(m_window.data(), sizeof(float), m_window.size(), fwin);
        printf("Read %zu window values\n", read);
        fclose(fwin);
        FILE* ffb = fopen(fb, "rb");
        if (!ffb) {
            return -1;
        }
        m_filterbank.resize(m_cfg.features, 1 + m_cfg.n_fft / 2);
        read = fread(m_filterbank.data.data(), sizeof(float), m_filterbank.shape.size(), ffb);
        printf("Read %zu filterbank values\n", read);
        fclose(ffb);
        return 0;
    }

    void fill_sin_cos_table(size_t n_fft) {
        m_sin_vals.resize(n_fft);
        m_cos_vals.resize(n_fft);
        for (size_t i = 0; i < n_fft; i++) {
            float theta = (2 * M_PI * i) / n_fft;
            m_sin_vals[i] = sinf(theta);
            m_cos_vals[i] = cosf(theta);
        }
    }

    void fill_window(size_t length, bool periodic) {
        size_t adjusted_length = length;
        if (periodic) {
            adjusted_length += 1;
        }
        float reciprocal = 1.0f / (adjusted_length - 1);
        // Hann window  
        float alpha = 0.5f;
        float beta = 0.5f;
        m_window.resize(length);
        for (size_t i = 0; i < length; i++) {
            m_window[i] = alpha - beta * cosf((2.0f * M_PI * i) * reciprocal);
        }
    }
    
    // Get the output sequence length given input audio length
    size_t get_seq_len(size_t seq_len) {
        if (seq_len == 0) {
            return 0;
        }
        // Assuming center is True, stft_pad_amount = n_fft // 2
        size_t stft_pad_amount = m_cfg.exact_pad ? (m_cfg.n_window_size / 2) : 0;
        size_t pad_amount = (stft_pad_amount > 0) ? stft_pad_amount * 2 : m_cfg.n_fft;
        return (seq_len + pad_amount - m_cfg.n_fft) / m_cfg.n_window_stride;
    }

    // Calculate output shape from input audio length
    Shape get_output_shape(size_t audio_len, bool linear_spec = false) {
        // Calculate number of frames (center=True padding)
        size_t pad_amount = m_cfg.n_fft / 2;
        size_t padded_length = audio_len + 2 * pad_amount;
        size_t n_frames = 1 + (padded_length - m_cfg.n_fft) / m_cfg.n_window_stride;
        
        // Pad to multiple of pad_to if required
        if (m_cfg.pad_to > 0) {
            size_t pad_amt = n_frames % m_cfg.pad_to;
            if (pad_amt != 0) {
                n_frames = n_frames + (m_cfg.pad_to - pad_amt);
            }
        }
        
        size_t n_features = linear_spec ? (1 + m_cfg.n_fft / 2) : m_cfg.features;
        return {
            .n_dims = 2,
            .dims = {n_features, n_frames}
        };
    }

    // Compute DFT for a single frame using precomputed sin/cos tables
    // Uses pre-allocated m_real_out and m_imag_out buffers
    void dft_frame(const std::span<float> frame, size_t n_fft) {
        assert(frame.size() == n_fft);
        size_t n_bins = 1 + n_fft / 2;

        for (size_t k = 0; k < n_bins; k++) {
            float real_sum = 0.0f;
            float imag_sum = 0.0f;
            for (size_t n = 0; n < n_fft; n++) {
                float sample = frame[n];
                // Use precomputed sin/cos: angle = 2*pi*k*n/n_fft
                size_t idx = (k * n) % n_fft;
                real_sum += sample * m_cos_vals[idx];
                imag_sum -= sample * m_sin_vals[idx];
            }
            m_real_out[k] = real_sum;
            m_imag_out[k] = imag_sum;
        }
    }

    // Compute STFT and return magnitude spectrogram
    // Output shape: [n_bins, n_frames] where n_bins = 1 + n_fft/2
    // Uses pre-allocated m_spectrogram, m_frame, m_real_out, m_imag_out buffers
    void stft_magnitude(const std::vector<float>& audio, size_t n_frames) {
        size_t n_fft = m_cfg.n_fft;
        size_t hop_length = m_cfg.n_window_stride;
        size_t win_length = m_cfg.n_window_size;
        size_t n_bins = 1 + n_fft / 2;

        // Calculate number of frames (center=True padding)
        size_t pad_amount = n_fft / 2;

        // Ensure spectrogram columns are sized correctly
        for (size_t k = 0; k < n_bins; k++) {
            if (m_spectrogram[k].size() < n_frames) {
                m_spectrogram[k].resize(n_frames);
            }
        }

        std::span<float> frame_span(m_frame.data(), m_frame.size());
        for (size_t t = 0; t < n_frames; t++) {
            size_t start = t * hop_length - pad_amount;

            // Extract and window the frame
            for (size_t i = 0; i < n_fft; i++) {
                size_t idx = start + i;
                float sample = 0.0f;
                if (idx < audio.size()) {
                    sample = audio[idx];
                }
                // Apply window (window is of size win_length, pad with zeros if n_fft > win_length)
                size_t win_offset = (n_fft - win_length) / 2;
                if (i >= win_offset && i < win_offset + win_length) {
                    sample *= m_window[i - win_offset];
                } else {
                    sample = 0.0f;
                }
                m_frame[i] = sample;
            }
            // Compute DFT (results stored in m_real_out, m_imag_out)
            dft_frame(frame_span, n_fft);

            // Compute magnitude
            for (size_t k = 0; k < n_bins; k++) {
                m_spectrogram[k][t] = sqrtf(m_real_out[k] * m_real_out[k] + m_imag_out[k] * m_imag_out[k]);
            }
        }
    }

    // Apply mel filterbank to spectrogram (stored in m_spectrogram)
    // Output: [n_mels, n_frames] written to mel_spec via mdspan2d
    void apply_filterbank(size_t n_frames, Tensor& mel_spec) {
        size_t n_mels = m_cfg.features;
        size_t n_bins = m_spectrogram.size();

        for (size_t m = 0; m < n_mels; m++) {
            for (size_t t = 0; t < n_frames; t++) {
                float sum = 0.0f;
                for (size_t k = 0; k < n_bins; k++) {
                    sum += m_filterbank(m, k) * m_spectrogram[k][t];
                }
                mel_spec(m, t) = sum;
            }
        }
    }

    // Forward pass: convert audio waveform to mel spectrogram features
    // Input: audio samples (span), seq_len (valid audio length)
    // Output: mel spectrogram written to mel_spec_data (contiguous, row-major [features x frames])
    // Returns Shape with dimensions and valid frame count
    size_t forward(
        std::span<float> audio,
        Tensor& mel_spec,
        bool linear_spec = false
    ) {
        size_t audio_len = audio.size();
        
        // Calculate output shape and resize buffer
        Shape shape = get_output_shape(audio_len, linear_spec);
        size_t n_frames = shape.dims[1];
        size_t out_seq_len = get_seq_len(audio_len);

        // Resize output buffer to exact size needed
        mel_spec.resize(shape);

        if (audio_len == 0) {
            mel_spec.fill(m_cfg.pad_value);
            return 0;
        }

        // Copy audio to internal buffer and apply pre-emphasis
        if (m_audio_buf.size() < audio_len) {
            m_audio_buf.resize(audio_len);
        }
        
        // Apply pre-emphasis while copying: y[n] = x[n] - preemph * x[n-1]
        if (m_cfg.preemph != 0.0f) {
            m_audio_buf[0] = audio[0];
            for (size_t i = 1; i < audio_len; i++) {
                m_audio_buf[i] = audio[i] - m_cfg.preemph * audio[i - 1];
            }
        } else {
            for (size_t i = 0; i < audio_len; i++) {
                m_audio_buf[i] = (i < audio_len) ? audio[i] : 0.0f;
            }
        }

        // Calculate n_frames before padding for STFT
        size_t pad_amount = m_cfg.n_fft / 2;
        size_t padded_length = audio_len + 2 * pad_amount;
        size_t stft_n_frames = 1 + (padded_length - m_cfg.n_fft) / m_cfg.n_window_stride;

        // Compute STFT magnitude spectrogram (stored in m_spectrogram)
        stft_magnitude(m_audio_buf, stft_n_frames);

        // Get power spectrum (raise magnitude to mag_power)
        size_t n_bins = m_spectrogram.size();
        if (m_cfg.mag_power == 2.0f) {
            // Optimize for power spectrogram
            for (size_t k = 0; k < n_bins; k++) {
                for (size_t t = 0; t < stft_n_frames; t++) {
                    float mag = m_spectrogram[k][t];
                    m_spectrogram[k][t] = mag * mag;
                }
            }
        } else if (m_cfg.mag_power != 1.0f) {
            for (size_t k = 0; k < n_bins; k++) {
                for (size_t t = 0; t < stft_n_frames; t++) {
                    m_spectrogram[k][t] = powf(m_spectrogram[k][t], m_cfg.mag_power);
                }
            }
        }

        // Return linear spectrogram if requested
        if (linear_spec) {
            for (size_t k = 0; k < n_bins; k++) {
                for (size_t t = 0; t < stft_n_frames; t++) {
                    mel_spec(k, t) = m_spectrogram[k][t];
                }
                // Pad remaining frames with pad_value
                for (size_t t = stft_n_frames; t < n_frames; t++) {
                    mel_spec(k, t) = m_cfg.pad_value;
                }
            }
            return out_seq_len;
        }

        // Apply mel filterbank (writes to mel_spec)
        apply_filterbank(stft_n_frames, mel_spec);

        // Apply log if required
        size_t n_mels = m_cfg.features;
        if (m_cfg.log) {
            if (m_cfg.log_zero_guard_type == LOG_ZERO_GUARD_ADD) {
                for (size_t m = 0; m < n_mels; m++) {
                    for (size_t t = 0; t < stft_n_frames; t++) {
                        mel_spec(m, t) = logf(mel_spec(m, t) + m_cfg.log_zero_guard_value);
                    }
                }
            } else { // LOG_ZERO_GUARD_CLAMP
                for (size_t m = 0; m < n_mels; m++) {
                    for (size_t t = 0; t < stft_n_frames; t++) {
                        mel_spec(m, t) = logf(std::max(mel_spec(m, t), m_cfg.log_zero_guard_value));
                    }
                }
            }
        }

        // Mask values beyond out_seq_len with pad_value (includes padding frames)
        for (size_t m = 0; m < n_mels; m++) {
            for (size_t t = out_seq_len; t < n_frames; t++) {
                mel_spec(m, t) = m_cfg.pad_value;
            }
        }

        return out_seq_len;
    }
};

#ifndef NMS_LIB_BUILD
int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    PreprocessorConfig &cfg = nemotron_speech_preprocessor;
    auto preprocessor = Preprocessor(cfg);
    preprocessor.load_weights("featurizer.window.bin", "featurizer.fb.bin");

    // Test forward pass with a simple sine wave
    printf("\nTesting forward pass with a sine wave...\n");
    size_t audio_len = 400;  // 1 second at 16kHz
    std::vector<float> audio(audio_len);
    for (size_t i = 0; i < audio_len; i++) {
        // 440 Hz sine wave
        audio[i] = sinf(2.0f * M_PI * 440.0f * i / cfg.sample_rate);
    }
    
    // Pre-allocate mel_spec buffer (contiguous, can be reused across calls)
    Tensor mel_spec;
    std::span<float> audio_span(audio.data(), audio.size());
    
    // Get output shape before running (useful for pre-allocation)
    size_t valid_frames = preprocessor.forward(audio_span, mel_spec);
    printf("Mel spectrogram shape: %zu x %zu (valid frames: %zu)\n", 
           mel_spec.shape.dims[0], mel_spec.shape.dims[1], valid_frames);
    
    // Print first few frames of first few mel bins
    printf("First 5 frames of first 5 mel bins:\n");
    for (size_t i = 0; i < std::min((size_t)5, mel_spec.shape.dims[0]); i++) {
        for (size_t j = 0; j < std::min((size_t)5, mel_spec.shape.dims[1]); j++) {
            printf("%0.4f ", mel_spec(i, j));
        }
        printf("\n");
    }

    // Demonstrate buffer reuse - second call should not allocate (same size)
    printf("\nSecond forward pass (reusing buffers)...\n");
    valid_frames = preprocessor.forward(audio_span, mel_spec);
    printf("Mel spectrogram shape: %zu x %zu (valid frames: %zu)\n", 
           mel_spec.shape.dims[0], mel_spec.shape.dims[1], valid_frames);
    
    return 0;
}
#endif