#include "include/ggml_weights.h"

#include <cstring>
#include <stdexcept>

namespace nemo {

// Read a value from file
template <typename T>
static bool read_value(FILE* f, T& value) {
    return fread(&value, sizeof(T), 1, f) == 1;
}

// Read bytes from file
static bool read_bytes(FILE* f, void* data, size_t size) {
    return fread(data, 1, size, f) == size;
}

bool ModelWeights::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Failed to open weights file: %s\n", path.c_str());
        return false;
    }

    // Read and verify magic
    char magic[4];
    if (!read_bytes(f, magic, 4) || memcmp(magic, "NEMO", 4) != 0) {
        fprintf(stderr, "Invalid magic in weights file\n");
        fclose(f);
        return false;
    }

    // Read version
    uint32_t version;
    if (!read_value(f, version) || version != 1) {
        fprintf(stderr, "Unsupported weights file version: %u\n", version);
        fclose(f);
        return false;
    }

    // Read number of tensors
    uint32_t n_tensors;
    if (!read_value(f, n_tensors)) {
        fprintf(stderr, "Failed to read tensor count\n");
        fclose(f);
        return false;
    }

    // printf("Loading %u tensors from %s\n", n_tensors, path.c_str());

    // Read each tensor
    for (uint32_t i = 0; i < n_tensors; i++) {
        Tensor tensor;

        // Read name
        uint32_t name_len;
        if (!read_value(f, name_len)) {
            fprintf(stderr, "Failed to read tensor name length\n");
            fclose(f);
            return false;
        }

        tensor.name.resize(name_len);
        if (!read_bytes(f, tensor.name.data(), name_len)) {
            fprintf(stderr, "Failed to read tensor name\n");
            fclose(f);
            return false;
        }

        // Read dimensions
        uint32_t n_dims;
        if (!read_value(f, n_dims)) {
            fprintf(stderr, "Failed to read tensor dimensions\n");
            fclose(f);
            return false;
        }

        tensor.shape.resize(n_dims);
        size_t numel = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            uint32_t dim;
            if (!read_value(f, dim)) {
                fprintf(stderr, "Failed to read tensor dimension\n");
                fclose(f);
                return false;
            }
            tensor.shape[d] = dim;
            numel *= dim;
        }

        // Read dtype
        uint32_t dtype;
        if (!read_value(f, dtype)) {
            fprintf(stderr, "Failed to read tensor dtype\n");
            fclose(f);
            return false;
        }
        tensor.dtype = static_cast<DType>(dtype);

        // Read data
        tensor.data.resize(numel);
        if (tensor.dtype == DType::F32) {
            if (!read_bytes(f, tensor.data.data(), numel * sizeof(float))) {
                fprintf(stderr, "Failed to read tensor data: %s\n", tensor.name.c_str());
                fclose(f);
                return false;
            }
        } else if (tensor.dtype == DType::F16) {
            // Read as f16 and convert to f32
            std::vector<uint16_t> f16_data(numel);
            if (!read_bytes(f, f16_data.data(), numel * sizeof(uint16_t))) {
                fprintf(stderr, "Failed to read tensor data: %s\n", tensor.name.c_str());
                fclose(f);
                return false;
            }
            // Convert f16 to f32
            for (size_t j = 0; j < numel; j++) {
                uint16_t h = f16_data[j];
                uint32_t sign = (h >> 15) & 0x1;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;

                uint32_t f;
                if (exp == 0) {
                    if (mant == 0) {
                        f = sign << 31;
                    } else {
                        // Denormalized
                        exp = 1;
                        while (!(mant & 0x400)) {
                            mant <<= 1;
                            exp--;
                        }
                        mant &= 0x3FF;
                        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                    }
                } else if (exp == 31) {
                    f = (sign << 31) | 0x7F800000 | (mant << 13);
                } else {
                    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                memcpy(&tensor.data[j], &f, sizeof(float));
            }
        } else {
            fprintf(stderr, "Unknown dtype: %u\n", dtype);
            fclose(f);
            return false;
        }

        tensors_[tensor.name] = std::move(tensor);
    }

    fclose(f);
    // printf("Loaded %zu tensors, %zu total parameters\n", tensors_.size(), total_params());
    return true;
}

const Tensor* ModelWeights::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return nullptr;
    }
    return &it->second;
}

const Tensor& ModelWeights::require(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

bool ModelWeights::has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

std::vector<std::string> ModelWeights::names() const {
    std::vector<std::string> result;
    result.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) {
        result.push_back(name);
    }
    return result;
}

void ModelWeights::print_info() const {
    for (const auto& [_, tensor] : tensors_) {
        tensor.print_info();
    }
}

size_t ModelWeights::total_params() const {
    size_t total = 0;
    for (const auto& [_, tensor] : tensors_) {
        total += tensor.numel();
    }
    return total;
}

}  // namespace nemo
