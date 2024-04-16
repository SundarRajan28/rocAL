/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "audio_source_evaluator.h"

#include "audio_decoder_factory.hpp"
#include "reader_factory.h"
#include<algorithm>
#include <unordered_set>
#ifdef ROCAL_AUDIO

size_t AudioSourceEvaluator::GetMaxSamples() {
    return _samples_max;
}

size_t AudioSourceEvaluator::GetMaxChannels() {
    return _channels_max;
}

AudioSourceEvaluatorStatus
AudioSourceEvaluator::Create(ReaderConfig reader_cfg, DecoderConfig decoder_cfg) {
    AudioSourceEvaluatorStatus status = AudioSourceEvaluatorStatus::OK;
    // Can initialize it to any decoder types if needed
    _decoder = create_audio_decoder(std::move(decoder_cfg));
    _reader = create_reader(std::move(reader_cfg));
    FindMaxDimension();
    return status;
}

void AudioSourceEvaluator::FindMaxDimension() {
    _reader->reset();
    std::unordered_set<std::string> unique_file_paths;
    while (_reader->count_items()) {
        size_t fsize = _reader->open();
        if (!fsize) continue;
        std::string file_name = _reader->file_path();
        if (unique_file_paths.count(file_name) == 0) {
            std::clog << "\n File Name in Source eval : " << file_name;
            unique_file_paths.insert(file_name);
            if (_decoder->Initialize(file_name.c_str()) != AudioDecoder::Status::OK) {
                WRN("Could not initialize audio decoder for file : " + _reader->id())
                continue;
            }
            int samples, channels;
            float sample_rate;
            if (_decoder->DecodeInfo(&samples, &channels, &sample_rate) != AudioDecoder::Status::OK) {
                WRN("Could not decode the header of the: " + _reader->id())
                continue;
            }
            if (samples <= 0 || channels <= 0)
                continue;
            _samples_max = std::max(samples, _samples_max);
            _channels_max = std::max(channels, _channels_max);
            _decoder->Release();
        }
        _reader->close();
    }
    // return the reader read pointer to the begining of the resource
    _reader->reset();
}
#endif