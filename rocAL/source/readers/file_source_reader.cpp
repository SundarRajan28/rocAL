/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cassert>
#include <algorithm>
#include <cstring>
#include <math.h>
#include "pipeline/commons.h"
#include "readers/file_source_reader.h"
#include "pipeline/filesystem.h"

FileSourceReader::FileSourceReader() {
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
    _file_id = 0;
    _shuffle = false;
    _file_count_all_shards = 0;
}

unsigned FileSourceReader::count_items() {
  if (_loop) return shard_size_with_padding();
std::cerr << "\n COUNT ITEMS: ";
  int size = std::max(shard_size_with_padding(), _batch_count);

  int ret = (size - _read_counter);
  if (_last_batch_info.first == RocalBatchPolicy::PARTIAL) {
    ret += (_batch_count -
            (shard_size_with_padding() - shard_size_without_padding()));
  } else if (_last_batch_info.first == RocalBatchPolicy::FILL) {
    ret += (_batch_count -
            (shard_size_with_padding() - shard_size_without_padding()));
  }
  std::cerr << "\n ret value is " << ret;
  return ((ret < 0) ? 0 : ret);
}

Reader::Status FileSourceReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _file_id = 0;
    _folder_path = desc.path();
    _file_list_path = desc.file_list_path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    _meta_data_reader = desc.meta_data_reader();
    _last_batch_info = desc.get_last_batch_policy();
    _pad_last_batch = _last_batch_info.second; // Currently last_batch_padded and pad_last_batch are same
    _stick_to_shard = desc.get_stick_to_shard();
    _shard_size = desc.get_shard_size();
    ret = subfolder_reading();
    _curr_file_idx = get_start_idx();
    std::cerr << "\n The current file index is " << _curr_file_idx << "\t for shard_id" << _shard_id;
    // shuffle dataset if set
    if (ret == Reader::Status::OK && _shuffle)
        std::random_shuffle(_file_names.begin(), _file_names.end());

    return ret;
}

void FileSourceReader::incremenet_read_ptr() {
    _read_counter++;
    std::cerr << "\n _read_counter:: " << _read_counter;
    // For DROP policy, when the pad is set to false the last batch is dropped and idx is advanced by skipping one batch in between
    if (_last_batch_info.first == RocalBatchPolicy::DROP && _pad_last_batch == false) {
        if (_read_counter + _batch_count - 1 > shard_size_without_padding()) {
            // Skip last batch entirely along with padded data idx
            if (_stick_to_shard == true) 
                _curr_file_idx = ((_curr_file_idx + _batch_count + 1) % shard_size_without_padding()) + _shard_start_idx;
            else 
                _curr_file_idx = (_curr_file_idx + _batch_count + 1) % _file_names.size();
            return;
        }
    } else if (_pad_last_batch == true && _read_counter >= shard_size_without_padding()) {   // Return last file_idx when pad set to true
            _curr_file_idx = _shard_start_idx + (shard_size_without_padding() - 1);
            return;
    }

    if (_stick_to_shard == true) {
        if ((_shard_start_idx <= (_curr_file_idx + 1)) && ((_curr_file_idx + 1) <= (_shard_start_idx + shard_size_without_padding() - 1))) 
            _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
        else if ((_curr_file_idx + 1) == (_shard_start_idx + shard_size_without_padding()))  // Last Element in the shard
            _curr_file_idx = _shard_start_idx;
    } else {
      _curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
    }
    std::cerr << " \n \n _curr_file_idx !!! :: "<< _curr_file_idx;
}

size_t FileSourceReader::open() {
    std::cerr << "\n _curr_file_idx :: " << _curr_file_idx << "\t " << "_file_names[_curr_file_idx] :: " << _file_names[_curr_file_idx];
    auto file_path = _file_names[_curr_file_idx];  // Get next file name
    incremenet_read_ptr();
    _last_file_path = _last_id = file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        _last_id.erase(0, last_slash_idx + 1);
    }

    _current_fPtr = fopen(file_path.c_str(), "rb");  // Open the file,

    if (!_current_fPtr)  // Check if it is ready for reading
        return 0;

    fseek(_current_fPtr, 0, SEEK_END);  // Take the file read pointer to the end

    _current_file_size = ftell(_current_fPtr);  // Check how many bytes are there between and the current read pointer position (end of the file)

    if (_current_file_size == 0) {  // If file is empty continue
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        return 0;
    }

    fseek(_current_fPtr, 0, SEEK_SET);  // Take the file pointer back to the start

    return _current_file_size;
}

size_t FileSourceReader::read_data(unsigned char* buf, size_t read_size) {
    if (!_current_fPtr)
        return 0;

    // Requested read size bigger than the file size? just read as many bytes as the file size
    read_size = (read_size > _current_file_size) ? _current_file_size : read_size;

    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
    return actual_read_size;
}

int FileSourceReader::close() {
    return release();
}

FileSourceReader::~FileSourceReader() {
    release();
}

int FileSourceReader::release() {
    if (!_current_fPtr)
        return 0;
    fclose(_current_fPtr);
    _current_fPtr = nullptr;
    return 0;
}

void FileSourceReader::reset() {
    if (_shuffle) std::random_shuffle(_file_names.begin(), _file_names.end());
    if ( _shard_count == 1 && _shard_size > 0) { // Single Shard
        _read_counter=0;
        _curr_file_idx =0;
        int size = _shard_size;
        int inc = 0;
        if (size < _batch_count) {
            inc = _batch_count;
        } else {
            if(size % _batch_count)
                inc = size;
            else
                inc = size + (size%_batch_count);
        }
            std::rotate(_file_names.begin(), _file_names.begin() + inc, _file_names.end());
    }
    else if (_shard_count > 1) { // Multiple Shards
        if(_stick_to_shard == false) {
            increment_shard_id();
            _last_batch_padded_size = 0;
            // _curr_file_idx;
            _file_id = 0;
            _read_counter = 0;
            get_start_idx();
            _padded_samples = shard_size_with_padding() - shard_size_without_padding();
            if (_padded_samples > 0 && _padded_samples < _batch_count) {
                replicate_last_image_to_fill_last_shard();
                LOG("FileReader in reset - Replicated " + _folder_path + _last_file_name + " " + TOSTR((_batch_count - _in_batch_read_count)) + " times to fill the last batch")
            }
            if (!_file_names.empty())
                LOG("FileReader in reset - Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path)
        }
        if (!_file_names.empty())
                LOG("FileReader in reset function - Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path)
    } else {
        if (_pad_last_batch == false) return;       // If padding is not set, need to continue the file_idx
        _curr_file_idx = _shard_start_idx;
        _read_counter = 0;
    }
}

void FileSourceReader::increment_shard_id() {
    _shard_id = (_shard_id + 1) % _shard_count;
}

Reader::Status FileSourceReader::generate_file_names() {
    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;

    while ((_entity = readdir(_sub_dir)) != nullptr) {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
    }
    closedir(_sub_dir);
    std::sort(entry_name_list.begin(), entry_name_list.end());

    auto ret = Reader::Status::OK;
    if (!_file_list_path.empty()) {  // Reads the file paths from the file list and adds to file_names vector for decoding
        if (_meta_data_reader) {
            auto vec_rel_file_path = _meta_data_reader->get_file_path_content(); // Get the relative file path's from meta_data_reader
            for(auto file_path: vec_rel_file_path) {
                if (filesys::path(file_path).is_relative()) {  // Only add root path if the file list contains relative file paths
                    if (!filesys::exists(_folder_path))
                        THROW("File list contains relative paths but root path doesn't exists");
                    _absolute_file_path = _folder_path + "/" + file_path;
                }
                if (filesys::is_regular_file(_absolute_file_path)) {
                    // if (get_file_shard_id() != _shard_id) { // Change sharding
                    //     _file_count_all_shards++;
                    //     incremenet_file_id();
                    //     continue;
                    // }
                    // _in_batch_read_count++;
                    // _in_batch_read_count = (_in_batch_read_count % _batch_count == 0) ? 0 : _in_batch_read_count;
                    // std::cerr << "\n _in_batch_read_count - !_file_list_path.empty(): " << _in_batch_read_count;
                    _last_file_name = _absolute_file_path;
                    _file_names.push_back(_absolute_file_path);
                    _file_count_all_shards++;
                    incremenet_file_id();
                }
            }
        } else {
            std::ifstream fp(_file_list_path);
            if (fp.is_open()) {
                while (fp) {
                    std::string file_label_path;
                    std::getline(fp, file_label_path);
                    std::istringstream ss(file_label_path);
                    std::string file_path;
                    std::getline(ss, file_path, ' ');
                    if (filesys::path(file_path).is_relative()) {  // Only add root path if the file list contains relative file paths
                        if (!filesys::exists(_folder_path))
                            THROW("File list contains relative paths but root path doesn't exists");
                        file_path = _folder_path + "/" + file_path;
                    }
                    std::string file_name = file_path.substr(file_path.find_last_of("/\\") + 1);

                    if (filesys::is_regular_file(file_path)) {
                        // if (get_file_shard_id() != _shard_id) { // Change sharding
                        //     _file_count_all_shards++;
                        //     incremenet_file_id();
                        //     continue;
                        // }
                        // _in_batch_read_count++;
                        // _in_batch_read_count = (_in_batch_read_count % _batch_count == 0) ? 0 : _in_batch_read_count;
                        _last_file_name = file_path;
                        _file_names.push_back(file_path);
                        _file_count_all_shards++;
                        incremenet_file_id();
                    }
                }
            }
        }
    } else {
        for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
            std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
            filesys::path pathObj(subfolder_path);
            if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
                // ignore files with unsupported extensions
                auto file_extension_idx = subfolder_path.find_last_of(".");
                if (file_extension_idx != std::string::npos) {
                    std::string file_extension = subfolder_path.substr(file_extension_idx + 1);
                    std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    if ((file_extension != "jpg") && (file_extension != "jpeg") && (file_extension != "png") && (file_extension != "ppm") && (file_extension != "bmp") && (file_extension != "pgm") && (file_extension != "tif") && (file_extension != "tiff") && (file_extension != "webp") && (file_extension != "wav"))
                        continue;
                }
                ret = open_folder();
                break;  // assume directory has only files.
            } else if (filesys::exists(pathObj) && filesys::is_directory(pathObj)) {
                _folder_path = subfolder_path;
                if (open_folder() != Reader::Status::OK)
                    WRN("FileReader ShardID [" + TOSTR(_shard_id) + "] File reader cannot access the storage at " + _folder_path);
            }
        }
    }

    if (_file_names.empty())
        WRN("FileReader ShardID [" + TOSTR(_shard_id) + "] Did not load any file from " + _folder_path)
    std::cerr << "\n shard id ::" << _shard_id;
    // uint images_to_pad_shard = shard_size_with_padding() - shard_size_without_padding();
    // std::cerr << "images_to_pad_shard: " << _shard_id << "\t "<< images_to_pad_shard;
    // if (!images_to_pad_shard) {
    //     for (uint i = 0; i < images_to_pad_shard; i++) {
    //         // if (get_file_shard_id() != _shard_id) { // Change sharding
    //         //     _file_count_all_shards++;
    //         //     incremenet_file_id();
    //         //     continue;
    //         // }
    //         // std::cerr << "\n file_names:" << _file_names.at(i);
    //         _last_file_name = _file_names.at(i);
    //         _file_names.push_back(_last_file_name);
    //         _file_count_all_shards++;
    //         incremenet_file_id();
    //     }
    // }

    return ret;
}

Reader::Status FileSourceReader::subfolder_reading() {
    auto ret = generate_file_names();
    auto _padded_samples = shard_size_with_padding() - shard_size_without_padding();
    if (_padded_samples > 0 && _padded_samples < _batch_count) {
        // This is to pad within a batch in a shard. Need to change this according to fill / drop or partial.
        // Adjust last batch only if the last batch padded is true.
        replicate_last_image_to_fill_last_shard();
        LOG("FileReader ShardID [" + TOSTR(_shard_id) + "] Replicated " + _folder_path + _last_file_name + " " + TOSTR((_batch_count - _in_batch_read_count)) + " times to fill the last batch")
    }

    if (!_file_names.empty())
        LOG("FileReader ShardID [" + TOSTR(_shard_id) + "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + STR(_folder_path))
    return ret;
}

void FileSourceReader::replicate_last_image_to_fill_last_shard() {
    if (_last_batch_info.first == RocalBatchPolicy::PARTIAL) {
        _last_batch_padded_size = _padded_samples;
    }
}

void FileSourceReader::replicate_last_batch_to_pad_partial_shard() {
    if (_file_names.size() >= _batch_count) {
        for (size_t i = 0; i < _batch_count; i++)
            _file_names.push_back(_file_names[i - _batch_count]);
    }
}

Reader::Status FileSourceReader::open_folder() {
    if ((_src_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        if (_entity->d_type != DT_REG)
            continue;

        std::string filename(_entity->d_name);
        auto file_extension_idx = filename.find_last_of(".");
        if (file_extension_idx != std::string::npos) {
            std::string file_extension = filename.substr(file_extension_idx + 1);
            std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if ((file_extension != "jpg") && (file_extension != "jpeg") && (file_extension != "png") && (file_extension != "ppm") && (file_extension != "bmp") && (file_extension != "pgm") && (file_extension != "tif") && (file_extension != "tiff") && (file_extension != "webp") && (file_extension != "wav"))
                continue;
        }
        if (!_meta_data_reader || _meta_data_reader->exists(_entity->d_name)) { // Check if the file is present in metadata reader and add to file names list, to avoid issues while lookup
            // if (get_file_shard_id() != _shard_id) { // Change sharding
            //     _file_count_all_shards++;
            //     incremenet_file_id();
            //     continue;
            // }
            // _in_batch_read_count++;
            // _in_batch_read_count = (_in_batch_read_count % _batch_count == 0) ? 0 : _in_batch_read_count;
            std::string file_path = _folder_path;
            file_path.append("/");
            file_path.append(_entity->d_name);
            _file_names.push_back(file_path);
            _file_count_all_shards++;
            incremenet_file_id();

            // uint images_to_pad_shard = shard_size_with_padding() - shard_size_without_padding();
            // if (!images_to_pad_shard) {
            //   for (int i = 0; i < images_to_pad_shard; i++) {
            //     // if (get_file_shard_id() != _shard_id) { // Change sharding
            //     //   _file_count_all_shards++;
            //     //   incremenet_file_id();
            //     //   continue;
            //     // }
            //     _last_file_name = _file_names.at(i);
            //     _file_names.push_back(_last_file_name);
            //     _file_count_all_shards++;
            //     incremenet_file_id();
            //   }
            // }

        } else {
            WRN("Skipping file," + _entity->d_name + " as it is not present in metadata reader")
        }
    }
    if (_file_names.empty())
        ERR("FileReader ShardID [" + TOSTR(_shard_id) + "] Did not load any file from " + _folder_path)
    std::sort(_file_names.begin(), _file_names.end());
    _last_file_name = _file_names[_file_names.size() - 1];

    closedir(_src_dir);
    return Reader::Status::OK;
}

size_t FileSourceReader::get_file_shard_id() {
    if (_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    return _file_id % _shard_count;
}

size_t FileSourceReader::last_batch_padded_size() {
    return _last_batch_padded_size;
}
std::string FileSourceReader::get_root_folder_path() {
    return _folder_path;
}

std::vector<std::string> FileSourceReader::get_file_paths_from_meta_data_reader() {   if (_meta_data_reader) {
        return _meta_data_reader->get_file_path_content();
    } else {
        std::clog << "\n Meta Data Reader is not initialized!";
        return {};
    }
}

size_t FileSourceReader::get_start_idx() {
    _shard_start_idx = get_dataset_size() * _shard_id / _shard_count;
    return _shard_start_idx;
}

size_t FileSourceReader::get_dataset_size() {
    return _file_count_all_shards;
}

size_t FileSourceReader::shard_size_without_padding() {
    return std::floor((_shard_id + 1) * get_dataset_size() / _shard_count) - floor(_shard_id * get_dataset_size() / _shard_count);
}

size_t FileSourceReader::shard_size_with_padding() {
  return std::ceil(get_dataset_size() * 1.0 / _shard_count);
}