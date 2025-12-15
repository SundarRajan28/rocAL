/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "meta_data/yolo_label_meta_data_reader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <cmath>

using namespace std;

YoloLabelMetaDataReader::YoloLabelMetaDataReader() : _yolo_label_metadata_read_time("yolo label meta read time", DBG_TIMING) {
}

void YoloLabelMetaDataReader::init(const MetaDataConfig &cfg, pMetaDataBatch meta_data_batch) {
    _labels_path = cfg.path();
    _images_path = cfg.images_path();
    _avoid_class_remapping = cfg.class_remapping();
    this->set_aspect_ratio_grouping(cfg.get_aspect_ratio_grouping());
    _output = meta_data_batch;
    _output->set_metadata_type(cfg.type());
}

std::string YoloLabelMetaDataReader::normalize_key(const std::string& image_name) {
    std::string key = image_name;
    auto last_slash = key.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        key = key.substr(last_slash + 1);
    }
    auto dot_pos = key.find_last_of('.');
    if (dot_pos != std::string::npos) {
        key = key.substr(0, dot_pos);
    }
    return key;
}

bool YoloLabelMetaDataReader::exists(const std::string &image_name) {
    std::string key = normalize_key(image_name);
    return _map_content.find(key) != _map_content.end();
}

ImgSize YoloLabelMetaDataReader::lookup_image_size(const std::string &image_name) {
    std::string key = normalize_key(image_name);
    auto it = _map_img_sizes.find(key);
    if (it == _map_img_sizes.end())
        THROW("ERROR: Given name not present in the map " + image_name)
    return it->second;
}

void YoloLabelMetaDataReader::lookup(const std::vector<std::string> &image_names) {
    if (image_names.empty()) {
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    for (unsigned i = 0; i < image_names.size(); i++) {
        std::string key = normalize_key(image_names[i]);
        auto it = _map_content.find(key);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map " + image_names[i])
        _output->get_bb_cords_batch()[i] = it->second->get_bb_cords();
        _output->get_labels_batch()[i] = it->second->get_labels();
        _output->get_img_sizes_batch()[i] = it->second->get_img_size();
        _output->get_image_id_batch()[i] = it->second->get_image_id();
        if (_output->get_metadata_type() == MetaDataType::PolygonMask) {
            _output->get_mask_cords_batch()[i] = it->second->get_mask_cords();
            _output->get_mask_polygons_count_batch()[i] = it->second->get_polygon_count();
            _output->get_mask_vertices_count_batch()[i] = it->second->get_vertices_count();
        }
    }
}

void YoloLabelMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, Labels bb_labels, ImgSize image_size, MaskCords mask_cords, std::vector<int> polygon_count, std::vector<std::vector<int>> vertices_count, int image_id) {
    if (exists(image_name)) {
        std::string key = normalize_key(image_name);
        auto it = _map_content.find(key);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_labels().push_back(bb_labels[0]);
        it->second->get_mask_cords().insert(it->second->get_mask_cords().end(), mask_cords.begin(), mask_cords.end());
        it->second->get_polygon_count().push_back(polygon_count[0]);
        it->second->get_vertices_count().push_back(vertices_count[0]);
        return;
    }
    std::string key = normalize_key(image_name);
    pMetaDataPolygonMask info = std::make_shared<PolygonMask>(bb_coords, bb_labels, image_size, mask_cords, polygon_count, vertices_count, image_id);
    _map_content.insert(pair<std::string, std::shared_ptr<PolygonMask>>(key, info));
}

void YoloLabelMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, Labels bb_labels, ImgSize image_size, int image_id) {
    if (exists(image_name)) {
        std::string key = normalize_key(image_name);
        auto it = _map_content.find(key);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_labels().push_back(bb_labels[0]);
        return;
    }
    std::string key = normalize_key(image_name);
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels, image_size, image_id);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(key, info));
}

BoundingBoxCord YoloLabelMetaDataReader::convert_yolo_to_ltrb(float x_center, float y_center, float width, float height,
                                                              int img_width, int img_height) {
    float x_center_px = x_center * img_width;
    float y_center_px = y_center * img_height;
    float w_px = width * img_width;
    float h_px = height * img_height;

    BoundingBoxCord box;
    box.l = x_center_px - w_px / 2.0f;
    box.t = y_center_px - h_px / 2.0f;
    box.r = x_center_px + w_px / 2.0f;
    box.b = y_center_px + h_px / 2.0f;

    box.l = std::max(0.0f, box.l);
    box.t = std::max(0.0f, box.t);
    box.r = std::min(static_cast<float>(img_width), box.r);
    box.b = std::min(static_cast<float>(img_height), box.b);

    return box;
}

BoundingBoxCord YoloLabelMetaDataReader::compute_bbox_from_polygon(const std::vector<float>& polygon_coords,
                                                                   int img_width, int img_height) {
    if (polygon_coords.size() < 4) {
        return BoundingBoxCord{0, 0, 0, 0};
    }

    float min_x = polygon_coords[0] * img_width;
    float max_x = min_x;
    float min_y = polygon_coords[1] * img_height;
    float max_y = min_y;

    for (size_t i = 2; i < polygon_coords.size(); i += 2) {
        float x = polygon_coords[i] * img_width;
        float y = polygon_coords[i + 1] * img_height;
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }

    BoundingBoxCord box;
    box.l = std::max(0.0f, min_x);
    box.t = std::max(0.0f, min_y);
    box.r = std::min(static_cast<float>(img_width), max_x);
    box.b = std::min(static_cast<float>(img_height), max_y);

    return box;
}

MaskCords YoloLabelMetaDataReader::convert_mask_to_pixel(const MaskCords& norm_coords, int img_width, int img_height) {
    MaskCords pixel_coords;
    pixel_coords.reserve(norm_coords.size());
    for (size_t i = 0; i < norm_coords.size(); i += 2) {
        pixel_coords.push_back(norm_coords[i] * img_width);
        if (i + 1 < norm_coords.size()) {
            pixel_coords.push_back(norm_coords[i + 1] * img_height);
        }
    }
    return pixel_coords;
}

filesys::path YoloLabelMetaDataReader::find_image_path(const std::string& basename) {
    // YOLO label reader supports only JPEG images.
    static const std::vector<std::string> extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"};
    for (const auto& ext : extensions) {
        filesys::path candidate = filesys::path(_images_path) / (basename + ext);
        if (filesys::exists(candidate)) {
            return candidate;
        }
    }
    return filesys::path();
}

ImgSize YoloLabelMetaDataReader::probe_image_size(const filesys::path& image_path) {
    std::ifstream file(image_path, std::ios::binary);
    if (!file.is_open()) {
        THROW("Cannot open image file: " + image_path.string())
    }

    unsigned char header[4];
    file.read(reinterpret_cast<char*>(header), 4);
    size_t bytes_read = file.gcount();

    if (bytes_read < 2) {
        THROW("Image file too small: " + image_path.string())
    }

    // Metadata pipeline currently supports only JPEG images for YOLO label reader.
    // Verify JPEG SOI marker (0xFF, 0xD8) before parsing the header.
    if (!(header[0] == 0xFF && header[1] == 0xD8)) {
        THROW("Unsupported image format for YOLO label reader (only JPEG is supported): " + image_path.string())
    }

    ImgSize size{0, 0};

    // Parse SOF marker to extract width/height, without applying EXIF orientation.
    // This keeps dimensions consistent with turbojpeg's tjDecompressHeader2 / decode_info.
    file.seekg(2, std::ios::beg);
    unsigned char buf[12];
    while (file.read(reinterpret_cast<char*>(buf), 4)) {
        if (buf[0] != 0xFF)
            break;
        unsigned char marker = buf[1];
        int length = (buf[2] << 8) | buf[3];
        if (length < 2) {
            THROW("Invalid JPEG segment length in: " + image_path.string())
        }

        // Accept all Start Of Frame markers that carry size info (C0-CF) except non-SOF markers like DHT/DAC.
        bool is_sof = (marker >= 0xC0 && marker <= 0xCF) && (marker != 0xC4) && (marker != 0xC8) && (marker != 0xCC);
        if (is_sof) {
            if (!file.read(reinterpret_cast<char*>(buf), 5)) {
                THROW("Unexpected EOF while reading JPEG SOF segment: " + image_path.string())
            }
            size.h = (buf[1] << 8) | buf[2];
            size.w = (buf[3] << 8) | buf[4];
            break;
        }
        file.seekg(length - 2, std::ios::cur);
    }

    if (size.w <= 0 || size.h <= 0) {
        THROW("Could not determine JPEG image dimensions: " + image_path.string())
    }

    return size;
}

void YoloLabelMetaDataReader::parse_label_file(const filesys::path& label_path, const std::string& image_key, ImgSize image_size) {
    std::ifstream file(label_path);
    if (!file.is_open()) {
        WRN("Cannot open label file: " + label_path.string())
        return;
    }

    BoundingBoxCords bb_coords;
    Labels bb_labels;
    MaskCords all_mask_cords;
    std::vector<int> polygon_count;
    std::vector<std::vector<int>> vertices_count;

    std::string line;
    int line_num = 0;
    int warn_count = 0;
    const int max_warnings_per_file = 3;

    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        std::istringstream iss(line);
        std::vector<float> tokens;
        float val;
        while (iss >> val) {
            tokens.push_back(val);
        }

        if (tokens.size() < 5) {
            if (warn_count < max_warnings_per_file) {
                WRN("Malformed line " + std::to_string(line_num) + " in " + label_path.string() + ": not enough tokens")
                warn_count++;
            }
            continue;
        }

        int class_id = static_cast<int>(tokens[0]);
        _observed_class_ids.insert(class_id);

        if (tokens.size() == 5) {
            float x_center = std::max(0.0f, std::min(1.0f, tokens[1]));
            float y_center = std::max(0.0f, std::min(1.0f, tokens[2]));
            float w = std::max(0.0f, std::min(1.0f, tokens[3]));
            float h = std::max(0.0f, std::min(1.0f, tokens[4]));

            BoundingBoxCord box = convert_yolo_to_ltrb(x_center, y_center, w, h, image_size.w, image_size.h);
            bb_coords.push_back(box);
            bb_labels.push_back(class_id);

            if (_output->get_metadata_type() == MetaDataType::PolygonMask) {
                polygon_count.push_back(0);
                vertices_count.push_back(std::vector<int>());
            }
        } else if ((tokens.size() - 1) % 2 == 0 && tokens.size() >= 7) {
            std::vector<float> norm_polygon_coords(tokens.begin() + 1, tokens.end());

            for (auto& coord : norm_polygon_coords) {
                coord = std::max(0.0f, std::min(1.0f, coord));
            }

            BoundingBoxCord box = compute_bbox_from_polygon(norm_polygon_coords, image_size.w, image_size.h);
            bb_coords.push_back(box);
            bb_labels.push_back(class_id);

            if (_output->get_metadata_type() == MetaDataType::PolygonMask) {
                MaskCords pixel_coords = convert_mask_to_pixel(
                    MaskCords(norm_polygon_coords.begin(), norm_polygon_coords.end()),
                    image_size.w, image_size.h);
                all_mask_cords.insert(all_mask_cords.end(), pixel_coords.begin(), pixel_coords.end());
                polygon_count.push_back(1);
                vertices_count.push_back(std::vector<int>{static_cast<int>(pixel_coords.size())});
            }
        } else {
            if (warn_count < max_warnings_per_file) {
                WRN("Invalid annotation format at line " + std::to_string(line_num) + " in " + label_path.string())
                warn_count++;
            }
        }
    }

    if (bb_coords.empty()) {
        bb_coords = BoundingBoxCords();
        bb_labels = Labels();
    }

    if (_output->get_metadata_type() == MetaDataType::PolygonMask) {
        if (bb_coords.empty()) {
            polygon_count = std::vector<int>();
            vertices_count = std::vector<std::vector<int>>();
            all_mask_cords = MaskCords();
        }
        add(image_key, bb_coords, bb_labels, image_size, all_mask_cords, polygon_count, vertices_count, 0);
    } else {
        add(image_key, bb_coords, bb_labels, image_size, 0);
    }
}

void YoloLabelMetaDataReader::read_all(const std::string &path) {
    _yolo_label_metadata_read_time.start();

    if (!filesys::exists(path) || !filesys::is_directory(path)) {
        THROW("Labels directory does not exist: " + path)
    }

    int files_processed = 0;
    int files_skipped = 0;

    for (const auto& entry : filesys::directory_iterator(path)) {
        if (!filesys::is_regular_file(entry)) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".txt") continue;

        std::string basename = entry.path().stem().string();

        filesys::path image_path = find_image_path(basename);
        if (image_path.empty()) {
            WRN("No matching image found for label file: " + entry.path().string())
            files_skipped++;
            continue;
        }

        ImgSize image_size;
        try {
            image_size = probe_image_size(image_path);
        } catch (const std::exception& e) {
            WRN("Failed to probe image size for " + image_path.string() + ": " + e.what())
            files_skipped++;
            continue;
        }

        _map_img_sizes[basename] = image_size;
        _relative_file_paths.push_back(image_path.filename().string());

        parse_label_file(entry.path(), basename, image_size);
        files_processed++;
    }

    if (!_avoid_class_remapping && !_observed_class_ids.empty()) {
        int continuous_idx = 1;
        for (int id : _observed_class_ids) {
            _label_info[id] = continuous_idx++;
        }

        for (auto& elem : _map_content) {
            Labels& bb_labels = elem.second->get_labels();
            Labels continuous_labels;
            for (int label : bb_labels) {
                auto it = _label_info.find(label);
                if (it != _label_info.end()) {
                    continuous_labels.push_back(it->second);
                } else {
                    continuous_labels.push_back(label);
                }
            }
            elem.second->set_labels(continuous_labels);
        }
    }

    _yolo_label_metadata_read_time.end();
}

void YoloLabelMetaDataReader::print_map_contents() {
    BoundingBoxCords bb_coords;
    Labels bb_labels;
    ImgSize img_size;
    MaskCords mask_cords;
    std::vector<int> polygon_size;
    std::vector<std::vector<int>> vertices_count;

    std::cout << "\nBBox Annotations List (YOLO format): \n";
    for (auto &elem : _map_content) {
        std::cout << "\nName :\t " << elem.first;
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_labels();
        img_size = elem.second->get_img_size();
        std::cout << "<wxh, num of bboxes>: " << img_size.w << " X " << img_size.h << " , " << bb_coords.size() << std::endl;
        for (unsigned int i = 0; i < bb_coords.size(); i++) {
            std::cout << " l : " << bb_coords[i].l << " t: :" << bb_coords[i].t << " r : " << bb_coords[i].r << " b: :" << bb_coords[i].b << " Label Id : " << bb_labels[i] << std::endl;
        }
        if (_output->get_metadata_type() == MetaDataType::PolygonMask) {
            int count = 0;
            mask_cords = elem.second->get_mask_cords();
            polygon_size = elem.second->get_polygon_count();
            vertices_count = elem.second->get_vertices_count();
            std::cout << "\nNumber of objects : " << bb_coords.size() << std::endl;
            for (unsigned int i = 0; i < bb_coords.size(); i++) {
                std::cout << "\nNumber of polygons for object[" << i << "]:" << polygon_size[i];
                for (int j = 0; j < polygon_size[i]; j++) {
                    std::cout << "\nPolygon size :" << vertices_count[i][j] << " Elements::";
                    for (int k = 0; k < vertices_count[i][j]; k++, count++)
                        std::cout << "\t " << mask_cords[count];
                }
            }
        }
    }
}

void YoloLabelMetaDataReader::release(std::string image_name) {
    std::string key = normalize_key(image_name);
    auto it = _map_content.find(key);
    if (it == _map_content.end()) {
        WRN("ERROR: Given name not present in the map " + image_name);
        return;
    }
    _map_content.erase(it);
}

void YoloLabelMetaDataReader::release() {
    _map_content.clear();
    _map_img_sizes.clear();
    _relative_file_paths.clear();
}
