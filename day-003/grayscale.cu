#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>
#include <vector_types.h>

#pragma pack(push, 1)
struct BMPFileHeader {
  uint16_t bfType;
  uint32_t bfSize;
  uint16_t bfReserved1;
  uint16_t bfReserved2;
  uint32_t bfOffBits;
};

struct BMPInfoHeader {
  uint32_t biSize;
  int32_t biWidth;
  int32_t biHeight;
  uint16_t biPlanes;
  uint16_t biBitCount;
  uint32_t biCompression;
  uint32_t biSizeImage;
  int32_t biXPelsPerMeter;
  int32_t biYPelsPerMeter;
  uint32_t biClrUsed;
  uint32_t biClrImportant;
};
#pragma pack(pop)

struct RGB {
  uint8_t b, g, r;
};

struct Image {
  int width;
  int height;
  std::vector<RGB> pixels;
};

__global__ void grayscale_kernel(RGB *pixels_in, RGB *pixels_out, int height,
                                 int width) {
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int x = threadIdx.x + blockDim.x * blockIdx.x;

  if (x < width && y < height) {
    int idx = y * width + x;
    RGB pixel_in = pixels_in[idx];

    uint8_t grayscale =
        (0.299 * pixel_in.r) + (0.587 * pixel_in.g) + (0.114 * pixel_in.b);
    pixels_out[idx].r = grayscale;
    pixels_out[idx].g = grayscale;
    pixels_out[idx].b = grayscale;
  }
}

void grayscale(Image &img_h) {
  size_t size_h = img_h.pixels.size() * sizeof(RGB);

  RGB *pixels_in_d, *pixels_out_d;
  cudaMalloc((void **)&pixels_in_d, size_h);
  cudaMalloc((void **)&pixels_out_d, size_h);

  cudaMemcpy(pixels_in_d, img_h.pixels.data(), size_h, cudaMemcpyHostToDevice);

  dim3 gridDim(((img_h.height + 15) / 16), (img_h.width + 15) / 16);
  dim3 blockDim(16, 16);
  grayscale_kernel<<<gridDim, blockDim>>>(pixels_in_d, pixels_out_d, img_h.height,
                                          img_h.width);

  cudaMemcpy(img_h.pixels.data(), pixels_out_d, size_h, cudaMemcpyDeviceToHost);
}

void write_bmp(const Image img, const std::string &filename) {
  int row_size = (3 * img.width + 3) & ~3;
  int image_size = row_size * img.height;
  int header_size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
  int full_size = header_size + image_size;

  BMPFileHeader file_header = {
      .bfType = 0x4D42,
      .bfSize = (uint32_t)full_size,
      .bfReserved1 = 0,
      .bfReserved2 = 0,
      .bfOffBits = (uint32_t)header_size,
  };

  BMPInfoHeader info_header = {
      .biSize = sizeof(BMPInfoHeader),
      .biWidth = img.width,
      .biHeight = img.height,
      .biPlanes = 1,
      .biBitCount = 24,
      .biCompression = 0,
      .biSizeImage = (uint32_t)image_size,
      .biXPelsPerMeter = 2835,
      .biYPelsPerMeter = 2835,
      .biClrUsed = 0,
      .biClrImportant = 0,
  };

  std::ofstream f(filename, std::ios::binary);

  if (!f.is_open())
    return;

  f.write(reinterpret_cast<char *>(&file_header), sizeof(BMPFileHeader));
  f.write(reinterpret_cast<char *>(&info_header), sizeof(BMPInfoHeader));

  std::vector<uint8_t> row_buffer(row_size, 0);
  for (int y = img.height - 1; y >= 0; y--) {
    for (int x = 0; x < img.width; x++) {
      int idx = y * img.width + x;
      unsigned char *r = &row_buffer.data()[x * 3];

      const RGB &pixel = img.pixels[idx];
      *(r + 0) = pixel.b;
      *(r + 1) = pixel.g;
      *(r + 2) = pixel.r;
    }

    f.write(reinterpret_cast<char *>(row_buffer.data()), row_size);
  }

  f.close();
}

std::optional<Image> read_bmp(const std::string &filename) {
  std::ifstream f(filename, std::ios::binary);

  if (!f.is_open())
    return std::nullopt;

  BMPFileHeader file_header;
  BMPInfoHeader info_header;

  f.read(reinterpret_cast<char *>(&file_header), sizeof(file_header));
  f.read(reinterpret_cast<char *>(&info_header), sizeof(info_header));

  if (!f || file_header.bfType != 0x4D42 || info_header.biBitCount != 24 ||
      info_header.biCompression != 0)
    return std::nullopt;

  Image img;
  img.width = info_header.biWidth;
  img.height = info_header.biHeight;
  img.pixels.resize(img.width * img.height);

  f.seekg(file_header.bfOffBits, std::ios::beg);
  if (!f)
    return std::nullopt;

  const int row_size = (img.width * 3 + 3) & ~3;
  std::vector<uint8_t> row_buffer(row_size);

  for (int y = 0; y < img.height; ++y) {
    f.read(reinterpret_cast<char *>(row_buffer.data()), row_size);
    if (!f)
      return std::nullopt;

    for (int x = 0; x < img.width; ++x) {
      const int idx = (img.height - 1 - y) * img.width + x;
      const uint8_t *src_pixel = row_buffer.data() + (x * 3);

      RGB &pixel = img.pixels[idx];
      pixel.r = src_pixel[2];
      pixel.g = src_pixel[1];
      pixel.b = src_pixel[0];
    }
  }

  return img;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Invalid arguments\n";
    return 1;
  }

  Image img = read_bmp(argv[1]).value();

  std::cout << img.width << "x" << img.height;
  grayscale(img);
  write_bmp(img, argv[2]);
}
