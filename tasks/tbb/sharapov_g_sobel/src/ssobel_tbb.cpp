// Copyright 2024 Sharapov Georgiy

#include "tbb/sharapov_g_sobel/include/ssobel_tbb.hpp"

#include <oneapi/tbb.h>

#include <cmath>
#include <random>
#include <thread>

SSobelTbb::GrayScale SSobelTbb::getPixel(const std::vector<SSobelTbb::GrayScale>& image, size_t x, size_t y,
                                         size_t width, size_t height) {
  if (x > width - 1) x = width - 1;
  if (x < 0) x = 0;
  if (y > height - 1) y = height - 1;
  if (y < 0) y = 0;

  return image[y * width + x];
}

std::vector<SSobelTbb::RGB> SSobelTbb::generateColorImage(size_t width, size_t height, size_t seed) {
  std::vector<SSobelTbb::RGB> image;
  image.reserve(width * height);

  std::mt19937 gen(static_cast<unsigned int>(seed));
  std::uniform_int_distribution<> rgb(0, 255);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      SSobelTbb::RGB pixel;
      pixel.r = static_cast<uint8_t>(rgb(gen));
      pixel.g = static_cast<uint8_t>(rgb(gen));
      pixel.b = static_cast<uint8_t>(rgb(gen));
      image.push_back(pixel);
    }
  }

  return image;
}

std::vector<SSobelTbb::GrayScale> SSobelTbb::convertToGrayScale(const std::vector<SSobelTbb::RGB>& colorImage,
                                                                size_t width, size_t height) {
  std::vector<SSobelTbb::GrayScale> grayImage(width * height);

  int imgSize = height * width;

  tbb::parallel_for(tbb::blocked_range<int>(0, imgSize), [&](const tbb::blocked_range<int>& r) {
    for (int index = r.begin(); index != r.end(); ++index) {
      const auto& pixel = colorImage[index];
      grayImage[index].value = static_cast<uint8_t>(0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b);
    }
  });

  return grayImage;
}

std::vector<SSobelTbb::GrayScale> SSobelTbb::SobelOperatorTbb(const std::vector<SSobelTbb::GrayScale>& grayImage,
                                                              size_t width, size_t height) {
  const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  int imgSize = height * width;

  std::vector<GrayScale> resultImg(width * height);
  tbb::parallel_for(tbb::blocked_range<int>(0, imgSize), [&](const tbb::blocked_range<int>& r) {
    for (int index = r.begin(); index != r.end(); ++index) {
      int i = index / width;
      int j = index % width;
      int sumX = 0;
      int sumY = 0;

      for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
          int posX = static_cast<int>(j) + y;
          int posY = static_cast<int>(i) + x;
          auto pixel = getPixel(grayImage, posX, posY, width, height);

          sumX += Gx[x + 1][y + 1] * pixel.value;
          sumY += Gy[x + 1][y + 1] * pixel.value;
        }
      }

      int sum = std::sqrt(sumX * sumX + sumY * sumY);

      sum = sum >= 200 ? 255 : 0;

      resultImg[index] = GrayScale{static_cast<uint8_t>(sum)};
    }
  });

  return resultImg;
}

void SSobelTbb::printPixel(const std::vector<GrayScale>& image, int width, int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      GrayScale pixel = getPixel(image, x, y, width, height);
      std::cout << static_cast<int>(pixel.value) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

bool SSobelTbb::validation() {
  try {
    internal_order_test();
  } catch (...) {
    return false;
  }

  return (taskData->inputs_count[0] == taskData->outputs_count[0] &&
          taskData->inputs_count[1] == taskData->outputs_count[1]);
}

bool SSobelTbb::pre_processing() {
  try {
    internal_order_test();

    imgWidth = taskData->inputs_count[0];
    imgHeight = taskData->inputs_count[1];
    imgSize = imgWidth * imgHeight;

    colored_img.reserve(imgSize);
    uint8_t* rawData = taskData->inputs[0];

    tbb::parallel_for(tbb::blocked_range<int>(0, imgSize), [&](const tbb::blocked_range<int>& r) {
      for (int index = r.begin(); index != r.end(); ++index) {
        colored_img[index] = RGB{rawData[index * 3], rawData[index * 3 + 1], rawData[index * 3 + 2]};
      }
    });

    grayscale_img = SSobelTbb::convertToGrayScale(colored_img, imgWidth, imgHeight);
  } catch (...) {
    return false;
  }
  return true;
}

bool SSobelTbb::run() {
  try {
    internal_order_test();

    result = SSobelTbb::SobelOperatorTbb(grayscale_img, imgWidth, imgHeight);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  } catch (...) {
    return false;
  }
  return true;
}

bool SSobelTbb::post_processing() {
  try {
    internal_order_test();

    int imgSize = grayscale_img.size();

    for (int i = 0; i < imgSize; ++i) {
      auto* pixel = reinterpret_cast<SSobelTbb::GrayScale*>(taskData->outputs[0] + i);
      *pixel = result[i];
    }
  } catch (...) {
    return false;
  }
  return true;
}
