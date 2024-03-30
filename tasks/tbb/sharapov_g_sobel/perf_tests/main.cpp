// Copyright 2024 Sharapov Georgiy

enum { IMG_SIZE = 2310 };

#include <gtest/gtest.h>
#include <oneapi/tbb.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "tbb/sharapov_g_sobel/include/ssobel_tbb.hpp"

TEST(sharapov_g_sobel_tbb, test_pipeline_run) {
  int sideLen = IMG_SIZE;
  std::vector<SSobelTbb::RGB> ColoredImage = SSobelTbb::generateColorImage(sideLen, sideLen, 1984);
  std::vector<SSobelTbb::GrayScale> EdgeImage(ColoredImage.size());

  // TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(ColoredImage.data()));
  taskDataTbb->inputs_count.emplace_back(sideLen);
  taskDataTbb->inputs_count.emplace_back(sideLen);
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(EdgeImage.data()));
  taskDataTbb->outputs_count.emplace_back(sideLen);
  taskDataTbb->outputs_count.emplace_back(sideLen);

  // Task
  auto SobelTbb = std::make_shared<SSobelTbb>(taskDataTbb);

  // Perf attr
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(SobelTbb);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Yes, this is not a very meaningful test, but in this case,
  // it's difficult to perform a check. There are two options:
  // use some reference library, like OpenCV, or check the result
  // visually (the latter is not an option). Suggestions are welcome ;)
  for (size_t i = 0; i < ColoredImage.size(); ++i) {
    EXPECT_TRUE(EdgeImage[i].value == 0 || 255);
  }
}

TEST(sharapov_g_sobel_tbb, test_task_run) {
  int sideLen = IMG_SIZE;
  std::vector<SSobelTbb::RGB> ColoredImage = SSobelTbb::generateColorImage(sideLen, sideLen, 1984);
  std::vector<SSobelTbb::GrayScale> EdgeImage(ColoredImage.size());

  // TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataTbb = std::make_shared<ppc::core::TaskData>();
  taskDataTbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(ColoredImage.data()));
  taskDataTbb->inputs_count.emplace_back(sideLen);
  taskDataTbb->inputs_count.emplace_back(sideLen);
  taskDataTbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(EdgeImage.data()));
  taskDataTbb->outputs_count.emplace_back(sideLen);
  taskDataTbb->outputs_count.emplace_back(sideLen);

  // Task
  auto SobelTbb = std::make_shared<SSobelTbb>(taskDataTbb);

  // Perf attr
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = oneapi::tbb::tick_count::now();
  perfAttr->current_timer = [&] { return (oneapi::tbb::tick_count::now() - t0).seconds(); };

  // Perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(SobelTbb);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  // Yes, this is not a very meaningful test, but in this case,
  // it's difficult to perform a check. There are two options:
  // use some reference library, like OpenCV, or check the result
  // visually (the latter is not an option). Suggestions are welcome ;)
  for (size_t i = 0; i < ColoredImage.size(); ++i) {
    EXPECT_TRUE(static_cast<int>(EdgeImage[i].value) == 0 || 255);
  }
}
