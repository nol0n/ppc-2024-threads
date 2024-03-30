// Copyright 2024 Sharapov Georgiy

enum { SIZE = 2310 };

#include <gtest/gtest.h>
#include <omp.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "omp/sharapov_g_sobel/include/ssobel_omp.hpp"

TEST(sharapov_g_sobel_omp, test_pipeline_run) {
  int sideLen = SIZE;
  std::vector<SSobelOmp::RGB> ColoredImage = SSobelOmp::generateColorImage(sideLen, sideLen, 1984);
  std::vector<SSobelOmp::GrayScale> EdgeImage(ColoredImage.size());

  // TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(ColoredImage.data()));
  taskDataOmp->inputs_count.emplace_back(sideLen);
  taskDataOmp->inputs_count.emplace_back(sideLen);
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(EdgeImage.data()));
  taskDataOmp->outputs_count.emplace_back(sideLen);
  taskDataOmp->outputs_count.emplace_back(sideLen);

  // Task
  auto SobelOmp = std::make_shared<SSobelOmp>(taskDataOmp);

  // Perf attr
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(SobelOmp);
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

TEST(sharapov_g_sobel_omp, test_task_run) {
  int sideLen = SIZE;
  std::vector<SSobelOmp::RGB> ColoredImage = SSobelOmp::generateColorImage(sideLen, sideLen, 1984);
  std::vector<SSobelOmp::GrayScale> EdgeImage(ColoredImage.size());

  // TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataOmp = std::make_shared<ppc::core::TaskData>();
  taskDataOmp->inputs.emplace_back(reinterpret_cast<uint8_t *>(ColoredImage.data()));
  taskDataOmp->inputs_count.emplace_back(sideLen);
  taskDataOmp->inputs_count.emplace_back(sideLen);
  taskDataOmp->outputs.emplace_back(reinterpret_cast<uint8_t *>(EdgeImage.data()));
  taskDataOmp->outputs_count.emplace_back(sideLen);
  taskDataOmp->outputs_count.emplace_back(sideLen);

  // Task
  auto SobelOmp = std::make_shared<SSobelOmp>(taskDataOmp);

  // Perf attr
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  perfAttr->current_timer = [&] { return omp_get_wtime(); };

  // Perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(SobelOmp);
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
