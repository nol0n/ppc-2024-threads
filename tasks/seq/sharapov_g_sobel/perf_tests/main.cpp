// Copyright 2024 Sharapov Georgiy

enum { SIZE = 2310 };

#include <gtest/gtest.h>
#include <omp.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/sharapov_g_sobel/include/ssobel_seq.hpp"

TEST(sharapov_g_sobel_seq, test_pipeline_run) {
  int sideLen = SIZE;
  std::vector<SSobelSeq::RGB> ColoredImage = SSobelSeq::generateColorImage(sideLen, sideLen, 1984);
  std::vector<SSobelSeq::GrayScale> EdgeImage(ColoredImage.size());

  // TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(ColoredImage.data()));
  taskDataSeq->inputs_count.emplace_back(sideLen);
  taskDataSeq->inputs_count.emplace_back(sideLen);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(EdgeImage.data()));
  taskDataSeq->outputs_count.emplace_back(sideLen);
  taskDataSeq->outputs_count.emplace_back(sideLen);

  // Task
  auto SobelSeq = std::make_shared<SSobelSeq>(taskDataSeq);

  // Perf attr
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(SobelSeq);
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

TEST(sharapov_g_sobel_seq, test_task_run) {
  int sideLen = SIZE;
  std::vector<SSobelSeq::RGB> ColoredImage = SSobelSeq::generateColorImage(sideLen, sideLen, 1984);
  std::vector<SSobelSeq::GrayScale> EdgeImage(ColoredImage.size());

  // TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(ColoredImage.data()));
  taskDataSeq->inputs_count.emplace_back(sideLen);
  taskDataSeq->inputs_count.emplace_back(sideLen);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(EdgeImage.data()));
  taskDataSeq->outputs_count.emplace_back(sideLen);
  taskDataSeq->outputs_count.emplace_back(sideLen);

  // Task
  auto SobelSeq = std::make_shared<SSobelSeq>(taskDataSeq);

  // Perf attr
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(SobelSeq);
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
