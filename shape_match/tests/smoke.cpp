#include "openshape/openshape.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>

int main() {
  try {
    cv::Mat templ = cv::Mat::zeros(80, 80, CV_8UC1);
    cv::rectangle(templ, cv::Rect(20, 20, 40, 30), cv::Scalar(255), 2);
    openshape::ShapeModelParams mp;
    mp.min_gradient_magnitude = 1;
    mp.min_model_points = 4;
    mp.max_model_points = 200;
    mp.min_point_distance = 2;
    const auto model = openshape::ShapeModelBuilder::create(openshape::ImageView(templ), mp);
    if (model.empty() || model.levels().empty()) return 1;
    cv::Mat scene = cv::Mat::zeros(160, 180, CV_8UC1);
    cv::rectangle(scene, cv::Rect(70, 60, 40, 30), cv::Scalar(255), 2);
    openshape::SearchParams sp;
    sp.min_score = 0.5;
    sp.num_matches = 1;
    sp.max_overlap = 0.5;
    const auto results = openshape::find_shape_models(openshape::ImageView(scene), model, sp);
    if (results.empty() || results.front().score < sp.min_score) return 2;
    if (std::abs(results.front().column - 90.0) > 4 || std::abs(results.front().row - 80.0) > 4) return 3;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 4;
  }
}
