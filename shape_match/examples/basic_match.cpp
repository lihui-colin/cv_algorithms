#include "openshape/openshape.hpp"
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: shape_match_example <scene> <template>\n";
    return 2;
  }
  try {
    const openshape::Image scene = openshape::Image::read(argv[1]);
    const openshape::Image templ = openshape::Image::read(argv[2]);
    openshape::ShapeModelParams mp;
    mp.roi = cv::Rect(0, 0, templ.mat().cols, templ.mat().rows);
    const auto model = openshape::ShapeModelBuilder::create(templ.view(), mp);
    openshape::SearchParams sp;
    sp.angle_start = -30; sp.angle_extent = 60; sp.angle_step = 2;
    sp.num_matches = 5; sp.min_score = 0.55;
    const auto results = openshape::find_shape_models(scene.view(), model, sp);
    for (const auto& r : results)
      std::cout << "row=" << r.row << " column=" << r.column << " angle=" << r.angle
                << " scale=" << r.scale << " score=" << r.score
                << " valid=" << r.valid_point_count << '\n';
  } catch (const std::exception& e) {
    std::cerr << "shape matching failed: " << e.what() << '\n';
    return 1;
  }
}
