#include "openshape/openshape.hpp"
#include <iostream>

int main() {
  openshape::SearchParams params;
  params.validate();
  std::cout << openshape::version_string << '\n';
  return openshape::version_major == 0 && openshape::version_minor == 3 ? 0 : 1;
}
