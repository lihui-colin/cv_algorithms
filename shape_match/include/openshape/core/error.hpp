#pragma once
#include <stdexcept>
#include <string>

namespace openshape {
class OpenShapeError : public std::runtime_error {
public:
  explicit OpenShapeError(const std::string& message) : std::runtime_error(message) {}
};
class InvalidArgument : public OpenShapeError { using OpenShapeError::OpenShapeError; };
class EmptyImage : public OpenShapeError { using OpenShapeError::OpenShapeError; };
class InvalidModel : public OpenShapeError { using OpenShapeError::OpenShapeError; };
}
