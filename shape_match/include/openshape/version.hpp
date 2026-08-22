#pragma once

#define OPENSHAPE_VERSION_MAJOR 0
#define OPENSHAPE_VERSION_MINOR 3
#define OPENSHAPE_VERSION_PATCH 0
#define OPENSHAPE_VERSION_STRING "0.3.0"

namespace openshape {
inline constexpr int version_major = OPENSHAPE_VERSION_MAJOR;
inline constexpr int version_minor = OPENSHAPE_VERSION_MINOR;
inline constexpr int version_patch = OPENSHAPE_VERSION_PATCH;
inline constexpr const char* version_string = OPENSHAPE_VERSION_STRING;
}
