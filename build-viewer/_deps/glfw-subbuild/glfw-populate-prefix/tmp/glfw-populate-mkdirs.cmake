# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/workspaces/ai_cpp/build-viewer/_deps/glfw-src"
  "/workspaces/ai_cpp/build-viewer/_deps/glfw-build"
  "/workspaces/ai_cpp/build-viewer/_deps/glfw-subbuild/glfw-populate-prefix"
  "/workspaces/ai_cpp/build-viewer/_deps/glfw-subbuild/glfw-populate-prefix/tmp"
  "/workspaces/ai_cpp/build-viewer/_deps/glfw-subbuild/glfw-populate-prefix/src/glfw-populate-stamp"
  "/workspaces/ai_cpp/build-viewer/_deps/glfw-subbuild/glfw-populate-prefix/src"
  "/workspaces/ai_cpp/build-viewer/_deps/glfw-subbuild/glfw-populate-prefix/src/glfw-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspaces/ai_cpp/build-viewer/_deps/glfw-subbuild/glfw-populate-prefix/src/glfw-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspaces/ai_cpp/build-viewer/_deps/glfw-subbuild/glfw-populate-prefix/src/glfw-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
