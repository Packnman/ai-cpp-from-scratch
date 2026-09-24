# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/workspaces/ai_cpp/build-viewer/_deps/marchingcubecpp-src"
  "/workspaces/ai_cpp/build-viewer/_deps/marchingcubecpp-build"
  "/workspaces/ai_cpp/build-viewer/_deps/marchingcubecpp-subbuild/marchingcubecpp-populate-prefix"
  "/workspaces/ai_cpp/build-viewer/_deps/marchingcubecpp-subbuild/marchingcubecpp-populate-prefix/tmp"
  "/workspaces/ai_cpp/build-viewer/_deps/marchingcubecpp-subbuild/marchingcubecpp-populate-prefix/src/marchingcubecpp-populate-stamp"
  "/workspaces/ai_cpp/build-viewer/_deps/marchingcubecpp-subbuild/marchingcubecpp-populate-prefix/src"
  "/workspaces/ai_cpp/build-viewer/_deps/marchingcubecpp-subbuild/marchingcubecpp-populate-prefix/src/marchingcubecpp-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspaces/ai_cpp/build-viewer/_deps/marchingcubecpp-subbuild/marchingcubecpp-populate-prefix/src/marchingcubecpp-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspaces/ai_cpp/build-viewer/_deps/marchingcubecpp-subbuild/marchingcubecpp-populate-prefix/src/marchingcubecpp-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
