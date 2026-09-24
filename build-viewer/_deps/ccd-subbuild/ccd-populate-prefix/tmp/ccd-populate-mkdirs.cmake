# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/workspaces/ai_cpp/build-viewer/_deps/ccd-src"
  "/workspaces/ai_cpp/build-viewer/_deps/ccd-build"
  "/workspaces/ai_cpp/build-viewer/_deps/ccd-subbuild/ccd-populate-prefix"
  "/workspaces/ai_cpp/build-viewer/_deps/ccd-subbuild/ccd-populate-prefix/tmp"
  "/workspaces/ai_cpp/build-viewer/_deps/ccd-subbuild/ccd-populate-prefix/src/ccd-populate-stamp"
  "/workspaces/ai_cpp/build-viewer/_deps/ccd-subbuild/ccd-populate-prefix/src"
  "/workspaces/ai_cpp/build-viewer/_deps/ccd-subbuild/ccd-populate-prefix/src/ccd-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspaces/ai_cpp/build-viewer/_deps/ccd-subbuild/ccd-populate-prefix/src/ccd-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspaces/ai_cpp/build-viewer/_deps/ccd-subbuild/ccd-populate-prefix/src/ccd-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
