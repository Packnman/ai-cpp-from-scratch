foreach(mode 0 1)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env "AI_CPP_CUDA_MEMORY_POOL=${mode}"
        "${PROGRAM}" "${CMAKE_CURRENT_BINARY_DIR}/cuda-memory-${mode}.txt"
        RESULT_VARIABLE result ERROR_VARIABLE diagnostics)
    message(STATUS "${diagnostics}")
    if(diagnostics MATCHES "CUDA memory shutdown:|CUDA memory release:|refusing to destroy|CUDA memory:.*:")
        message(FATAL_ERROR "Allocator cleanup failed: ${diagnostics}")
    endif()
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Numeric snapshot failed for allocator ${mode}")
    endif()
endforeach()
execute_process(COMMAND "${PROGRAM}" --compare
    "${CMAKE_CURRENT_BINARY_DIR}/cuda-memory-0.txt"
    "${CMAKE_CURRENT_BINARY_DIR}/cuda-memory-1.txt" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Allocator numerical comparison failed")
endif()
