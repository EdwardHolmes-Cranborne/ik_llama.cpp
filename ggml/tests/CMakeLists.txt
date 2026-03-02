# ANE test targets (only built when GGML_ANE is enabled)
if (GGML_ANE AND APPLE)
    # Phase 0: Hardware characterization tests
    add_executable(test-ane-hardware test-ane-hardware.mm)
    target_link_libraries(test-ane-hardware PRIVATE ggml)
    target_include_directories(test-ane-hardware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_include_directories(test-ane-hardware PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    set_target_properties(test-ane-hardware PROPERTIES
        COMPILE_FLAGS "-fno-objc-arc -std=c++17"
    )
    target_link_libraries(test-ane-hardware PRIVATE
        "-framework Foundation"
        "-framework IOSurface"
        "-framework Accelerate"
        "-framework Metal"
        "-ldl"
    )
    install(TARGETS test-ane-hardware RUNTIME)
    add_test(NAME test-ane-hardware COMMAND test-ane-hardware --quick)
    set_property(TEST test-ane-hardware PROPERTY LABELS "ane")

    # Phase 1: Backend unit tests
    add_executable(test-ane-backend test-ane-backend.mm)
    target_link_libraries(test-ane-backend PRIVATE ggml)
    target_include_directories(test-ane-backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
    target_include_directories(test-ane-backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    set_target_properties(test-ane-backend PROPERTIES
        COMPILE_FLAGS "-fno-objc-arc -std=c++17"
    )
    target_link_libraries(test-ane-backend PRIVATE
        "-framework Foundation"
        "-framework IOSurface"
        "-framework Accelerate"
        "-framework Metal"
        "-ldl"
    )
    install(TARGETS test-ane-backend RUNTIME)
    add_test(NAME test-ane-backend COMMAND test-ane-backend)
    set_property(TEST test-ane-backend PROPERTY LABELS "ane")
endif()
