set(LIB_SOURCES
    dataset/Dataset.cpp
    dataset/ArffLoader.cpp
    id3/ID3.cpp
    id3/DecisionTree.cpp
    naive_bayes/GaussianNB.cpp
    utils/ConfusionMatrix.cpp
)

add_library(ml_lib STATIC ${LIB_SOURCES})
target_include_directories(ml_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(run_id3_naive main.cpp)
target_link_libraries(run_id3_naive ml_lib)
