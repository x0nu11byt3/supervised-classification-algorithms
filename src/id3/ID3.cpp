#include "id3/ID3.h"

namespace ml {

void ID3::train(const Dataset& data) {
    tree.build(data.getInstances(), maxDepth, minSamplesSplit);
}

int ID3::predict(const std::vector<double>& features) const {
    return tree.predict(features);
}

} // namespace ml
