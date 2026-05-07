#ifndef ID3_H
#define ID3_H

#include "../Classifier.h"
#include "DecisionTree.h"

namespace ml {

class ID3 : public Classifier {
public:
    void train(const Dataset& data) override;
    int predict(const std::vector<double>& features) const override;
    void setMaxDepth(int depth) { maxDepth = depth; }
    void setMinSamplesSplit(int minSamples) { minSamplesSplit = minSamples; }

private:
    DecisionTree tree;
    int maxDepth = -1;
    int minSamplesSplit = 2;
};

} // namespace ml

#endif
