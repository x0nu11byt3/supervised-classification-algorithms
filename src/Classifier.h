#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "dataset/Dataset.h"
#include <vector>

namespace ml {

class Classifier {
public:
    virtual ~Classifier() = default;
    virtual void train(const Dataset& data) = 0;
    virtual int predict(const std::vector<double>& features) const = 0;
    virtual double accuracy(const Dataset& test) const;
};

inline double Classifier::accuracy(const Dataset& test) const {
    if (test.getInstances().empty()) return 0.0;
    int correct = 0;
    for (const auto& inst : test.getInstances()) {
        int pred = predict(inst.features);
        if (pred == inst.label) correct++;
    }
    return static_cast<double>(correct) / test.getInstances().size();
}

} // namespace ml

#endif
