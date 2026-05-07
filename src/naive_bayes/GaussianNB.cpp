#include "naive_bayes/GaussianNB.h"
#include <cmath>
#include <limits>
#include <iostream>

namespace ml {

void GaussianNB::train(const Dataset& data) {
    auto instances = data.getInstances();
    std::map<int, std::vector<Instance>> grouped;
    for (const auto& inst : instances) grouped[inst.label].push_back(inst);
    int total = instances.size();
    int nFeatures = data.getNumFeatures();
    for (auto& pair : grouped) {
        int label = pair.first;
        auto& group = pair.second;
        double prior = group.size() / (double)total;
        std::vector<double> mean(nFeatures, 0.0);
        for (const auto& inst : group) {
            for (int f = 0; f < nFeatures; ++f)
                mean[f] += inst.features[f];
        }
        for (int f = 0; f < nFeatures; ++f) mean[f] /= group.size();
        std::vector<double> var(nFeatures, 0.0);
        for (const auto& inst : group) {
            for (int f = 0; f < nFeatures; ++f)
                var[f] += (inst.features[f] - mean[f]) * (inst.features[f] - mean[f]);
        }
        for (int f = 0; f < nFeatures; ++f) var[f] = var[f] / group.size() + 1e-9; // suavizado
        model[label] = {prior, mean, var};
    }
}

double GaussianNB::logPDF(double x, double mean, double var) const {
    double frac = (x - mean) / sqrt(var);
    return -0.5 * log(2 * M_PI * var) - 0.5 * frac * frac;
}

int GaussianNB::predict(const std::vector<double>& features) const {
    double bestLogProb = -std::numeric_limits<double>::infinity();
    int bestClass = -1;
    for (auto& pair : model) {
        int label = pair.first;
        const auto& stats = pair.second;
        double logProb = log(stats.prior);
        for (size_t f = 0; f < features.size(); ++f) {
            logProb += logPDF(features[f], stats.mean[f], stats.var[f]);
        }
        if (logProb > bestLogProb) {
            bestLogProb = logProb;
            bestClass = label;
        }
    }
    return bestClass;
}

} // namespace ml
