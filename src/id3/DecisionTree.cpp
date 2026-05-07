#include "id3/DecisionTree.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <map>
#include <iostream>

namespace ml {

void DecisionTree::build(const std::vector<Instance>& data, int maxDepth, int minSamplesSplit) {
    root = buildTree(data, 0, maxDepth, minSamplesSplit);
}

int DecisionTree::predict(const std::vector<double>& features) const {
    Node* node = root.get();
    while (!node->isLeaf) {
        if (features[node->featureIndex] <= node->threshold)
            node = node->left.get();
        else
            node = node->right.get();
    }
    return node->predictedClass;
}

std::unique_ptr<Node> DecisionTree::buildTree(const std::vector<Instance>& data, int depth, int maxDepth, int minSamplesSplit) {
    auto node = std::make_unique<Node>();
    // If they are all the same -> sheet
    int firstLabel = data[0].label;
    bool allSame = true;
    for (const auto& inst : data) {
        if (inst.label != firstLabel) { allSame = false; break; }
    }
    if (allSame || data.size() < (size_t)minSamplesSplit || (maxDepth != -1 && depth >= maxDepth)) {
        node->isLeaf = true;
        node->predictedClass = majorityClass(data);
        return node;
    }

    double bestGain = -1.0;
    int bestFeature = -1;
    double bestThreshold = 0.0;
    int nFeatures = data[0].features.size();
    for (int f = 0; f < nFeatures; ++f) {
        double thr;
        double gain = informationGain(data, f, thr);
        if (gain > bestGain) {
            bestGain = gain;
            bestFeature = f;
            bestThreshold = thr;
        }
    }
    if (bestFeature == -1 || bestGain < 1e-9) {
        node->isLeaf = true;
        node->predictedClass = majorityClass(data);
        return node;
    }

    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    std::vector<Instance> leftData, rightData;

    splitData(data, bestFeature, bestThreshold, leftData, rightData);
    if (leftData.empty() || rightData.empty()) {
        node->isLeaf = true;
        node->predictedClass = majorityClass(data);
        return node;
    }
    node->left = buildTree(leftData, depth+1, maxDepth, minSamplesSplit);
    node->right = buildTree(rightData, depth+1, maxDepth, minSamplesSplit);
    return node;
}

double DecisionTree::entropy(const std::vector<Instance>& data) const {

    std::map<int, int> counts;
    for (const auto& inst : data) counts[inst.label]++;
    double ent = 0.0;
    double total = data.size();
    for (auto& p : counts) {
        double p_i = p.second / total;
        ent -= p_i * log2(p_i);
    }
    return ent;

}

double DecisionTree::informationGain(const std::vector<Instance>& data, int featureIdx, double& bestThreshold) const {

    double totalEntropy = entropy(data);
    // Extract values from the feature and sort them
    std::vector<std::pair<double, int>> sorted;
    for (const auto& inst : data) {
        sorted.emplace_back(inst.features[featureIdx], inst.label);
    }
    std::sort(sorted.begin(), sorted.end());
    double bestGain = -1.0;
    bestThreshold = 0.0;
    // Test thresholds between consecutive values
    for (size_t i = 0; i < sorted.size() - 1; ++i) {
        if (sorted[i].first == sorted[i+1].first) continue;
        double candidate = (sorted[i].first + sorted[i+1].first) / 2.0;
        std::vector<Instance> leftVec, rightVec;
        for (const auto& inst : data) {
            if (inst.features[featureIdx] <= candidate)
                leftVec.push_back(inst);
            else
                rightVec.push_back(inst);
        }
        double leftEntropy = entropy(leftVec);
        double rightEntropy = entropy(rightVec);
        double p_left = leftVec.size() / (double)data.size();
        double p_right = rightVec.size() / (double)data.size();
        double gain = totalEntropy - (p_left * leftEntropy + p_right * rightEntropy);
        if (gain > bestGain) {
            bestGain = gain;
            bestThreshold = candidate;
        }
    }

    return bestGain;

}

void DecisionTree::splitData(const std::vector<Instance>& data, int featureIdx, double threshold,

    std::vector<Instance>& left, std::vector<Instance>& right) const {
    for (const auto& inst : data) {
        if (inst.features[featureIdx] <= threshold)
            left.push_back(inst);
        else
            right.push_back(inst);
    }

}

int DecisionTree::majorityClass(const std::vector<Instance>& data) const {
    std::map<int, int> counts;
    for (const auto& inst : data) counts[inst.label]++;
    int bestClass = -1;
    int maxCount = -1;
    for (auto& p : counts) {
        if (p.second > maxCount) {
            maxCount = p.second;
            bestClass = p.first;
        }
    }
    return bestClass;
}

} // namespace ml
