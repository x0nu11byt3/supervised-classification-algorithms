#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include "../dataset/Dataset.h"
#include <vector>
#include <memory>

namespace ml {

    struct Node {
        bool isLeaf = false;
        int predictedClass = -1;
        int featureIndex = -1;
        double threshold = 0.0;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    class DecisionTree {

        public:
            void build(const std::vector<Instance>& data, int maxDepth, int minSamplesSplit);
            int predict(const std::vector<double>& features) const;

        private:
            std::unique_ptr<Node> root;
            std::unique_ptr<Node> buildTree(const std::vector<Instance>& data, int depth, int maxDepth, int minSamplesSplit);
            double entropy(const std::vector<Instance>& data) const;
            double informationGain(const std::vector<Instance>& data, int featureIdx, double& bestThreshold) const;
            void splitData(const std::vector<Instance>& data, int featureIdx, double threshold,
                        std::vector<Instance>& left, std::vector<Instance>& right) const;
            int majorityClass(const std::vector<Instance>& data) const;
        };

    } // namespace ml

#endif
