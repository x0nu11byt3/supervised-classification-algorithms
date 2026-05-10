#include "dataset/Dataset.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iostream>
#include <map>
#include <string>
#include <stdexcept> // std::runtime_error

namespace ml {

    void Dataset::loadFromCSV(const std::string& filename, bool hasHeader) {

        instances.clear();
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("The file could not be opened: " + filename);
        }
        std::string line;
        if (hasHeader) std::getline(file, line); // jump head

        std::map<std::string, int> labelMap;
        int nextLabel = 0;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> features;
            std::string lastCell;
            int col = 0;

            while (std::getline(ss, cell, ',')) {
                if (ss.eof()) {
                    lastCell = cell;  // last column
                } else {
                    features.push_back(std::stod(cell));
                }
                col++;
            }

            // Convert the last column to an integer
            int label;
            try {
                label = std::stoi(lastCell);
            } catch (...) {
                // It's text; use the map
                if (labelMap.find(lastCell) == labelMap.end()) {
                    labelMap[lastCell] = nextLabel++;
                }
                label = labelMap[lastCell];
            }

            if (numFeatures == 0) numFeatures = features.size();
            instances.push_back({features, label});
        }

        std::cout << "instances loaded" << instances.size() << "  with " << numFeatures << " features.\n";
    }

    void Dataset::split(double trainRatio, Dataset& train, Dataset& test) const {

        std::vector<Instance> shuffled = instances;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(shuffled.begin(), shuffled.end(), g);

        size_t trainSize = static_cast<size_t>(trainRatio * shuffled.size());

        for (size_t i = 0; i < trainSize; ++i)
            train.instances.push_back(shuffled[i]);

        for (size_t i = trainSize; i < shuffled.size(); ++i)
            test.instances.push_back(shuffled[i]);

        train.numFeatures = numFeatures;
        test.numFeatures = numFeatures;

    }

    void ml::Dataset::setInstances(const std::vector<Instance>& newInstances) {
        instances = newInstances;
        numFeatures = newInstances.empty() ? 0 : newInstances[0].features.size();
    }

    const std::vector<Instance>& Dataset::getInstances() const {
        return instances;

    }

    int Dataset::getNumFeatures() const {
        return numFeatures;

    }

    std::vector<int> Dataset::getUniqueLabels() const {

        std::vector<int> labels;

        for (const auto& inst : instances) {
            if (std::find(labels.begin(), labels.end(), inst.label) == labels.end())
                labels.push_back(inst.label);
        }
        return labels;
    }

} // namespace ml
