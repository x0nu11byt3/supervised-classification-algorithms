#include "UCILoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>

namespace ml {

    void UCILoader::loadData(const std::string& dataFilePath, Dataset& dataset) {

        std::ifstream file(dataFilePath);
        if (!file.is_open()) {
            throw std::runtime_error("It couldn't be opened: " + dataFilePath);
        }

        std::vector<Instance> instances;
        std::string line;
        int numFeatures = -1;
        // to convert string values to integers
        std::map<std::string, int> labelMap;
        int nextLabelId = 0;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> features;
            std::string classStr;
            int col = 0;

            while (std::getline(ss, cell, ',')) {
                // If it is the last column, it is the class (it can be a string or a number)
                if (ss.eof()) {
                    classStr = cell;
                } else {
                    features.push_back(std::stod(cell));
                }
                col++;
            }

            // Convert class to integer
            int label;
            if (labelMap.find(classStr) == labelMap.end()) {
                labelMap[classStr] = nextLabelId++;
            }
            label = labelMap[classStr];

            if (numFeatures == -1) numFeatures = features.size();
            instances.push_back({features, label});
        }

        // Assign to the dataset
        dataset = Dataset(); // clean
        for (const auto& inst : instances) {
            dataset.getInstances().push_back(inst); // implement method add or class friend
        }
    }

} // namespace ml
