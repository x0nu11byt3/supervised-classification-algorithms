#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>

namespace ml {

    struct Instance {
        std::vector<double> features;
        int label;
    };

    class Dataset {
        public:
            void loadFromCSV(const std::string& filename, bool hasHeader = true);
            void split(double trainRatio, Dataset& train, Dataset& test) const;

            void setInstances(const std::vector<Instance>& newInstances);
            const std::vector<Instance>& getInstances() const;
            int getNumFeatures() const;
            std::vector<int> getUniqueLabels() const;

        private:
            std::vector<Instance> instances;
            int numFeatures = 0;
    };

} // namespace ml

#endif
