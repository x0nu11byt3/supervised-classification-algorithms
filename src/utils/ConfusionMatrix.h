#ifndef CONFUSION_MATRIX_H
#define CONFUSION_MATRIX_H

#include <vector>
#include <map>
#include <iostream>

namespace ml {

    class ConfusionMatrix {
        public:
            // Constructor: accepts unique tags (e.g. {0,1,2} or {1,2,3})
            ConfusionMatrix(const std::vector<int>& labels);

            // Adds a pair (actual, predicted)
            void add(int trueLabel, int predLabel);

            // Calculate the accuracy
            double accuracy() const;

            // Calculate the error (1 - accuracy)
            double error() const;

            // Prints the formatted array (similar to the images)
            void print() const;

            // Returns the raw array (for testing purposes)
            const std::vector<std::vector<int>>& getMatrix() const { return matrix; }

        private:
            std::vector<int> classLabels;           // sorted tags
            std::map<int, int> labelToIndex;        // tag mapping -> index
            std::vector<std::vector<int>> matrix;   // matrix n x n
            int total = 0;
    };

} // namespace ml

#endif
