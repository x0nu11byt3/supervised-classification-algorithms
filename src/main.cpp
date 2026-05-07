#include "dataset/Dataset.h"
#include "id3/ID3.h"
#include "naive_bayes/GaussianNB.h"
#include <iostream>
#include <memory>

using namespace ml;

int main(int argc, char* argv[]) {

    std::string filename = "../data/iris.csv";  // adjust route

    if (argc > 1) filename = argv[1];

    Dataset data;

    try {

        data.loadFromCSV(filename, false); // without head

    } catch (const std::exception& e) {

        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        return 1;

    }

    Dataset train, test;
    data.split(0.7, train, test);
    std::cout << "Training: " << train.getInstances().size()
              << "Instances, Test: " << test.getInstances().size() << std::endl;

    // Naive Bayes
    GaussianNB nb;
    nb.train(train);
    double acc_nb = nb.accuracy(test);
    std::cout << "Accuracy Naive Bayes: " << acc_nb << std::endl;

    // ID3 decision tree (with maxDepth=5 to prevent overfitting)
    ID3 tree;
    tree.setMaxDepth(5);
    tree.setMinSamplesSplit(2);
    tree.train(train);
    double acc_tree = tree.accuracy(test);
    std::cout << "Accuracy ID3: " << acc_tree << std::endl;

    // Example of an individual prediction
    if (!test.getInstances().empty()) {

        auto& first = test.getInstances()[0];
        int pred_nb = nb.predict(first.features);
        int pred_tree = tree.predict(first.features);
        std::cout << "\nFirst-instance ruling test:" << std::endl;
        std::cout << "  Real: " << first.label << " | NB: " << pred_nb << " | ID3: " << pred_tree << std::endl;

    }

    return 0;
}
