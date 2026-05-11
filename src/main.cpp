#include "dataset/Dataset.h"
#include "id3/ID3.h"
#include "naive_bayes/GaussianNB.h"
#include "utils/ConfusionMatrix.h"
#include "dataset/ArffLoader.h"
#include <iostream>
#include <memory>

using namespace ml;

int main(int argc, char* argv[]) {
    std::string filename = "../data/iris.csv";
    if (argc > 1) filename = argv[1];

    Dataset data;

    try {
        // Detect file extension
        std::string ext = filename.substr(filename.find_last_of(".") + 1);
        if (ext == "arff") {
            // class = last attribute
            ArffLoader::load(filename, data, -1);
        } else {
            // For CSV: we assume that the class is in the LAST column and that there is no header
            data.loadFromCSV(filename, false);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        return 1;
    }

    Dataset train, test;
    data.split(0.8, train, test);
    std::cout << "Training: " << train.getInstances().size()
              << " instances, Test: " << test.getInstances().size() << std::endl;

    // Naive Bayes
    GaussianNB nb;
    nb.train(train);
    double acc_nb = nb.accuracy(test);
    std::cout << "Accuracy Naive Bayes (directa): " << acc_nb << std::endl;

    // Decision Tree (ID3)
    ID3 tree;
    tree.setMaxDepth(5);
    tree.setMinSamplesSplit(2);
    tree.train(train);
    std::cout << "\nDecision Tree (ID3):" << std::endl;
    tree.print();
    double acc_tree = tree.accuracy(test);
    std::cout << "Accuracy ID3 (directa): " << acc_tree << std::endl;

    // Confusion matrices
    std::vector<int> uniqueLabels = train.getUniqueLabels();
    ConfusionMatrix cm_nb(uniqueLabels);
    ConfusionMatrix cm_tree(uniqueLabels);

    for (const auto& inst : test.getInstances()) {
        int pred_nb = nb.predict(inst.features);
        int pred_tree = tree.predict(inst.features);
        cm_nb.add(inst.label, pred_nb);
        cm_tree.add(inst.label, pred_tree);
    }

    std::cout << "\n Results with a Confusion Matrix \n";
    std::cout << "\n Naive Bayes \n";
    cm_nb.print();
    std::cout << "\n Decision Tree (ID3) \n";
    cm_tree.print();

    if (!test.getInstances().empty()) {
        auto& first = test.getInstances()[0];
        int pred_nb = nb.predict(first.features);
        int pred_tree = tree.predict(first.features);
        std::cout << "\nFirst instance trial:" << std::endl;
        std::cout << "  Real: " << first.label << " | NB: " << pred_nb << " | ID3: " << pred_tree << std::endl;
    }

    return 0;
}
