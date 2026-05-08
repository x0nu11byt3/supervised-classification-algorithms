#include "dataset/Dataset.h"
#include "id3/ID3.h"
#include "naive_bayes/GaussianNB.h"
#include "utils/ConfusionMatrix.h"
#include <iostream>
#include <memory>

using namespace ml;

int main(int argc, char* argv[]) {

    // ruta por defecto

    std::string filename = "../data/iris.csv";
    if (argc > 1) filename = argv[1];

    Dataset data;

    try {

        data.loadFromCSV(filename, false); // sin cabecera

    } catch (const std::exception& e) {

        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        return 1;

    }

    Dataset train, test;
    data.split(0.7, train, test);
    std::cout << "Training: " << train.getInstances().size()
              << " instances, Test: " << test.getInstances().size() << std::endl;

    // --- Naive Bayes ---
    GaussianNB nb;
    nb.train(train);
    double acc_nb = nb.accuracy(test);
    std::cout << "Accuracy Naive Bayes (directa): " << acc_nb << std::endl;


    // --- Árbol de Decisión (ID3) ---
    ID3 tree;
    tree.setMaxDepth(5);
    tree.setMinSamplesSplit(2);
    tree.train(train);
    std::cout << "\nÁrbol de Decisión (ID3):" << std::endl;
    tree.print();                      // imprime la estructura del árbol
    double acc_tree = tree.accuracy(test);
    std::cout << "Accuracy ID3 (directa): " << acc_tree << std::endl;


    // --- Matrices de Confusión para ambos clasificadores ---
    // Obtener etiquetas únicas del conjunto de entrenamiento (para tener el orden de clases)
    std::vector<int> uniqueLabels = train.getUniqueLabels();
    ConfusionMatrix cm_nb(uniqueLabels);
    ConfusionMatrix cm_tree(uniqueLabels);

    // Evaluar cada instancia de prueba
    for (const auto& inst : test.getInstances()) {
        int pred_nb = nb.predict(inst.features);
        int pred_tree = tree.predict(inst.features);
        cm_nb.add(inst.label, pred_nb);
        cm_tree.add(inst.label, pred_tree);
    }

    // Mostrar resultados detallados
    std::cout << "\n=== Resultados con Matriz de Confusión ===\n";
    std::cout << "\n--- Naive Bayes ---\n";
    cm_nb.print();
    std::cout << "\n--- Árbol de Decisión (ID3) ---\n";
    cm_tree.print();

    // Ejemplo de predicción sobre la primera instancia de prueba
    if (!test.getInstances().empty()) {
        auto& first = test.getInstances()[0];
        int pred_nb = nb.predict(first.features);
        int pred_tree = tree.predict(first.features);
        std::cout << "\nPrimera instancia de prueba:" << std::endl;
        std::cout << "  Real: " << first.label << " | NB: " << pred_nb << " | ID3: " << pred_tree << std::endl;
    }

    return 0;
}
