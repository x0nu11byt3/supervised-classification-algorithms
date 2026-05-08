#include "ConfusionMatrix.h"
#include <iomanip>
#include <algorithm>

namespace ml {

ConfusionMatrix::ConfusionMatrix(const std::vector<int>& labels)
    : classLabels(labels) {
    // Ordenar etiquetas para presentación consistente
    std::sort(classLabels.begin(), classLabels.end());
    int n = classLabels.size();
    matrix.assign(n, std::vector<int>(n, 0));
    for (size_t i = 0; i < classLabels.size(); ++i) {
        labelToIndex[classLabels[i]] = static_cast<int>(i);
    }
}

void ConfusionMatrix::add(int trueLabel, int predLabel) {
    auto itTrue = labelToIndex.find(trueLabel);
    auto itPred = labelToIndex.find(predLabel);
    if (itTrue != labelToIndex.end() && itPred != labelToIndex.end()) {
        matrix[itTrue->second][itPred->second]++;
        total++;
    } else {
        // Opcional: imprimir advertencia si hay etiquetas no esperadas
        std::cerr << "Advertencia: etiqueta no reconocida (real=" << trueLabel
                  << ", pred=" << predLabel << ")\n";
    }
}

double ConfusionMatrix::accuracy() const {
    if (total == 0) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < matrix.size(); ++i) {
        correct += matrix[i][i];
    }
    return static_cast<double>(correct) / total;
}

double ConfusionMatrix::error() const {
    return 1.0 - accuracy();
}

void ConfusionMatrix::print() const {
    int n = classLabels.size();

    // Calcular anchos de columna para formato bonito
    int maxLabelWidth = 0;
    for (int label : classLabels) {
        maxLabelWidth = std::max(maxLabelWidth, (int)std::to_string(label).length());
    }
    int cellWidth = std::max(4, maxLabelWidth + 2);

    // Cabecera
    std::cout << "\nMatriz de Confusión:\n";
    std::cout << std::setw(maxLabelWidth + 2) << " ";
    for (int label : classLabels) {
        std::cout << std::setw(cellWidth) << label;
    }
    std::cout << std::endl;

    // Filas
    for (size_t i = 0; i < classLabels.size(); ++i) {
        std::cout << std::setw(maxLabelWidth + 2) << classLabels[i];
        for (size_t j = 0; j < classLabels.size(); ++j) {
            std::cout << std::setw(cellWidth) << matrix[i][j];
        }
        std::cout << std::endl;
    }

    // Resumen
    std::cout << "Exactitud: " << std::fixed << std::setprecision(4) << accuracy()
              << " (" << accuracy() * 100 << "%)\n";
    std::cout << "Error: " << error() << std::endl;
}

} // namespace ml
