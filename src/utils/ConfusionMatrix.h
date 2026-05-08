#ifndef CONFUSION_MATRIX_H
#define CONFUSION_MATRIX_H

#include <vector>
#include <map>
#include <iostream>

namespace ml {

class ConfusionMatrix {
public:
    // Constructor: recibe las etiquetas únicas (p.ej. {0,1,2} o {1,2,3})
    ConfusionMatrix(const std::vector<int>& labels);

    // Añade un par (real, predicho)
    void add(int trueLabel, int predLabel);

    // Calcula la exactitud
    double accuracy() const;

    // Calcula el error (1 - accuracy)
    double error() const;

    // Imprime la matriz formateada (similar a las imágenes)
    void print() const;

    // Devuelve la matriz cruda (para pruebas)
    const std::vector<std::vector<int>>& getMatrix() const { return matrix; }

private:
    std::vector<int> classLabels;           // etiquetas ordenadas
    std::map<int, int> labelToIndex;        // mapeo etiqueta -> índice
    std::vector<std::vector<int>> matrix;   // matriz n x n
    int total = 0;
};

} // namespace ml

#endif
