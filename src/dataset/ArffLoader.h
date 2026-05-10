#ifndef ARFF_LOADER_H
#define ARFF_LOADER_H

#include "Dataset.h"
#include <string>
#include <vector>
#include <fstream>

namespace ml {

class ArffLoader {

    public:
        // Loads an ARFF file into a Dataset object.
        // Parameters:
        //   filename: path to the .arff file
        //   dataset:  Dataset object where the data will be stored
        //   classIndex: index of the attribute to be used as the class (0-based).
        //               If it is -1 (default), the last attribute is used.
        static void load(const std::string& filename, Dataset& dataset, int classIndex = -1);

    private:
        struct Attribute {
            std::string name;
            bool isNumeric;                     // true = numeric, false = nominal
            std::vector<std::string> nominalValues; // list of nominal values in order
        };

        static void parseHeader(std::ifstream& file, std::vector<Attribute>& attributes, std::string& relationName);
        static void parseData(std::ifstream& file, const std::vector<Attribute>& attributes, int classIndex, Dataset& dataset);
        static std::string trim(const std::string& str);
        static std::vector<std::string> split(const std::string& str, char delimiter);
    };

} // namespace ml

#endif
