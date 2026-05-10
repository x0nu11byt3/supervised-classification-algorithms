#ifndef UCI_LOADER_H
#define UCI_LOADER_H

#include "Dataset.h"
#include <string>

namespace ml {

    class UCILoader {
        public:
            // Load from .data file (assumes CSV format without headers, with the last column containing the class)
            static void loadData(const std::string& dataFilePath, Dataset& dataset);

            // Optional: read .name to retrieve attribute names
            static std::vector<std::string> loadAttributeNames(const std::string& nameFilePath);
    };

} // namespace ml

#endif
