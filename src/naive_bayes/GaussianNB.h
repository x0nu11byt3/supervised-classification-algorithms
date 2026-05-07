#ifndef GAUSSIAN_NB_H
#define GAUSSIAN_NB_H

#include "../Classifier.h"
#include <vector>
#include <map>

namespace ml {

class GaussianNB : public Classifier {
public:
    void train(const Dataset& data) override;
    int predict(const std::vector<double>& features) const override;

private:
    struct ClassStats {
        double prior;
        std::vector<double> mean;
        std::vector<double> var; // variance
    };
    std::map<int, ClassStats> model;
    double logPDF(double x, double mean, double var) const;
};

} // namespace ml

#endif
