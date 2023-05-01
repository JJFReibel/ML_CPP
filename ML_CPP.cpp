#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

// C++ ML
// By JJ Reibel

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
           std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>
trainValTestSplit(std::vector<double>& X, std::vector<double>& y, double valSize, double testSize, int epochs, int64_t randomState) {

    // Get the total number of samples in the dataset
    int nSamples = X.size();

    // Set the random seed if provided
    if (randomState != 0) {
        std::mt19937 gen(randomState);
        std::shuffle(X.begin(), X.end(), gen);
        std::shuffle(y.begin(), y.end(), gen);
    }

    // Create a list of indices that correspond to the samples in the dataset
    std::vector<int> idx(nSamples);
    std::iota(idx.begin(), idx.end(), 0);

    // Shuffle the indices
    std::shuffle(idx.begin(), idx.end(), std::default_random_engine(randomState));

    // Calculate the number of samples to allocate to the validation and test sets
    int nVal = std::ceil(nSamples * valSize);
    int nTest = std::ceil(nSamples * testSize);

    // Initialize the starting and ending indices of each epoch
    std::vector<int> epochStartIdx(epochs);
    std::vector<int> epochEndIdx(epochs);
    for (int i = 0; i < epochs; i++) {
        epochStartIdx[i] = i * nSamples / epochs;
    }
    std::copy(epochStartIdx.begin() + 1, epochStartIdx.end(), epochEndIdx.begin());
    epochEndIdx.back() = nSamples;

    // Initialize the slices to hold the indices of the samples in each set for each epoch
    std::vector<std::vector<int>> trainIdxEpoch(epochs);
    std::vector<std::vector<int>> valIdxEpoch(epochs);
    std::vector<std::vector<int>> testIdxEpoch(epochs);

    // Loop through each epoch
    for (int i = 0; i < epochs; i++) {

        // Get the indices of the samples in the current epoch
        std::vector<int> epochIndices(idx.begin() + epochStartIdx[i], idx.begin() + epochEndIdx[i]);

        // Calculate the indices of the samples to allocate to the validation and test sets
        std::vector<int> valIdx(epochIndices.begin(), epochIndices.begin() + nVal);
        std::vector<int> testIdx(epochIndices.begin() + nVal, epochIndices.begin() + nVal + nTest);
        std::vector<int> trainIdx(epochIndices.begin() + nVal + nTest, epochIndices.end());

        // Add the indices to the appropriate slices for the current epoch
        trainIdxEpoch[i] = trainIdx;
        valIdxEpoch[i] = valIdx;
        testIdxEpoch[i] = testIdx;
    }

    // Initialize vectors to hold the data for each epoch
    std::vector<std::vector<double>> XTrainEpoch(epochs);
    std::vector<std::vector<double>> XValEpoch(epochs);
    std::vector<std::vector<double>> XTestEpoch(epochs);
    std::vector<std::vector<double>> yTrainEpoch(epochs);
    std::vector<std::vector<double>> yValEpoch(epochs);
    std::vector<std::vector<double>> yTestEpoch(epochs);


// Loop through each epoch
for (int i = 0; i < epochs; i++) {

    // Get the indices of the samples for the current epoch
    std::vector<int> trainIdx = trainIdxEpoch[i];
    std::vector<int> valIdx = valIdxEpoch[i];
    std::vector<int> testIdx = testIdxEpoch[i];

    // Allocate the data to the appropriate sets for the current epoch
    std::vector<double> XTrain(trainIdx.size());
    std::vector<double> XVal(valIdx.size());
    std::vector<double> XTest(testIdx.size());
    std::vector<double> yTrain(trainIdx.size());
    std::vector<double> yVal(valIdx.size());
    std::vector<double> yTest(testIdx.size());

    // Loop through each index and assign the data to the appropriate set
    for (int j = 0; j < trainIdx.size(); j++) {
        XTrain[j] = X[trainIdx[j]];
        yTrain[j] = y[trainIdx[j]];
    }
    for (int j = 0; j < valIdx.size(); j++) {
        XVal[j] = X[valIdx[j]];
        yVal[j] = y[valIdx[j]];
    }
    for (int j = 0; j < testIdx.size(); j++) {
        XTest[j] = X[testIdx[j]];
        yTest[j] = y[testIdx[j]];
    }

    // Add the data to the appropriate vector for the current epoch
    XTrainEpoch[i] = XTrain;
    XValEpoch[i] = XVal;
    XTestEpoch[i] = XTest;
    yTrainEpoch[i] = yTrain;
    yValEpoch[i] = yVal;
    yTestEpoch[i] = yTest;
}

// Return a tuple of the data for each set and each epoch
return std::make_tuple(XTrainEpoch, XValEpoch, XTestEpoch, yTrainEpoch, yValEpoch, yTestEpoch);

}






int main() {

    // Generate some sample data
    std::vector<double> X(1000);
    std::vector<double> y(1000);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    for (int i = 0; i < 1000; i++) {
        X[i] = distribution(generator);
        y[i] = X[i] + distribution(generator) * 0.1;
    }

    // Call the trainValTestSplit function
    double valSize = 0.2;
    double testSize = 0.2;
    int epochs = 5;
    int64_t randomState = 123;
    auto result = trainValTestSplit(X, y, valSize, testSize, epochs, randomState);

    // Print the shapes of the data for each set and epoch
    std::cout << "XTrainEpoch shape: ";
    for (const auto& v : std::get<0>(result)) {
        std::cout << v.size() << " ";
    }
    std::cout << "\nXValEpoch shape: ";
    for (const auto& v : std::get<1>(result)) {
        std::cout << v.size() << " ";
    }
    std::cout << "\nXTestEpoch shape: ";
    for (const auto& v : std::get<2>(result)) {
        std::cout << v.size() << " ";
    }
    std::cout << "\nyTrainEpoch shape: ";
    for (const auto& v : std::get<3>(result)) {
        std::cout << v.size() << " ";
    }
    std::cout << "\nyValEpoch shape: ";
    for (const auto& v : std::get<4>(result)) {
        std::cout << v.size() << " ";
    }
    std::cout << "\nyTestEpoch shape: ";
    for (const auto& v : std::get<5>(result)) {
        std::cout << v.size() << " ";
    }
    std::cout << std::endl;

    return 0;
}
