#include <iostream>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

// Data structures to hold the training data and user/item averages
unordered_map<int, unordered_map<int, double>> trainTable; // Maps UserID -> ItemID -> Rating
// unordered_map<int, unordered_map<int, double>> similarityCache; // Caches user-user/item-item similarity

// Model parameters
int numUsers = 0;            // Total number of users
int numItems = 0;            // Total number of items
double lambda = 0.1;         // Regularization term
double alpha = 0.015;        // Learning rate
int numFactors = 9;          // Number of latent factors
int numIterations = 50;      // Number of SGD iterations
double globalAvgRating = 0.0; // Global average rating

// Biases and latent factors
unordered_map<int, double> userBiases; // Maps UserID -> Bias
unordered_map<int, double> itemBiases; // Maps ItemID -> Bias
vector<vector<double>> U, V;           // Latent factors for users (U) and items (V)

// Initialize latent factors and biases
void initializeMatrices() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-0.01, 0.01);

    U.assign(numUsers, vector<double>(numFactors));
    V.assign(numItems, vector<double>(numFactors));

    for (auto& userFactors : U) {
        generate(userFactors.begin(), userFactors.end(), [&]() { return dis(gen); });
    }
    for (auto& itemFactors : V) {
        generate(itemFactors.begin(), itemFactors.end(), [&]() { return dis(gen); });
    }

    userBiases.reserve(numUsers);
    itemBiases.reserve(numItems);

    for (int i = 0; i < numUsers; ++i) {
    userBiases[i] = 0.0;
    }

    for (int i = 0; i < numItems; ++i) {
     itemBiases[i] = 0.0;
    }

   // cout << "Matrices and biases initialized successfully." << endl;
}

// Compute global average rating
void computeGlobalAvgRating() {
    double totalRating = 0.0;
    int count = 0;

    for (const auto& user : trainTable) {
        for (const auto& item : user.second) {
            totalRating += item.second;
            count++;
        }
    }

    if (count > 0) {
    globalAvgRating = totalRating / count;
    } 
    else {
    globalAvgRating = 0.0;
    }

}

// Predict rating using matrix factorization
double predictRating(int userID, int itemID) {
    if (userID < 0 || userID >= numUsers || itemID < 0 || itemID >= numItems) {
    return globalAvgRating;
    }

    double prediction = globalAvgRating + userBiases[userID] + itemBiases[itemID];
    for (int k = 0; k < numFactors; ++k) {
        prediction += U[userID][k] * V[itemID][k];
    }
    return max(1.0, min(5.0, prediction));
}

// Stochastic Gradient Descent (SGD) for training
void stochasticGradientDescent() {
    for (int iter = 0; iter < numIterations; ++iter) {

        for (const auto& user : trainTable) {

            int userID = user.first;
            for (const auto& item : user.second) {

                int itemID = item.first;
                double rating = item.second;

                double error = rating - predictRating(userID, itemID);
                userBiases[userID] += alpha * (error - lambda * userBiases[userID]);
                itemBiases[itemID] += alpha * (error - lambda * itemBiases[itemID]);

                for (int k = 0; k < numFactors; ++k) {

                    double oldU = U[userID][k];
                    U[userID][k] += alpha * (error * V[itemID][k] - lambda * U[userID][k]);
                    V[itemID][k] += alpha * (error * oldU - lambda * V[itemID][k]);

                }
            }
        }
    }
}







// // Compute cosine similarity
// double cosineSimilarity(const unordered_map<int, double>& vec1, const unordered_map<int, double>& vec2) {
//     double dotProduct = 0.0, magnitude1 = 0.0, magnitude2 = 0.0;

//     for (const auto& pair : vec1) {
//         if (vec2.count(pair.first)) {
//             dotProduct += vec1.at(pair.first) * vec2.at(pair.first);
//         }
//         magnitude1 += pair.second * pair.second;
//     }
//     for (const auto& pair : vec2) {
//         magnitude2 += pair.second * pair.second;
//     }

//     if (magnitude1 == 0.0 || magnitude2 == 0.0) return 0.0;
//     return dotProduct / (sqrt(magnitude1) * sqrt(magnitude2));
// }

// // Precompute similarities
// void computeSimilarities(bool userBased = true) {
//     if (userBased) {
//         for (const auto& user1 : trainTable) {
//             for (const auto& user2 : trainTable) {
//                 if (user1.first >= user2.first) continue;
//                 double similarity = cosineSimilarity(user1.second, user2.second);
//                 similarityCache[user1.first][user2.first] = similarity;
//                 similarityCache[user2.first][user1.first] = similarity;
//             }
//         }
//     } else {
//         unordered_map<int, unordered_map<int, double>> itemRatings;
//         for (const auto& user : trainTable) {
//             for (const auto& item : user.second) {
//                 itemRatings[item.first][user.first] = item.second;
//             }
//         }
//         for (const auto& item1 : itemRatings) {
//             for (const auto& item2 : itemRatings) {
//                 if (item1.first >= item2.first) continue;
//                 double similarity = cosineSimilarity(item1.second, item2.second);
//                 similarityCache[item1.first][item2.first] = similarity;
//                 similarityCache[item2.first][item1.first] = similarity;
//             }
//         }
//     }
// }

// // Predict rating combining matrix factorization and cosine similarity
// double predictRatingCombined(int userID, int itemID, double beta = 0.2) {
//     double mfPrediction = predictRating(userID, itemID);

//     double similarityCorrection = 0.0, similaritySum = 0.0;

//     if (similarityCache.count(userID)) {

//         for (const auto& neighbor : similarityCache[userID]) {

//             int neighborID = neighbor.first;
//             if (trainTable[neighborID].count(itemID)) {
//                 similarityCorrection += neighbor.second * trainTable[neighborID][itemID];
//                 similaritySum += fabs(neighbor.second);

//             }
//         }
//     }

//     if (similaritySum > 0) {
//         similarityCorrection /= similaritySum;
//     }

//     return (1 - beta) * mfPrediction + beta * similarityCorrection;
// }







int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    string line;
    bool isTest = false;
    vector<pair<int, int>> testPairs;

    while (getline(cin, line)) {
        if (line == "train dataset") {
            isTest = false;
            continue;
        } else if (line == "test dataset") {
            isTest = true;
            continue;
        }

        stringstream ss(line);
        int userId, itemId;
        double rating;

        if (isTest) {
            ss >> userId >> itemId;
            testPairs.emplace_back(userId, itemId);
        } else {
            ss >> userId >> itemId >> rating;
            trainTable[userId][itemId] = rating;
            numUsers = max(numUsers, userId + 1);
            numItems = max(numItems, itemId + 1);
        }
    }

    computeGlobalAvgRating();
    initializeMatrices();
    // computeSimilarities(true); // Compute user-user similarity
    stochasticGradientDescent();

    for (const auto& testPair : testPairs) {
        int userID = testPair.first;
        int itemID = testPair.second;
        double predictedRating = predictRating(userID, itemID); // Use pure matrix factorization
        cout << fixed << setprecision(2) << predictedRating << '\n';
    }

    return 0;
}
