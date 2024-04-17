#include <iostream>
#include <vector>

std::vector<std::vector<int>> position = {
    {0, 0, 0, -2, 3, 10, 20, 20, 20, 0},
    {0, 0, 0, 0, 0, 18, 27, 30, 30, 0},
    {0, 0, 0, -2, 4, 22, 30, 45, 50, 0},
    {0, 0, 0, 0, 0, 35, 40, 55, 65, 2},
    {0, 0, 0, 6, 7, 40, 42, 55, 70, 4},
    {0, 0, 0, 0, 0, 35, 40, 55, 65, 2},
    {0, 0, 0, -2, 4, 22, 30, 45, 50, 0},
    {0, 0, 0, 0, 0, 18, 27, 30, 30, 0},
    {0, 0, 0, -2, 3, 10, 20, 20, 20, 0},
};

// "Transpose" along the secondary diagonal
std::vector<std::vector<int>> transposeSec(const std::vector<std::vector<int>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<int>> transposed(cols, std::vector<int>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[cols - 1 - j][rows - 1 - i] = matrix[i][j];
        }
    }

    return transposed;
}

int main() {
    std::vector<std::vector<int>> transposed = transposeSec(position);
    // Output the transposed matrix
    for (int i = 0; i < transposed.size(); ++i) {
        std::cout << "{";
        for (int j = 0; j < transposed[0].size(); ++j) {
            std::cout << transposed[i][j] << ", ";
        }
        std::cout << "}," << std::endl;
    }
    return 0;
}

