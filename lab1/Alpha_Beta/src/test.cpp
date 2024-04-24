#include <fstream>
#include "node.h"

using namespace ChineseChess;

// 博弈树搜索，depth为搜索深度
int alphaBeta(GameTreeNode * node, int alpha, int beta, int depth, bool isMaximizer) {
    std::vector<GameTreeNode *>* children = node->getChildren();
    if (depth == 0 || children->empty()) { // 叶子节点或者搜索到最大深度
        return node->getEvaluationScore();
    }
    // FIXME: alpha-beta 剪枝过程
    if (isMaximizer) {
        int maxEval = std::numeric_limits<int>::min();
        for (GameTreeNode * child : *children) {
            int eval = alphaBeta(child, alpha, beta, depth - 1, false);
            maxEval = std::max(maxEval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha) {
                break;
            }
        }
        return maxEval;
    } else {
        int minEval = std::numeric_limits<int>::max();
        for (GameTreeNode * child : *children) {
            int eval = alphaBeta(child, alpha, beta, depth - 1, true);
            minEval = std::min(minEval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha) {
                break;
            }
        }
        return minEval;
    }
}

int main(int argc, char* argv[]) {
    int fn = 1; // 默认文件编号
    if (argc > 1) { // 命令行参数
        fn = std::stoi(argv[1]);
    }
    std::string filename = "../input/" + std::to_string(fn) + ".txt";
    std::ifstream file(filename);

    std::vector<std::vector<char>> board;

    std::string line;
    int n = 0;
    while (std::getline(file, line)) {
        std::vector<char> row;

        for (char ch : line) {
            row.push_back(ch);
        }

        board.push_back(row);
        n = n + 1;
        if (n >= 10)
            break;
    }

    file.close();
    GameTreeNode root(true, board, std::numeric_limits<int>::min());

    // DONE: 调用 alphaBeta 函数
    int depth = 3;
    int score = alphaBeta(&root, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), depth, true);
    std::cout << "The best score is " << score << std::endl;

    // 代码测试
    ChessBoard* _board = root.getBoardClass();
    std::vector<std::vector<char>>* cur_board = _board->getBoard();

    for (int i = 0; i < cur_board->size(); i++) {
        for (int j = 0; j < cur_board->at(0).size(); j++) {
            std::cout << cur_board->at(i).at(j);
        }
        std::cout << std::endl;
    }

    std::vector<Move>* red_moves = _board->getMoves(true);
    std::vector<Move>* black_moves = _board->getMoves(false);

    std::cout << "Red moves: " << std::endl;
    for (int i = 0; i < red_moves->size(); i++) {
        std::cout << "(" << red_moves->at(i).init_x << ", " << red_moves->at(i).init_y << ") -> ";
        std::cout << "(" << red_moves->at(i).next_x << ", " << red_moves->at(i).next_y << ") ";
        std::cout << "score " << red_moves->at(i).score << std::endl;
    }
    std::cout << "Black moves: " << std::endl;
    for (int i = 0; i < black_moves->size(); i++) {
        std::cout << "(" << black_moves->at(i).init_x << ", " << black_moves->at(i).init_y << ") -> ";
        std::cout << "(" << black_moves->at(i).next_x << ", " << black_moves->at(i).next_y << ") ";
        std::cout << "score " << black_moves->at(i).score << std::endl;
    }

    // std::string output_filename = "../output/output_" + std::to_string(fn) + ".txt";
    // std::vector<GameTreeNode *> children = root.getChildren();
    // Move* best_move;
    // for (GameTreeNode * child : children) {
    //     if (child->getEvaluationScore() == score) {
    //         best_move = child->getMove();
    //         break;
    //     }
    // }

    return 0;
}