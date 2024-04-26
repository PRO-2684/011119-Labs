#include <fstream>
#include "node.h"

using namespace ChineseChess;

void showMove(Move* move) {
    std::cout << "(" << move->init_x << ", " << move->init_y << ") -> ";
    std::cout << "(" << move->next_x << ", " << move->next_y << ")";
}

// 博弈树搜索，depth为搜索深度
int alphaBeta(GameTreeNode* node, int alpha, int beta, int depth, bool isMaximizer, Move* best_move = nullptr) {
    if (depth == 0 || node->getBoardClass()->judgeTermination()) {  // 叶子节点，或者搜索到最大深度，或者棋局结束
        return node->getEvaluationScore();
    }
    std::vector<GameTreeNode*>* children = node->getChildren();
    if (children->empty()) {
        return node->getEvaluationScore();
    }
    // FIXME: alpha-beta 剪枝过程
    if (isMaximizer) {
        int maxEval = std::numeric_limits<int>::min();
        for (GameTreeNode* child : *children) {
            int eval = alphaBeta(child, alpha, beta, depth - 1, false);
            if (eval > maxEval) {
                maxEval = eval;
                if (best_move != nullptr) {
                    best_move->init_x = child->move->init_x;
                    best_move->init_y = child->move->init_y;
                    best_move->next_x = child->move->next_x;
                    best_move->next_y = child->move->next_y;
                    best_move->score = eval;
                }
            }
            alpha = std::max(alpha, eval);
            if (beta <= alpha) {
                break;
            }
        }
        return maxEval;
    } else {
        int minEval = std::numeric_limits<int>::max();
        for (GameTreeNode* child : *children) {
            int eval = alphaBeta(child, alpha, beta, depth - 1, true);
            if (eval < minEval) {
                minEval = eval;
                if (best_move != nullptr) {
                    best_move->init_x = child->move->init_x;
                    best_move->init_y = child->move->init_y;
                    best_move->next_x = child->move->next_x;
                    best_move->next_y = child->move->next_y;
                    best_move->score = eval;
                }
            }
            beta = std::min(beta, eval);
            if (beta <= alpha) {
                break;
            }
        }
        return minEval;
    }
}

void solve(int fn, int maxDepth, bool debug) {
    std::cout << "Solving file " << fn << std::endl;
    std::string filename = "../input/" + std::to_string(fn) + ".txt";
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<char>> board;
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
    Move best_move;
    int score = alphaBeta(&root, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), maxDepth, true, &best_move);
    std::cout << "The best score is " << score << std::endl;

    // 代码测试
    ChessBoard* _board = root.getBoardClass();
    std::vector<std::vector<char>>* cur_board = _board->getBoard();

    if (debug) {
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
            showMove(&red_moves->at(i));
            std::cout << " score " << red_moves->at(i).score << std::endl;
        }
        std::cout << "Black moves: " << std::endl;
        for (int i = 0; i < black_moves->size(); i++) {
            showMove(&black_moves->at(i));
            std::cout << " score " << black_moves->at(i).score << std::endl;
        }
    }

    std::cout << "The best move is: ";
    showMove(&best_move);
    std::cout << std::endl;

    // Output (eg: K (4,0) (5,0))
    std::string output_filename = "../output/output_" + std::to_string(fn) + ".txt";
    std::ofstream output_file(output_filename);
    char piece = cur_board->at(best_move.init_y).at(best_move.init_x);
    // 输出坐标系
    output_file << piece << " (" << best_move.init_x << ", " << 9 - best_move.init_y << ") ("
                << best_move.next_x << ", " << 9 - best_move.next_y << ")" << std::endl;
}

int main(int argc, char* argv[]) {
    // 命令行参数：第一个参数为文件编号，第二个参数为搜索深度
    int fn = -1;        // 默认文件编号
    int maxDepth = 4;  // 默认搜索深度
    if (argc > 1) {    // 命令行参数 fn
        fn = std::stoi(argv[1]);
    }
    if (argc > 2) {  // 命令行参数 maxDepth
        maxDepth = std::stoi(argv[2]);
    }

    if (fn == -1) {
        for (int i = 1; i <= 10; i++) {
            solve(i, maxDepth, false);
        }
    } else {
        solve(fn, maxDepth, true);
    }
}
