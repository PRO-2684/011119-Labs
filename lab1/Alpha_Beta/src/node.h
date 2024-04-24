#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace ChineseChess {
// 棋力评估，这里的棋盘方向和输入棋盘方向不同，在使用时需要仔细
// 生成合法动作代码部分已经使用，经过测试是正确的，大家可以参考
std::vector<std::vector<int>> JiangPosition = {
    // 非棋盘坐标系
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, -8, -9, 0, 0, 0, 0, 0, 0, 0},
    {5, -8, -9, 0, 0, 0, 0, 0, 0, 0},
    {1, -8, -9, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
};

std::vector<std::vector<int>> ShiPosition = {
    // 非棋盘坐标系
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 3, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
};

std::vector<std::vector<int>> XiangPosition = {
    // 非棋盘坐标系
    {0, 0, -2, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 3, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, -2, 0, 0, 0, 0, 0, 0, 0},
};

std::vector<std::vector<int>> MaPosition = {
    // 非棋盘坐标系
    {0, -3, 5, 4, 2, 2, 5, 4, 2, 2},
    {-3, 2, 4, 6, 10, 12, 20, 10, 8, 2},
    {2, 4, 6, 10, 13, 11, 12, 11, 15, 2},
    {0, 5, 7, 7, 14, 15, 19, 15, 9, 8},
    {2, -10, 4, 10, 15, 16, 12, 11, 6, 2},
    {0, 5, 7, 7, 14, 15, 19, 15, 9, 8},
    {2, 4, 6, 10, 13, 11, 12, 11, 15, 2},
    {-3, 2, 4, 6, 10, 12, 20, 10, 8, 2},
    {0, -3, 5, 4, 2, 2, 5, 4, 2, 2},
};

std::vector<std::vector<int>> PaoPosition = {
    // 非棋盘坐标系
    {0, 0, 1, 0, -1, 0, 0, 1, 2, 4},
    {0, 1, 0, 0, 0, 0, 3, 1, 2, 4},
    {1, 2, 4, 0, 3, 0, 3, 0, 0, 0},
    {3, 2, 3, 0, 0, 0, 2, -5, -4, -5},
    {3, 2, 5, 0, 4, 4, 4, -4, -7, -6},
    {3, 2, 3, 0, 0, 0, 2, -5, -4, -5},
    {1, 2, 4, 0, 3, 0, 3, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 3, 1, 2, 4},
    {0, 0, 1, 0, -1, 0, 0, 1, 2, 4},
};

std::vector<std::vector<int>> JuPosition = {
    // 非棋盘坐标系
    {-6, 5, -2, 4, 8, 8, 6, 6, 6, 6},
    {6, 8, 8, 9, 12, 11, 13, 8, 12, 8},
    {4, 6, 4, 4, 12, 11, 13, 7, 9, 7},
    {12, 12, 12, 12, 14, 14, 16, 14, 16, 13},
    {0, 0, 12, 14, 15, 15, 16, 16, 33, 14},
    {12, 12, 12, 12, 14, 14, 16, 14, 16, 13},
    {4, 6, 4, 4, 12, 11, 13, 7, 9, 7},
    {6, 8, 8, 9, 12, 11, 13, 8, 12, 8},
    {-6, 5, -2, 4, 8, 8, 6, 6, 6, 6},
};

std::vector<std::vector<int>> BingPosition = {
    // 非棋盘坐标系
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

// 棋子价值评估
std::map<std::string, int> piece_values = {
    {"Jiang", 10000},
    {"Shi", 10},
    {"Xiang", 30},
    {"Ma", 300},
    {"Ju", 500},
    {"Pao", 300},
    {"Bing", 90}};

// 行棋可能性评估，这里更多是对下一步动作的评估
std::map<std::string, int> next_move_values = {
    // {"Jiang", 9999},
    // {"Ma", 100},
    // {"Ju", 500},
    // {"Pao", 100},
    // {"Bing", -20}
    {"k", 9999},
    {"n", 100},
    {"r", 500},
    {"c", 100},
    {"p", -20}};

// 动作结构体，每个动作设置 score，可以方便剪枝 (非棋盘坐标系)
struct Move {
    int init_x;
    int init_y;
    int next_x;
    int next_y;
    int score;
};

// 定义棋盘上的棋子结构体 (非棋盘坐标系)
struct ChessPiece {
    char name;           // 棋子名称
    int init_x, init_y;  // 棋子的坐标
    bool color;          // 棋子阵营 true 为红色、false 为黑色
};

// 定义棋盘类
class ChessBoard {
   private:
    int sizeX, sizeY;                      // 棋盘大小，固定 (board[sizeX][sizeY])
    std::vector<ChessPiece> pieces;        // 棋盘上所有棋子 (非棋盘坐标系)
    std::vector<std::vector<char>> board;  // 当前棋盘、二维数组表示 (棋盘坐标系)
    std::vector<Move> red_moves;           // 红方棋子的合法动作 (非棋盘坐标系)
    std::vector<Move> black_moves;         // 黑方棋子的合法动作 (非棋盘坐标系)
   public:
    // 判断颜色，红色返回 true，黑色返回 false
    bool colorOf(char piece) {
        return piece >= 'A' && piece <= 'Z';
    }

    // 转小写
    char toLower(char piece) {
        if (colorOf(piece)) {
            return piece - 'A' + 'a';
        }
        return piece;
    }

    // 初始化棋盘，提取棋盘上棋子，并生成所有合法动作
    void initializeBoard(const std::vector<std::vector<char>> init_board) {
        board = init_board;
        sizeX = board.size();
        sizeY = board[0].size();

        for (int i = 0; i < sizeX; ++i) {
            for (int j = 0; j < sizeY; ++j) {
                char pieceChar = board[i][j];
                if (pieceChar == '.')
                    continue;

                ChessPiece piece;
                piece.init_x = j;
                piece.init_y = i;
                piece.color = colorOf(pieceChar);
                piece.name = pieceChar;
                pieces.push_back(piece);

                switch (pieceChar) {
                    case 'R':
                        generateJuMoves(j, i, piece.color);
                        break;
                    case 'C':
                        generatePaoMoves(j, i, piece.color);
                        break;
                    case 'N':
                        generateMaMoves(j, i, piece.color);
                        break;
                    case 'B':
                        generateXiangMoves(j, i, piece.color);
                        break;
                    case 'A':
                        generateShiMoves(j, i, piece.color);
                        break;
                    case 'K':
                        generateJiangMoves(j, i, piece.color);
                        break;
                    case 'P':
                        generateBingMoves(j, i, piece.color);
                        break;
                    case 'r':
                        generateJuMoves(j, i, piece.color);
                        break;
                    case 'c':
                        generatePaoMoves(j, i, piece.color);
                        break;
                    case 'n':
                        generateMaMoves(j, i, piece.color);
                        break;
                    case 'b':
                        generateXiangMoves(j, i, piece.color);
                        break;
                    case 'a':
                        generateShiMoves(j, i, piece.color);
                        break;
                    case 'k':
                        generateJiangMoves(j, i, piece.color);
                        break;
                    case 'p':
                        generateBingMoves(j, i, piece.color);
                        break;
                    default:
                        break;
                }
            }
        }
    }

    // 生成车的合法动作 (非棋盘坐标系)
    void generateJuMoves(int x, int y, bool color) {
        // 前后左右分别进行搜索，遇到棋子停止，不同阵营可以吃掉
        std::vector<Move> JuMoves;
        for (int i = x + 1; i < sizeY; i++) {
            Move cur_move = {x, y, i, y, 0};
            if (board[y][i] != '.') {
                bool cur_color = colorOf(board[y][i]);
                if (cur_color != color) {
                    JuMoves.push_back(cur_move);
                }
                break;
            }
            JuMoves.push_back(cur_move);
        }

        for (int i = x - 1; i >= 0; i--) {
            Move cur_move = {x, y, i, y, 0};
            if (board[y][i] != '.') {
                bool cur_color = colorOf(board[y][i]);
                if (cur_color != color) {
                    JuMoves.push_back(cur_move);
                }
                break;
            }
            JuMoves.push_back(cur_move);
        }

        for (int j = y + 1; j < sizeX; j++) {
            Move cur_move = {x, y, x, j, 0};
            if (board[j][x] != '.') {
                bool cur_color = colorOf(board[j][x]);
                if (cur_color != color) {
                    JuMoves.push_back(cur_move);
                }
                break;
            }
            JuMoves.push_back(cur_move);
        }

        for (int j = y - 1; j >= 0; j--) {
            Move cur_move = {x, y, x, j, 0};
            if (board[j][x] != '.') {
                bool cur_color = colorOf(board[j][x]);
                if (cur_color != color) {
                    JuMoves.push_back(cur_move);
                }
                break;
            }
            JuMoves.push_back(cur_move);
        }
        for (int i = 0; i < JuMoves.size(); i++) {
            if (color) {
                JuMoves[i].score = JuPosition[JuMoves[i].next_x][9 - JuMoves[i].next_y] - JuPosition[x][9 - y];
                red_moves.push_back(JuMoves[i]);
            } else {
                JuMoves[i].score = JuPosition[JuMoves[i].next_x][JuMoves[i].next_y] - JuPosition[x][y];
                black_moves.push_back(JuMoves[i]);
            }
        }
    }

    // 生成马的合法动作 (非棋盘坐标系)
    void generateMaMoves(int x, int y, bool color) {
        // 遍历所有可能动作，筛选
        std::vector<Move> MaMoves;
        int dx[] = {2, 1, -1, -2, -2, -1, 1, 2};
        int dy[] = {1, 2, 2, 1, -1, -2, -2, -1};
        // 简化，不考虑拌马脚
        // TODO: 可以实现拌马脚过程
        for (int i = 0; i < 8; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx < 0 || nx >= 9 || ny < 0 || ny >= 10)
                continue;
            Move cur_move = {x, y, nx, ny, 0};
            if (board[ny][nx] != '.') {
                // 注意棋盘坐标系，这里 nx、ny 相反是正确的
                bool cur_color = colorOf(board[ny][nx]);
                if (cur_color != color) {
                    MaMoves.push_back(cur_move);  // 可以吃子
                }
                continue;
            }
            MaMoves.push_back(cur_move);  // 可以走
        }
        for (int i = 0; i < MaMoves.size(); i++) {
            if (color) {
                MaMoves[i].score = MaPosition[MaMoves[i].next_x][9 - MaMoves[i].next_y] - MaPosition[x][9 - y];
                red_moves.push_back(MaMoves[i]);
            } else {
                MaMoves[i].score = MaPosition[MaMoves[i].next_x][MaMoves[i].next_y] - MaPosition[x][y];
                black_moves.push_back(MaMoves[i]);
            }
        }
    }

    // 生成炮的合法动作 (非棋盘坐标系)
    void generatePaoMoves(int x, int y, bool color) {
        // 和车生成动作相似，需要考虑炮翻山吃子的情况
        std::vector<Move> PaoMoves;
        // FIXME:
        for (int i = x + 1; i < sizeY; i++) {
            Move cur_move = {x, y, i, y, 0};
            if (board[y][i] != '.') {
                int next_x = -1;
                for (int j = i + 1; j < sizeY; j++) {  // 遍历后续位置
                    if (board[y][j] != '.') {          // 遇到棋子
                        bool cur_color = colorOf(board[y][j]);
                        if (cur_color != color) {  // 遇到对方棋子
                            next_x = j;            // 可以吃子
                        }
                        break;
                    }
                }
                if (next_x != -1) {  // 可以吃子
                    cur_move.next_x = next_x;
                } else {  // 不能吃子 - 不能走
                    break;
                }
            }
            PaoMoves.push_back(cur_move);
        }

        for (int i = 0; i < PaoMoves.size(); i++) {
            if (color) {
                PaoMoves[i].score = PaoPosition[PaoMoves[i].next_x][9 - PaoMoves[i].next_y] - PaoPosition[x][9 - y];
                red_moves.push_back(PaoMoves[i]);
            } else {
                PaoMoves[i].score = PaoPosition[PaoMoves[i].next_x][PaoMoves[i].next_y] - PaoPosition[x][y];
                black_moves.push_back(PaoMoves[i]);
            }
        }
    }

    // 生成相的合法动作 (非棋盘坐标系)
    void generateXiangMoves(int x, int y, bool color) {
        std::vector<Move> XiangMoves;
        // FIXME:
        int dx[] = {2, 2, -2, -2};
        int dy[] = {2, -2, 2, -2};
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx < 0 || nx >= 9 || ny < 0 || ny >= 10)
                continue;
            if (color && ny < 5 || !color && ny >= 5)
                continue;  // 相不能过河
            Move cur_move = {x, y, nx, ny, 0};
            if (board[ny][nx] != '.') {
                bool cur_color = colorOf(board[ny][nx]);  // 挡路棋子颜色
                // 注意棋盘坐标系，这里 nx、ny 相反是正确的
                if (cur_color != color) {
                    XiangMoves.push_back(cur_move);  // 可以吃子
                }
                continue;
            }
            XiangMoves.push_back(cur_move);  // 可以走
        }

        for (int i = 0; i < XiangMoves.size(); i++) {
            if (color) {
                XiangMoves[i].score = XiangPosition[XiangMoves[i].next_x][9 - XiangMoves[i].next_y] - XiangPosition[x][9 - y];
                red_moves.push_back(XiangMoves[i]);
            } else {
                XiangMoves[i].score = XiangPosition[XiangMoves[i].next_x][XiangMoves[i].next_y] - XiangPosition[x][y];
                black_moves.push_back(XiangMoves[i]);
            }
        }
    }

    // 生成士的合法动作 (非棋盘坐标系)
    void generateShiMoves(int x, int y, bool color) {
        std::vector<Move> ShiMoves;
        // FIXME:
        int dx[] = {1, 1, -1, -1};
        int dy[] = {1, -1, 1, -1};
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx < 3 || nx >= 6 || ny < 0 || ny >= 10)
                continue;  // 士的活动范围
            if (color && ny < 7 || !color && ny >= 3)
                continue;  // 士不能出此方的九宫格
            Move cur_move = {x, y, nx, ny, 0};
            if (board[ny][nx] != '.') {
                bool cur_color = colorOf(board[ny][nx]);  // 挡路棋子颜色
                // 注意棋盘坐标系，这里 nx、ny 相反是正确的
                if (cur_color != color) {
                    ShiMoves.push_back(cur_move);  // 可以吃子
                }
                continue;
            }
            ShiMoves.push_back(cur_move);  // 可以走
        }

        for (int i = 0; i < ShiMoves.size(); i++) {
            if (color) {
                ShiMoves[i].score = ShiPosition[ShiMoves[i].next_x][9 - ShiMoves[i].next_y] - ShiPosition[x][9 - y];
                red_moves.push_back(ShiMoves[i]);
            } else {
                ShiMoves[i].score = ShiPosition[ShiMoves[i].next_x][ShiMoves[i].next_y] - ShiPosition[x][y];
                black_moves.push_back(ShiMoves[i]);
            }
        }
    }

    // 生成将的合法动作 (非棋盘坐标系)
    void generateJiangMoves(int x, int y, bool color) {
        std::vector<Move> JiangMoves;
        // FIXME:
        int dx[] = {1, 0, -1, 0};
        int dy[] = {0, 1, 0, -1};
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx < 3 || nx >= 6 || ny < 0 || ny >= 10)
                continue;  // 将的活动范围
            if (color && ny < 7 || !color && ny >= 3)
                continue;  // 将不能出此方的九宫格
            Move cur_move = {x, y, nx, ny, 0};
            if (board[ny][nx] != '.') {
                bool cur_color = colorOf(board[ny][nx]);  // 挡路棋子颜色
                // 注意棋盘坐标系，这里 nx、ny 相反是正确的
                if (cur_color != color) {
                    JiangMoves.push_back(cur_move);  // 可以吃子
                }
                continue;
            }
            JiangMoves.push_back(cur_move);  // 可以走
        }

        for (int i = 0; i < JiangMoves.size(); i++) {
            if (color) {
                JiangMoves[i].score = JiangPosition[JiangMoves[i].next_x][9 - JiangMoves[i].next_y] - JiangPosition[x][9 - y];
                red_moves.push_back(JiangMoves[i]);
            } else {
                JiangMoves[i].score = JiangPosition[JiangMoves[i].next_x][JiangMoves[i].next_y] - JiangPosition[x][y];
                black_moves.push_back(JiangMoves[i]);
            }
        }
    }

    // 生成兵的合法动作 (非棋盘坐标系)
    void generateBingMoves(int x, int y, bool color) {
        // 需要分条件考虑，小兵在过楚河汉界之前只能前进，之后可以左右前
        std::vector<Move> BingMoves;
        // FIXME:
        if (color) {  // 红色方
            Move cur_move = {x, y, x, y - 1, 0};
            if (board[y - 1][x] == '.' || colorOf(board[y - 1][x]) == false) {
                BingMoves.push_back(cur_move);  // 可以前进或吃子
            }
            if (y < 5) {  // 过河之后
                cur_move = {x, y, x + 1, y, 0};
                if (x < 8 && (board[y][x + 1] == '.' || colorOf(board[y][x + 1]) == false)) {
                    BingMoves.push_back(cur_move);
                }
                cur_move = {x, y, x - 1, y, 0};
                if (x > 0 && (board[y][x - 1] == '.' || colorOf(board[y][x - 1]) == false)) {
                    BingMoves.push_back(cur_move);
                }
            }
        } else {  // 黑色方
            Move cur_move = {x, y, x, y + 1, 0};
            if (board[y + 1][x] == '.' || colorOf(board[y + 1][x]) == true) {
                BingMoves.push_back(cur_move);  // 可以前进或吃子
            }
            if (y >= 5) {  // 过河之后
                cur_move = {x, y, x + 1, y, 0};
                if (x < 8 && (board[y][x + 1] == '.' || colorOf(board[y][x + 1]) == true)) {
                    BingMoves.push_back(cur_move);
                }
                cur_move = {x, y, x - 1, y, 0};
                if (x > 0 && (board[y][x - 1] == '.' || colorOf(board[y][x - 1]) == true)) {
                    BingMoves.push_back(cur_move);
                }
            }
        }

        for (int i = 0; i < BingMoves.size(); i++) {
            if (color) {
                BingMoves[i].score = BingPosition[BingMoves[i].next_x][9 - BingMoves[i].next_y] - BingPosition[x][9 - y];
                red_moves.push_back(BingMoves[i]);
            } else {
                BingMoves[i].score = BingPosition[BingMoves[i].next_x][BingMoves[i].next_y] - BingPosition[x][y];
                black_moves.push_back(BingMoves[i]);
            }
        }
    }

    // 终止判断
    bool judgeTermination() {
        // FIXME:
        // 判断是否结束
        bool red_king = false;    // 是否有红将
        bool black_king = false;  // 是否有黑将
        for (int i = 0; i < pieces.size(); i++) {
            if (pieces[i].name == 'K') {
                red_king = true;
            } else if (pieces[i].name == 'k') {
                black_king = true;
            }
        }
        return !red_king || !black_king;
    }

    // 棋盘分数评估，根据当前棋盘进行棋子价值和棋力评估，max 玩家减去 min 玩家分数
    int evaluateNode() {
        // FIXME:
        int red_score = 0;
        int black_score = 0;
        for (int i = 0; i < pieces.size(); i++) {
            int x = pieces[i].init_x;
            int y = pieces[i].init_y;
            char name = pieces[i].name;
            char dest = board[y][x];
            // 棋力评估：对每个棋子的位置进行评分
            switch (name) {
                case 'R':
                    red_score += JuPosition[x][9 - y];
                    break;
                case 'C':
                    red_score += PaoPosition[x][9 - y];
                    break;
                case 'N':
                    red_score += MaPosition[x][9 - y];
                    break;
                case 'B':
                    red_score += XiangPosition[x][9 - y];
                    break;
                case 'A':
                    red_score += ShiPosition[x][9 - y];
                    break;
                case 'K':
                    red_score += JiangPosition[x][9 - y];
                    break;
                case 'P':
                    red_score += BingPosition[x][9 - y];
                    break;
                case 'r':
                    black_score += JuPosition[x][y];
                    break;
                case 'c':
                    black_score += PaoPosition[x][y];
                    break;
                case 'n':
                    black_score += MaPosition[x][y];
                    break;
                case 'b':
                    black_score += XiangPosition[x][y];
                    break;
                case 'a':
                    black_score += ShiPosition[x][y];
                    break;
                case 'k':
                    black_score += JiangPosition[x][y];
                    break;
                case 'p':
                    black_score += BingPosition[x][y];
                    break;
                default:
                    break;
            }
            // 行棋可能性评估：根据棋子下一步的可能动作来判断行棋的优劣
            if (colorOf(name)) {
                red_score += next_move_values[std::string(1, toLower(dest))];
            } else {
                black_score += next_move_values[std::string(1, toLower(dest))];
            }
            // 棋子价值评估：各棋子的固定价值
            if (colorOf(name)) {
                red_score += piece_values[std::string(1, toLower(dest))];
            } else {
                black_score += piece_values[std::string(1, toLower(dest))];
            }
        }
        return red_score - black_score;
    }

    // 测试接口
    std::vector<Move>* getMoves(bool color) {
        if (color)
            return &red_moves;
        return &black_moves;
    }

    std::vector<ChessPiece> getChessPiece() {  // Unused?
        return pieces;
    }

    std::vector<std::vector<char>>* getBoard() {
        return &board;
    }
};

// 定义博弈树节点类
class GameTreeNode {
   private:
    bool color;                           // 当前玩家类型，true为红色方、false为黑色方
    ChessBoard board;                     // 当前棋盘状态
    std::vector<GameTreeNode*> children;  // 子节点列表
    int evaluationScore;                  // 棋盘评估分数

   public:
    // 构造函数
    GameTreeNode(bool color, std::vector<std::vector<char>> initBoard, int evaluationScore)
        : color(color), evaluationScore(evaluationScore) {
        board.initializeBoard(initBoard);
        children.clear();
    }

    // 根据当前棋盘和动作构建新棋盘（子节点）
    GameTreeNode* updateBoard(std::vector<std::vector<char>>* cur_board, Move move, bool color) {
        // FIXME:
        std::vector<std::vector<char>> next_board = *cur_board;
        next_board[move.next_y][move.next_x] = next_board[move.init_y][move.init_x];
        next_board[move.init_y][move.init_x] = '.';
        GameTreeNode* next_node = new GameTreeNode(!color, next_board, 0);
        return next_node;
    }

    // 返回节点评估分数
    int getEvaluationScore() {
        evaluationScore = board.evaluateNode();
        return evaluationScore;
    }

    // 返回棋盘类
    ChessBoard* getBoardClass() {
        return &board;
    }

    // 返回子节点列表 (Lazy Evaluation)
    std::vector<GameTreeNode*>* getChildren() {
        // FIXME:
        if (children.empty()) {
            std::vector<std::vector<char>>* cur_board = board.getBoard();
            std::vector<Move>* moves = board.getMoves(color);
            // 为合法动作创建子节点
            for (int i = 0; i < moves->size(); i++) {
                GameTreeNode* child = updateBoard(cur_board, moves->at(i), color);
                children.push_back(child);
            }
        }
        return &children;
    }

    ~GameTreeNode() {
        for (GameTreeNode* child : children) {
            delete child;
        }
    }
};

}  // namespace ChineseChess