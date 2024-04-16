#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

struct Map_Cell {
    int type;  // 0: 可通行, 1: 不可通行, 2: 补给点, 3: 起点, 4: 终点
    // DONE: 定义地图信息
    int x;  // 横坐标
    int y;  // 纵坐标
};

struct Search_Cell {
    int h;  // 启发式函数值
    int g;  // 起点到当前点的代价
    // DONE: 定义搜索状态
    int x;                // 当前横坐标
    int y;                // 当前纵坐标
    int q;                // 剩余步数
    Search_Cell* parent;  // 上一节点
};

// 自定义比较函数对象，按照 Search_Cell 结构体的 g + h 属性进行比较
struct CompareF {
    bool operator()(const Search_Cell* a, const Search_Cell* b) const {
        return (a->g + a->h) > (b->g + b->h);  // 较小的 g + h 值优先级更高
    }
};

// DONE: 定义启发式函数：当前节点到终点的曼哈顿距离
int Heuristic_Funtion(Search_Cell* cell, pair<int, int> end_point) {
    // return 0; // 无启发式函数时，退化为一致代价搜索
    return abs(cell->x - end_point.first) + abs(cell->y - end_point.second);
}

void Astar_search(const string input_file, int& step_nums, string& way) {
    ifstream file(input_file);
    if (!file.is_open()) {
        cout << "Error opening file!" << endl;
        return;
    }

    string line;
    getline(file, line);  // 读取第一行
    stringstream ss(line);
    string word;
    vector<string> words;
    while (ss >> word) {
        words.push_back(word);
    }
    int M = stoi(words[0]);  // 行数
    int N = stoi(words[1]);  // 列数
    int T = stoi(words[2]);  // 最大步数

    pair<int, int> start_point;  // 起点
    pair<int, int> end_point;    // 终点
    Map_Cell** Map = new Map_Cell*[M];
    // 加载地图
    for (int i = 0; i < M; i++) {
        Map[i] = new Map_Cell[N];
        getline(file, line);
        stringstream ss(line);
        string word;
        vector<string> words;
        while (ss >> word) {
            words.push_back(word);
        }
        for (int j = 0; j < N; j++) {
            Map[i][j].type = stoi(words[j]);
            Map[i][j].x = i;
            Map[i][j].y = j;
            if (Map[i][j].type == 3) {
                start_point = {i, j};
            } else if (Map[i][j].type == 4) {
                end_point = {i, j};
            }
        }
    }
    // 以上为预处理部分
    // ------------------------------------------------------------------

    Search_Cell* search_cell = new Search_Cell;
    search_cell->g = 0;
    search_cell->h = Heuristic_Funtion(search_cell, end_point);
    search_cell->x = start_point.first;
    search_cell->y = start_point.second;
    search_cell->q = T;  // 剩余步数，初始为最大步数
    search_cell->parent = nullptr;

    priority_queue<Search_Cell*, vector<Search_Cell*>, CompareF> open_list;  // 正在探索的节点
    vector<Search_Cell*> close_list;                                         // 已经探索过的节点
    open_list.push(search_cell);                                             // 将起点加入 open_list

    while (!open_list.empty()) {
        // DONE: A* 搜索过程实现
        Search_Cell* current_cell = open_list.top();
        open_list.pop();
        close_list.push_back(current_cell);
        if (current_cell->x == end_point.first && current_cell->y == end_point.second) {
            break;  // 到达终点
        }
        const pair<int, int> directions[4] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int i = 0; i < 4; i++) {
            int new_x = current_cell->x + directions[i].first;
            int new_y = current_cell->y + directions[i].second;
            if (new_x < 0 || new_x >= M || new_y < 0 || new_y >= N) {
                continue;  // 越界
            }
            if (Map[new_x][new_y].type == 1) {
                continue;  // 不可通行
            }
            Search_Cell* new_cell = new Search_Cell;
            new_cell->x = new_x;
            new_cell->y = new_y;
            new_cell->q = current_cell->q - 1;
            if (Map[new_x][new_y].type == 2 || Map[new_x][new_y].type == 4) {
                new_cell->q = T;  // 补给点或终点，剩余步数恢复至最大步数
            }
            if (new_cell->q <= 0) {
                continue;  // 剩余步数不足
            }
            new_cell->g = current_cell->g + 1;
            new_cell->h = Heuristic_Funtion(new_cell, end_point);
            new_cell->parent = current_cell;
            bool flag = false;  // 是否在 close_list 中
            for (int j = 0; j < close_list.size(); j++) {
                if (new_cell->x == close_list[j]->x && new_cell->y == close_list[j]->y && new_cell->q <= close_list[j]->q && new_cell->g >= close_list[j]->g) {
                    flag = true; // 已经探索过相同或更优的节点
                    break;
                }
            }
            if (flag) {  // 已经探索过
                continue;
            }
            open_list.push(new_cell);
        }
    }

    // ------------------------------------------------------------------
    // DONE: 填充 step_nums 与 way

    const Search_Cell* last_cell = close_list[close_list.size() - 1];
    if (close_list.size() == 0 || last_cell->x != end_point.first || last_cell->y != end_point.second) {
        step_nums = -1;
        way = "";
    } else {
        Search_Cell* temp = close_list[close_list.size() - 1];
        step_nums = 0;
        while (temp->parent != nullptr) {
            step_nums++;
            if (temp->x - temp->parent->x == 1) {
                way = "D" + way;
            } else if (temp->x - temp->parent->x == -1) {
                way = "U" + way;
            } else if (temp->y - temp->parent->y == 1) {
                way = "R" + way;
            } else if (temp->y - temp->parent->y == -1) {
                way = "L" + way;
            }
            temp = temp->parent;
        }
    }

    // ------------------------------------------------------------------
    // 释放动态内存
    for (int i = 0; i < M; i++) {
        delete[] Map[i];
    }
    delete[] Map;
    while (!open_list.empty()) {
        auto temp = open_list.top();
        delete[] temp;
        open_list.pop();
    }
    for (int i = 0; i < close_list.size(); i++) {
        delete[] close_list[i];
    }

    return;
}

void output(const string output_file, int& step_nums, string& way) {
    ofstream file(output_file);
    if (file.is_open()) {
        file << step_nums << endl;
        if (step_nums >= 0) {
            file << way << endl;
        }
        file.close();
    } else {
        cerr << "Can not open file: " << output_file << endl;
    }
    return;
}

int main(int argc, char* argv[]) {
    // 所有测试用例
    string input_base = "../input/input_";
    string output_base = "../output/output_";
    for (int i = 1; i < 11; i++) {
        int step_nums = -1;
        string way = "";
        cout << "Processing input_" << i << ".txt" << endl;
        Astar_search(input_base + to_string(i) + ".txt", step_nums, way);
        output(output_base + to_string(i) + ".txt", step_nums, way);
        cout << "Done!" << endl;
    }

    // 单个测试用例
    // int step_nums = -1;
    // string way = "";
    // Astar_search("../input/input_7.txt", step_nums, way);
    // output("../output/output_7.txt", step_nums, way);
    return 0;
}