#include <iostream>
#include "math.h"
#include <map>

using namespace std;

#define TARGET_STATE 9
#define SIZE_X 5
#define SIZE_Y 5
#define ACTION_SIZE 4

#define DEBUG 0

class QTable {
private:
    float** q_table;
    int table_size;
    int action_size;

public:
    QTable() {
        table_size = 0;
        action_size = 0;
        q_table = NULL;
    }

    QTable(int num_states, int num_actions){
        this -> table_size = num_states;
        this -> action_size = num_actions;
        this -> q_table = new float*[this -> table_size];
        for (int i = 0; i < this -> table_size; i++){
            this -> q_table[i] = new float[this -> action_size];
        }
    }

    float get_value(int state, int action){
        if (state < 0 || state >= this -> table_size || action < 0 || action >= this -> action_size){
            return -1;
        }
        return this -> q_table[state][action];
    }

    void set_value(int state, int action, float value){
        this -> q_table[state][action] = value;
    }

    void print_table(){
        for (int i = 0; i < this -> table_size; i++){
            for (int j = 0; j < this -> action_size; j++){
                cout << this -> q_table[i][j] << " ";
            }
            cout << endl;
        }
    }

    void print_table_max() {
        for (int i = 0; i < this -> table_size; i++){
            float max = -1;
            int max_action = -1;
            for (int j = 0; j < this -> action_size; j++){
                if (this -> q_table[i][j] > max){
                    max = this -> q_table[i][j];
                    max_action = j;
                }
            }
            cout << "#" << max_action << "," << max << " ";
            if (i % SIZE_X == SIZE_X - 1){
                cout << endl;
            }
        }
    }

};

class QObject {
private:
    QTable q_table;

    /*
        Internal parameters
    */
    int cur_state;
    int cur_action;
    int next_state;

    /*
        specific parameters to certain problem
    */
    int size_x;
    int size_y;
    int target_state;
    int table_size;
    int action_size;

    int prev_distance;


public:
    QObject() {
        this -> target_state = TARGET_STATE;
        this -> size_x = SIZE_X;
        this -> size_y = SIZE_Y;
        this -> table_size = this -> size_x * this -> size_y;
        this -> action_size = ACTION_SIZE;
        this -> q_table = QTable(this -> table_size, this -> action_size);
        this -> cur_state = 0;
        this -> cur_action = 3;
        this -> next_state = 0;
        this -> prev_distance = this -> cal_distance(0) + 1;

    }

    QObject(int num_states, int num_actions, int target_state) : q_table(num_states, num_actions){
        this -> target_state = target_state;
        this -> table_size = num_states;
        this -> action_size = num_actions;
        this -> cur_state = 0;
        this -> cur_action = 0;
        this -> next_state = 0;
        this -> prev_distance = this -> cal_distance(0) + 1;
    }

    void reset() {
        this -> cur_state = 0;
        this -> cur_action = 3;
        this -> next_state = 0;
        this -> prev_distance = this -> cal_distance(0) + 1;
    }

    bool is_end_state() {
        #if DEBUG
            cout << "is end state - current state: " << this -> cur_state << " target state: " << this -> target_state << endl;
        #endif
        if (this -> target_state == this -> cur_state || state_valid(this -> cur_state) == false) {
            return true;
        }
        return false;
    }
    
    // void get_state() {
    //     return this -> cur_state;
    // }
    
    void choose_max_action() {
        float max_action_val = 0;
        for (int i = 0; i < this -> action_size; i++) {
            if (this -> q_table.get_value(this -> cur_state, i) > max_action_val) {
                max_action_val = this -> q_table.get_value(this -> cur_state, i);
                this -> cur_action = i;
            }
        }

        #if DEBUG
            cout << "choose max action - max action: " << this -> cur_action << endl;
        #endif

    }

    bool state_valid(int state) {
        if (state < 0 || state >= this -> table_size) {
            return false;
        }
        return true;
    }

    bool action_valid(int state, int action) {
        if (state < this -> size_x && action == 0) {
            return false;
        }
        if (state >= this -> size_x * (this -> size_y - 1) && action == 1) {
            return false;
        }
        if (state % this -> size_x == 0 && action == 2) {
            return false;
        }
        if (state % this -> size_x == this -> size_x - 1 && action == 3) {
            return false;
        }
        return true;
    }

    void choose_max_action(int state) {
        float max_action_val = 0;
        for (int i = 0; i < this -> action_size; i++) {
            if (this -> q_table.get_value(state, i) > max_action_val) {
                this -> cur_action = i;
                max_action_val = this -> q_table.get_value(this -> cur_state, i);
            }
        }
        #if DEBUG
            cout << "choose max action - max action: " << this -> cur_action << endl;
        #endif
    }

    void choose_random_action() {
        this -> cur_action = rand() % this -> action_size;
        // while (action_valid(this -> cur_state, this -> cur_action) == false) {
        //     this -> cur_action = rand() % this -> action_size;
        // }
        #if DEBUG
            cout << "choose random action - random action: " << this -> cur_action << endl;
        #endif
    }
    
    void get_next_state() {
        /*
            0 - down
            1 - up
            2 - left
            3 - right
        */
        if (action_valid(this -> cur_state, this -> cur_action) == false) {
            this -> next_state = this -> cur_state;
            return;
        }
        switch (this -> cur_action) {
            case 0:
                this -> next_state = this -> cur_state - this -> size_x;
                break;
            case 1:
                this -> next_state = this -> cur_state + this -> size_x;
                break;
            case 2:
                this -> next_state = this -> cur_state - 1;
                break;
            case 3:
                this -> next_state = this -> cur_state + 1;
                break;
        }
        #if DEBUG
            cout << "get next state - next state: " << this -> next_state << endl;
        #endif
    }

    int cal_distance(int state) {
        int diff = this -> target_state - state;
        int distance = 0;
        // if ((state + diff) > 0 && (state + diff) < this -> size_x) {
        //     distance = diff;
        // }
        // else {
            int target_x = this -> target_state % this -> size_x;
            int target_y = this -> target_state / this -> size_x;
            int cur_x = state % this -> size_x;
            int cur_y = state / this -> size_x;
            distance = abs(target_x - cur_x) + abs(target_y - cur_y);
            //distance = (diff - this -> size_x) % this -> size_x + (this -> size_x - diff) / this -> size_x;
        // }
        #if DEBUG
            cout << "cal distance - diff: " << diff << " distance: " << distance << endl;
        #endif
        return distance;
    }

    float cal_reward() {
        float next_distance = (float)cal_distance(this -> next_state);
        float cur_distance = (float)cal_distance(this -> cur_state);
        float reward = cur_distance - next_distance;
        // if (distance < this -> prev_distance) {
        //     this -> prev_distance = distance;
        // }
        return reward;
    };

    float get_cur_q_value() {
        return this -> q_table.get_value(this -> cur_state, this -> cur_action);
        #if DEBUG
            cout << "get cur q value - q value: " << this -> q_table.get_value(this -> cur_state, this -> cur_action) << endl;
        #endif
    }

    float get_next_q_value() {
        float max_action_val = 0;
        for (int i = 0; i < this -> action_size; i++) {
            if (this -> q_table.get_value(this -> next_state, i) > max_action_val) {
                max_action_val = this -> q_table.get_value(this -> next_state, i);
            }
        }
        #if DEBUG
            cout << "get next q value - q value: " << max_action_val << endl;
        #endif
        return max_action_val;
    }

    void set_q_value(float q_value) {
        this -> q_table.set_value(this -> cur_state, this -> cur_action, q_value);
    }

    void take_next_state() {
        this -> cur_state = this -> next_state;
        #if DEBUG
            cout << "take next state - current state: " << this -> cur_state << endl;
        #endif
    }

    void print_q_table() {
        this -> q_table.print_table_max();
    }
};

class QLP {
public:
    float alpha;
    float epsilon;
    float gamma;
    int num_train;
    QObject q_object;

    QLP(float alpha, float epsilon, float gamma, int num_train) {
        this->alpha = alpha;
        this->epsilon = epsilon;
        this->gamma = gamma;
        this->num_train = num_train;
        this->q_object = QObject();
    }

    void choose_action() {
        float rand_num = (float)rand() / RAND_MAX;
        if (rand_num > this->epsilon) {
            this->q_object.choose_max_action();
        }
        else {
            this->q_object.choose_random_action();
        }
        #if DEBUG
            cout << "choose action - rand: " << rand_num << " epsilon: " << this -> epsilon << endl;
        #endif
    }

    float cal_q_value(float reward) {
        return (1 - this->alpha) * this->q_object.get_cur_q_value() + this->alpha * (reward + this->gamma * this->q_object.get_next_q_value());
    }

    void train() {
        int train_step = 0;
        int epoch_step = 0;
        while (train_step < this->num_train) {
            cout << "======================" << endl;
            cout << "start epoch: " << epoch_step << endl;
            q_object.reset();
            while (!q_object.is_end_state()) {
                if (train_step > this->num_train) {
                    break;
                }
                cout << "----------------------" << endl;
                cout << "train step: " << train_step << endl;
                //this->q_object.get_state();
                this -> choose_action();
                this -> q_object.get_next_state();
                float reward = q_object.cal_reward();
                float q_value = this->cal_q_value(reward);
                cout << "q_value: " << q_value << endl;
                this -> q_object.set_q_value(q_value);
                this -> q_object.take_next_state();
                train_step ++;
            }
            epoch_step ++;
        }
    }

};

int main() {
    QLP qlp = QLP(0.1, 0.6, 0.9, 1000);
    qlp.train();
    qlp.q_object.print_q_table();
    return 0;
}