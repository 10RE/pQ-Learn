#include <iostream>
#include "math.h"
#include <map>
#include <float.h>

using namespace std;

#define SIZE_X 2
#define SIZE_Y 2
#define ACTION_SIZE 2
#define MAX_VEHICLE_NUM 6

#define BLOCK_SIZE 1024

#define ALPHA 0.5
#define BETA 0.1 //decay reward from surroundings
#define GAMMA 0.5
#define EPSILON 0.4
#define VEHICLE_RATE 0.3

#define VERBOSE 0

#define NUM_TRAIN 10000

#define DEBUG 0

#define USE_CUDA 0

class QTable {
public:
    double* q_table;
    double* cuda_q_table;
    int table_size;
    int state_size;
    int action_size;

public:
    QTable() {
        table_size = 0;
        action_size = 0;
        q_table = NULL;
    }

    QTable(int num_states, int num_actions){
        this -> state_size = num_states;
        this -> action_size = num_actions;
        this -> table_size = num_states * num_actions;
        this -> q_table = new double[this -> table_size];
        for (int i = 0; i < this -> table_size; i++) {
            this -> q_table[i] = 0;
        }
#if USE_CUDA
        if(cudaMalloc(&(this -> cuda_q_table), sizeof(double) * (this -> table_size)) != cudaSuccess){
            cout << "Could not allocate on GPU" << endl;
        }
        this -> attach();
#endif
    }

    double get_value(int state, int action){
        if (state < 0 || state >= this -> state_size || action < 0 || action >= this -> action_size){
            return -1;
        }
        return this -> q_table[(state * this -> action_size + action)];
    }

    void set_value(int state, int action, double value){
        this -> q_table[(state * this -> action_size + action)] = value;
    }

    void print_table(){
        for (int i = 0; i < this -> table_size; i += this -> action_size){
            for (int j = 0; j < this -> action_size; j++){
                cout << this -> q_table[i + j] << " ";
            }
            cout << endl;
        }
    }

    void attach(){
        if(cudaMemcpy(
                this -> cuda_q_table, 
                this -> q_table, 
                sizeof(double) * this -> table_size,
                cudaMemcpyHostToDevice
            ) != cudaSuccess){
            cout << "Could not copy to GPU" << endl;
        }
    }

    double* get_table_cuda(){
        return this -> cuda_q_table;
    }

    void detach(){
        if(cudaMemcpy(
                this -> q_table, 
                this -> cuda_q_table, 
                sizeof(double) * this -> table_size,
                cudaMemcpyDeviceToHost
            ) != cudaSuccess){
            cout << "Could not copy to CPU" << endl;
        }
    }

    void print_table_max() {
        int col = 0;
        for (int i = 0; i < this -> table_size; i += this -> action_size){
            double max = -1;
            int max_action = -1;
            for (int j = 0; j < this -> action_size; j++){
                if (this -> q_table[i + j] > max){
                    max = this -> q_table[i + j];
                    max_action = j;
                }
            }
            cout << "#" << max_action << "," << max << " ";
            if (col % MAX_VEHICLE_NUM == MAX_VEHICLE_NUM - 1){
                cout << endl;
            }
            col ++;
        }
    }

};

class Intersection {
public:
    QTable q_table;

    /*
        Internal parameters
    */
    int* cur_state; // 0 - 8 number of cars in each direction top, top-turn, right, right-turn, bottom, bottom-turn, left, left-turn, 9 yellow light
    int cur_state_int;
    int cur_action;
    int* next_state;
    int next_state_int;
    //int next_state[9];

    int* out_vehicle;
    int* in_vehicle;

    /*
        specific parameters to certain problem
    */
    int max_vehicle_num;
    double* lane_property_prob;
    int lane_size;
    int state_size;
    int action_size;

    //int prev_distance;

public:

    void init_array(int* array, int size, int value){
        for (int i = 0; i < size; i++){
            array[i] = value;
        }
    }

    Intersection() {
        this -> action_size = 2;
        this -> lane_size = 4;
        this -> max_vehicle_num = MAX_VEHICLE_NUM;
        this -> state_size = ((int)pow(this -> max_vehicle_num, this -> lane_size));
        this -> q_table = QTable(this -> state_size, this -> action_size);

        this -> cur_state = new int[this -> lane_size];
        init_array(this -> cur_state, this -> lane_size, 0);

        this -> next_state = new int[this -> lane_size];
        init_array(this -> next_state, this -> lane_size, 0);

        this -> out_vehicle = new int[this -> lane_size];
        init_array(this -> out_vehicle, this -> lane_size, 0);

        this -> in_vehicle = new int[this -> lane_size];
        init_array(this -> in_vehicle, this -> lane_size, 0);

        this -> lane_property_prob = new double[this -> lane_size * 2];
        for (int i = 0; i < this -> lane_size * 2; i++){
            this -> lane_property_prob[i] = 0.5;
        }

    }

    // Intersection(double* lane_property_prob) {
    //     this -> action_size = 2;
    //     this -> lane_size = 4;
    //     this -> max_vehicle_num = MAX_VEHICLE_NUM;
    //     this -> state_size = ((int)pow(this -> max_vehicle_num, this -> lane_size));
    //     this -> q_table = QTable(this -> state_size, this -> action_size);

    //     this -> cur_state = new int[this -> lane_size];
    //     init_array(this -> cur_state, this -> lane_size, 0);

    //     this -> next_state = new int[this -> lane_size];
    //     init_array(this -> next_state, this -> lane_size, 0);

    //     this -> out_vehicle = new int[this -> lane_size];
    //     init_array(this -> out_vehicle, this -> lane_size, 0);

    //     this -> in_vehicle = new int[this -> lane_size];
    //     init_array(this -> in_vehicle, this -> lane_size, 0);


    //     this -> lane_property_prob = new double[this -> lane_size * 2];
    //     for (int i = 0; i < this -> lane_size * 2; i++){
    //         this -> lane_property_prob[i] = lane_property_prob[i];
    //     }
    // }

    void reset() {
        init_array(this -> cur_state, this -> lane_size, 0);
        this -> cur_state_int = 0;
        init_array(this -> next_state, this -> lane_size, 0);
        this -> next_state_int = 0;
    }

    bool is_end_state() {
        for (int i = 0; i < this -> lane_size; i++){
            // #if DEBUG
            //     cout << "is_end_state: " << this -> cur_state[i] << " ";
            // #endif
            if (this -> cur_state[i] > this -> max_vehicle_num){
                return true;
            }
        }
        return false;
    }

    int get_state(int* state) {
        int state_ret = 0;
        for (int i = 0; i < this -> lane_size; i++){
            state_ret += state[i] * (int)pow(this -> max_vehicle_num, i);
        }
        //state += this -> cur_state[this -> lane_size] * (int)pow(this -> max_vehicle_num, this -> lane_size);
        return state_ret;
    }
    
    void choose_max_action() {
        double max_action_val = 0;
        int state_offset = this -> cur_state_int;
        this -> cur_action = 0;
        for (int i = 0; i < this -> action_size; i++) {
            int val = this -> q_table.get_value(state_offset + i, i);
            if (val > max_action_val) {
                max_action_val = val;
                this -> cur_action = i;
            }
        }
        #if DEBUG
            cout << "choose_max_action: " << this -> cur_action << endl;
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

    void update_out_vehicle() {
        this -> out_vehicle[0] = this -> next_state[1] - this -> cur_state[1];
        this -> out_vehicle[1] = this -> next_state[0] - this -> cur_state[0];
        this -> out_vehicle[2] = this -> next_state[3] - this -> cur_state[3];
        this -> out_vehicle[3] = this -> next_state[2] - this -> cur_state[2];
        // double rand_num = ((double)rand()) / RAND_MAX;
        // if (rand_num < this -> lane_property_prob[0]){
        //     this -> out_vehicle[0] = this -> next_state[1] - this -> cur_state[1];
        // }
        // rand_num = ((double)rand()) / RAND_MAX;
        // if (rand_num < this -> lane_property_prob[1]){
        //     this -> out_vehicle[1] = this -> next_state[0] - this -> cur_state[0];
        // }
        // rand_num = ((double)rand()) / RAND_MAX;
        // if (rand_num < this -> lane_property_prob[2]){
        //     this -> out_vehicle[2] = this -> next_state[3] - this -> cur_state[3];
        // }
        // rand_num = ((double)rand()) / RAND_MAX;
        // if (rand_num < this -> lane_property_prob[3]){
        //     this -> out_vehicle[3] = this -> next_state[2] - this -> cur_state[2];
        // }
    }

    int get_out_vehicle(int direction) {
        return this -> out_vehicle[direction];
    }

    void reset_in_vehicle() {
        for (int i = 0; i < this -> lane_size; i++){
            this -> in_vehicle[i] = 0;
        }
    }

    void increase_in_vehicle(int direction) {
        this -> in_vehicle[direction] += 1;
    }

    void update_with_in_vehicle() {
        for (int i = 0; i < this -> lane_size; i++){
            this -> next_state[i] += this -> in_vehicle[i];
        }
    }

    bool state_valid(int state) {
        if (state < 0 || state >= this -> state_size) {
            return false;
        }
        return true;
    }

    // bool action_valid(int state, int action) {
    //     if (state < this -> size_x && action == 0) {
    //         return false;
    //     }
    //     if (state >= this -> size_x * (this -> size_y - 1) && action == 1) {
    //         return false;
    //     }
    //     if (state % this -> size_x == 0 && action == 2) {
    //         return false;
    //     }
    //     if (state % this -> size_x == this -> size_x - 1 && action == 3) {
    //         return false;
    //     }
    //     return true;
    // }

    void get_next_state() {
        this -> get_next_state(this -> cur_action);
        #if DEBUG
            cout << "get_next_state: cur_state: ";
            for (int i = 0; i < this -> lane_size; i++){
                cout << this -> cur_state[i] << " ";
            }
            cout << "next_state: ";
            for (int i = 0; i < this -> lane_size; i++){
                cout << this -> next_state[i] << " ";
            }
            cout << endl;
        #endif
    }

    void get_next_state(int action) {
        int cur_intersection_state = this -> cur_state[8];
        int lane = action * 2;
        for (int i = 0; i < this -> lane_size; i++){
            if (i == lane || i == (lane + 1)) {
                this -> next_state[i] = this -> cur_state[i] > 0 ? this -> cur_state[i] - 1 : this -> cur_state[i];
            }
            else {
                this -> next_state[i] = this -> cur_state[i];
            }            
        }
        this -> next_state_int = this -> get_state(this -> next_state);
    }

    int cal_vehicle_num(int* state) {
        int vehicle_num = 0;
        for (int i = 0; i < this -> lane_size; i++){
            vehicle_num += state[i];
        }
        return vehicle_num;
    }

    double get_reward() {
        double next_reward = 0;
        int larger_val_h = this -> next_state[2] > this -> next_state[3] ? this -> next_state[2] : this -> next_state[3];
        int larger_val_v = this -> next_state[0] > this -> next_state[1] ? this -> next_state[0] : this -> next_state[1];
        if (this -> cur_action == 0) {
            next_reward += larger_val_v * 0.3;
            next_reward -= larger_val_h * larger_val_h / this -> max_vehicle_num;
        }
        else {
            next_reward += larger_val_h * 0.3;
            next_reward -= larger_val_v * larger_val_v / this -> max_vehicle_num;
        } 
        int next_vehicle_num = this -> cal_vehicle_num(this -> next_state);
        int cur_vehicle_num = this -> cal_vehicle_num(this -> cur_state);
        //reward = cur_vehicle_num - next_vehicle_num;
        return next_reward;
    }

    double get_cur_q_value() {
        return this -> q_table.get_value(this -> cur_state_int, this -> cur_action);
        #if DEBUG
            cout << "get cur q value - q value: " << this -> q_table.get_value(this -> cur_state_int, this -> cur_action) << endl;
        #endif
    }

    double get_next_q_value() {
        double max_action_val = 0;
        for (int i = 0; i < this -> action_size; i++) {
            if (this -> q_table.get_value(this -> next_state_int, i) > max_action_val) {
                max_action_val = this -> q_table.get_value(this -> next_state_int, i);
            }
        }
        #if DEBUG
            cout << "get next q value - q value: " << max_action_val << endl;
        #endif
        return max_action_val;
    }

    void set_q_value(double q_value) {
        this -> q_table.set_value(this -> cur_state_int, this -> cur_action, q_value);
    }

    void take_next_state() {
        for (int i = 0; i < this -> lane_size; i++){
            this -> cur_state[i] = this -> next_state[i];
            this -> cur_state_int = this -> next_state_int;
        }
        /*
        
        update with random 
        
        
        */
        #if DEBUG
            cout << "take next state - current state: " << this -> cur_state << endl;
        #endif
    }

    void print_q_table() {
        this -> q_table.print_table_max();
    }

    void print_status() {
        for (int i = 0; i < this -> lane_size; i++){
            cout << this -> cur_state[i] << " ";
        }
        cout << endl;
    }

    double* get_q_table_cuda() {
        return this -> q_table.get_table_cuda();
    }
};

/*
while (train_step < this->num_train) {
    reset all intersections
    while ( all intersection are not end state) {
        all intersection choose action
        all intersection get next state
        all intersection cal reward (counting numbers of vehicle that will enter the intersection)
        cal q value (every intersection update next state q with incoming vehicles)
        set q value
        take next state
        train_step ++
    }



*/

class IntersectionMesh {
public:
    int size_x;
    int size_y;
    
    int mesh_size;

    double epsilon;
    double alpha;
    double beta;

    double vehicle_rate;

    int num_train;

    Intersection* mesh;

    IntersectionMesh() {
        this -> size_x = SIZE_X;
        this -> size_y = SIZE_Y;
        this -> mesh_size = this -> size_x * this -> size_y;
        this -> mesh = new Intersection[this -> size_x * this -> size_y];

        this -> epsilon = EPSILON;
        this -> alpha = ALPHA;
        this -> beta = BETA;

        this -> vehicle_rate = VEHICLE_RATE;

        this -> num_train = NUM_TRAIN;

    }

    void reset() {
        for (int i = 0; i < this -> mesh_size; i++) {
#if DEBUG
    cout << "reset - intersection: " << i << endl;
#endif
            this -> mesh[i].reset();
        }
    }

    bool is_end_state() {
        for (int i = 0; i < this -> mesh_size; i++) {
            if (this -> mesh[i].is_end_state()) {
                #if DEBUG
                    cout << "is end state - intersection: " << i << " is end state" << endl;
                #endif
                return true;
            }
        }
        #if DEBUG
            cout << "is end state - intersection: is not end state" << endl;
        #endif
        return false;
    }

    void choose_action() {
        for (int i = 0; i < this -> mesh_size; i++) {
            double rand_num = (double)rand() / RAND_MAX;
            #if DEBUG
                cout << "choose action - intersection: " << i << " rand num: " << rand_num << endl;
            #endif
            if (rand_num < this -> epsilon) {
                this -> mesh[i].choose_random_action();
            }
            else {
                this -> mesh[i].choose_max_action();
            }
        }
    }

    void choose_max_action() {
        for (int i = 0; i < this -> mesh_size; i++) {
            this -> mesh[i].choose_max_action();
        }
    }

    void get_next_state() {
        for (int i = 0; i < this -> mesh_size; i++) {
            #if DEBUG
                cout << "get next state - intersection: " << i << endl;
            #endif
            this -> mesh[i].get_next_state();
        }
    }

    void update_out_vehicle() {
        for (int i = 0; i < this -> mesh_size; i++) {
            #if DEBUG
                cout << "update out vehicle - intersection: " << i << endl;
            #endif
            this -> mesh[i].update_out_vehicle();
        }
    }

    void sync() {
        for (int i = 0; i < this -> mesh_size; i++) {
            #if DEBUG
                cout << "sync - intersection: " << i << endl;
            #endif
            this -> mesh[i].reset_in_vehicle();
            int id = i - this -> size_x;
            if (id > 0 && mesh[id].get_out_vehicle(1)) {
                this -> mesh[i].increase_in_vehicle(0);
            }
            id = i + this -> size_x;
            if (id < this -> mesh_size && mesh[id].get_out_vehicle(0)) {
                this -> mesh[i].increase_in_vehicle(1);
            }
            id = i - 1;
            if (i % size_x != 0 && mesh[id].get_out_vehicle(3)) {
                this -> mesh[i].increase_in_vehicle(2);
            }
            id = i + 1;
            if (i % size_x != size_x - 1 && mesh[id].get_out_vehicle(2)) {
                this -> mesh[i].increase_in_vehicle(3);
            }
            this -> mesh[i].update_with_in_vehicle();
        }
    }

    double get_reward(int intersection_id) {
        #if DEBUG
            cout << "get reward - intersection: " << intersection_id << endl;
        #endif
        return this -> mesh[intersection_id].get_reward();
    }

    // double get_reward() {
    //     double reward = 0;
    //     for (int i = 0; i < this -> mesh_size; i++) {
    //         reward += this -> mesh[i].get_reward(i);
    //     }
    //     return reward;
    // }

    void update_q_value() {
        #if DEBUG
            cout << "update q value: ";
        #endif
        double* reward = new double[this -> mesh_size];
        for (int i = 0; i < this -> mesh_size; i++) {
            reward[i] = this -> mesh[i].get_reward();
            #if DEBUG
                cout << reward[i] << " ";
            #endif
        }
        #if DEBUG
            cout << endl;
        #endif
        for (int i = 0; i < this -> mesh_size; i ++) {
            double surrounding_reward = 0;

            int id = i - this -> size_x;
            if (id > 0) {
                surrounding_reward += reward[id];
            }
            id = i + this -> size_x;
            if (id < this -> mesh_size) {
                surrounding_reward += reward[id];
            }
            id = i - 1;
            if (i % size_x != 0) {
                surrounding_reward += reward[id];
            }
            id = i + 1;
            if (i % size_x != size_x - 1) {
                surrounding_reward += reward[id];
            }

            double q_value = 0;
            q_value += (1 - this->alpha) * this -> mesh[i].get_cur_q_value();
            q_value += this -> alpha * (reward[i] + this -> beta * surrounding_reward);
            q_value += this -> alpha * this -> mesh[i].get_next_q_value();
            #if DEBUG
                cout << "update q value - intersection: " << i << " surrounding:" << surrounding_reward << " q value: " << q_value << endl;
            #endif
            this -> mesh[i].set_q_value(q_value);
        }
    }

    void take_next_state() {
        for (int i = 0; i < this -> mesh_size; i++) {
            this -> mesh[i].take_next_state();
        }
    }

    void generate_new_vehicle() {
        for (int i = 0; i < this -> mesh_size; i++) {
            this -> mesh[i].reset_in_vehicle();
            if (i < this -> size_x) {
                double rand_num = (double)rand() / RAND_MAX;
                if (rand_num < this -> vehicle_rate) {
                    this -> mesh[i].increase_in_vehicle(0);
                }
            }
            if (i >= this -> mesh_size - this -> size_x) {
                double rand_num = (double)rand() / RAND_MAX;
                if (rand_num < this -> vehicle_rate) {
                    this -> mesh[i].increase_in_vehicle(1);
                }
            }
            if (i % this -> size_x == 0) {
                double rand_num = (double)rand() / RAND_MAX;
                if (rand_num < this -> vehicle_rate) {
                    this -> mesh[i].increase_in_vehicle(2);
                }
            }
            if (i % this -> size_x == this -> size_x - 1) {
                double rand_num = (double)rand() / RAND_MAX;
                if (rand_num < this -> vehicle_rate) {
                    this -> mesh[i].increase_in_vehicle(3);
                }
            }
            this -> mesh[i].update_with_in_vehicle();
        }
    }

    void print() {
        for (int i = 0; i < this -> mesh_size; i++) {
            cout << "intersection: " << i << endl;
            this -> mesh[i].print_q_table();
        }
    }

    void train() {
        int train_step = 0;
        int epoch_step = 0;
        cout << train_step << endl;
        while (train_step < this->num_train) {
            cout << "======================" << endl;
            cout << "start epoch: " << epoch_step << endl;
            this -> reset();
            while (!this -> is_end_state()) {
                if (train_step > this->num_train) {
                    break;
                }
                cout << "----------------------" << endl;
                cout << "train step: " << train_step << endl;
                #if DEBUG
                    for (int i = 0; i < this -> mesh_size; i++) {
                        cout << "train - intersection: " << i << " cur_state: " << this -> mesh[i].cur_state_int << endl;
                    }
                #endif

                #if VERBOSE
                    for (int i = 0; i < this -> mesh_size; i++) {
                        this -> mesh[i].print_status();
                    }
                #endif

                this -> choose_action();
                this -> get_next_state();
                this -> update_out_vehicle();
                this -> sync();
                this -> update_q_value();

                this -> generate_new_vehicle();
                this -> take_next_state();

                train_step ++;
            }
            epoch_step ++;
        }
    }

    void run() {
        int train_step = 0;
        this -> reset();
        while (!this -> is_end_state()) {
            if (train_step > 1000000) {
                cout << "Run succeed" << endl;
                break;
            }

            #if DEBUG
                for (int i = 0; i < this -> mesh_size; i++) {
                    cout << "train - intersection: " << i << " cur_state: " << this -> mesh[i].cur_state_int << endl;
                }
            #endif

            for (int i = 0; i < this -> mesh_size; i++) {
                cout << "intersection: " << i << endl;
                this -> mesh[i].print_status();
            }

            this -> choose_max_action();
            this -> get_next_state();
            this -> update_out_vehicle();
            this -> sync();
            this -> update_q_value();

            this -> generate_new_vehicle();
            this -> take_next_state();

            train_step ++;
        }
        cout << "Steps: " << train_step << endl;
    }
};

int main() {
    IntersectionMesh mesh = IntersectionMesh();
    mesh.train();
    //mesh.print();
    mesh.run();
    return 0;
}