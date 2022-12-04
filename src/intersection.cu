#include <iostream>
#include "math.h"
#include <map>
#include <float.h>

using namespace std;

#define TARGET_STATE 55
#define SIZE_X 10
#define SIZE_Y 10
#define ACTION_SIZE 4
#define STATE_SIZE 100
#define MAX_VEHICLE_NUM 10

#define BLOCK_SIZE 1024

#define ALPHA 0.1
#define GAMMA 0.9
#define EPSILON 0.8

#define NUM_TRAIN 1000000

#define DEBUG 0

#define USE_CUDA 1

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
            if (col % SIZE_X == SIZE_X - 1){
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
    Intersection() {
        this -> action_size = 4;
        this -> lane_size = 8;
        this -> max_vehicle_num = MAX_VEHICLE_NUM;
        this -> state_size = (((int)pow(this -> max_vehicle_num, 8)) * 4);
        this -> q_table = QTable(this -> state_size, this -> action_size);

        this -> cur_state = new int[this -> lane_size];
        this -> cur_state = {0};

        this -> next_state = new int[this -> lane_size];
        this -> next_state = {0};

        this -> out_vehicle = new int[4];
        this -> out_vehicle = {0};

        this -> in_vehicle = new int[4];
        this -> in_vehicle = {0};

        this -> lane_property_prob = new double[this -> lane_size * 2];
        for (int i = 0; i < this -> lane_size * 2; i++){
            this -> lane_property_prob[i] = 0.5;
        }

    }

    Intersection(double* lane_property_prob) {
        this -> action_size = 4;
        this -> lane_size = 8;
        this -> max_vehicle_num = MAX_VEHICLE_NUM;
        this -> state_size = (((int)pow(this -> max_vehicle_num, 8)) * 4);
        this -> q_table = QTable(this -> state_size, this -> action_size);

        this -> cur_state = new int[this -> lane_size + 1];
        this -> cur_state = {0};

        this -> next_state = new int[this -> lane_size];
        this -> next_state = {0};

        this -> lane_property_prob = new double[this -> lane_size * 2];
        for (int i = 0; i < this -> lane_size * 2; i++){
            this -> lane_property_prob[i] = lane_property_prob[i];
        }
    }

    void reset() {
        this -> cur_state = {0};
        this -> next_state = {0};
    }

    bool is_end_state() {
        for (int i = 0; i < this -> lane_size; i++){
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
    
    // void get_state() {
    //     return this -> cur_state;
    // }
    
    void choose_max_action() {
        double max_action_val = 0;
        int state_offset = this -> cur_state_int;
        for (int i = 0; i < this -> action_size; i++) {
            int val = this -> q_table.get_value(state_offset + i, i);
            if (val > max_action_val) {
                max_action_val = val;
                this -> cur_action = i;
            }
        }
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
        this -> out_vehicle = {0};
        this -> out_vehicle[0] = this -> next_state[1] - this -> cur_state[1] + this -> next_state[6] - this -> cur_state[6];
        this -> out_vehicle[1] = this -> next_state[0] - this -> cur_state[0] + this -> next_state[7] - this -> cur_state[7];
        this -> out_vehicle[2] = this -> next_state[3] - this -> cur_state[3] + this -> next_state[5] - this -> cur_state[5];
        this -> out_vehicle[3] = this -> next_state[2] - this -> cur_state[2] + this -> next_state[4] - this -> cur_state[4];
    }

    int get_out_vehicle(int direction) {
        return this -> out_vehicle[direction];
    }

    void reset_in_vehicle() {
        this -> in_vehicle = {0};
    }

    void increase_in_vehicle(int direction) {
        this -> in_vehicle[direction] += 1;
    }

    void update_with_in_vehicle() {
        
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
        int cur_intersection_state = this -> cur_state[8];
        int lane = this -> cur_action * 2;
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

    // void get_next_state() {
    //     /*
    //         0 - down
    //         1 - up
    //         2 - left
    //         3 - right
    //     */
    //     if (action_valid(this -> cur_state, this -> cur_action) == false) {
    //         this -> next_state = this -> cur_state;
    //         return;
    //     }
    //     switch (this -> cur_action) {
    //         case 0:
    //             this -> next_state = this -> cur_state - this -> size_x;
    //             break;
    //         case 1:
    //             this -> next_state = this -> cur_state + this -> size_x;
    //             break;
    //         case 2:
    //             this -> next_state = this -> cur_state - 1;
    //             break;
    //         case 3:
    //             this -> next_state = this -> cur_state + 1;
    //             break;
    //     }
    //     #if DEBUG
    //         cout << "get next state - next state: " << this -> next_state << endl;
    //     #endif
    // }

    int cal_vehicle_num(int* state) {
        int vehicle_num = 0;
        for (int i = 0; i < this -> lane_size; i++){
            vehicle_num += state[i];
        }
        return vehicle_num;
    }

    double get_reward() {
        double reward = 0;
        int next_vehicle_num = this -> cal_vehicle_num(this -> next_state);
        int cur_vehicle_num = this -> cal_vehicle_num(this -> cur_state);
        reward = cur_vehicle_num - next_vehicle_num;
        return reward;
    }

    double get_cur_q_value() {
        return this -> q_table.get_value(this -> cur_state_int, this -> cur_action);
        #if DEBUG
            cout << "get cur q value - q value: " << this -> q_table.get_value(this -> cur_state, this -> cur_action) << endl;
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
    int epsilon;
    int mesh_size;

    Intersection* mesh;

    IntersectionMesh() {
        this -> size_x = SIZE_X;
        this -> size_y = SIZE_Y;
        this -> mesh_size = this -> size_x * this -> size_y;
        this -> mesh = new Intersection[this -> size_x * this -> size_y];

        this -> epsilon = EPSILON;
    }

    bool is_end_state() {
        for (int i = 0; i < this -> mesh_size; i++) {
            if (this -> mesh[i].is_end_state() == false) {
                return false;
            }
        }
        return true;
    }

    void choose_action() {
        for (int i = 0; i < this -> mesh_size; i++) {
            double rand_num = (double)rand() / RAND_MAX;
            if (rand_num < this -> epsilon) {
                this -> mesh[i].choose_random_action();
            }
            else {
                this -> mesh[i].choose_max_action();
            }
        }
    }

    void get_next_state() {
        for (int i = 0; i < this -> mesh_size; i++) {
            this -> mesh[i].get_next_state();
        }
    }

    void update_out_vehicle() {
        for (int i = 0; i < this -> mesh_size; i++) {
            this -> mesh[i].update_out_vehicle();
        }
    }

    void sync() {
        for (int i = 0; i < this -> mesh_size; i++) {
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
        }
    }

    double get_reward() {
        double reward = 0;
        for (int i = 0; i < this -> mesh_size; i++) {
            reward += this -> mesh[i].get_reward();
        }
        return reward;
    }

    double cal_q_value(double reward) {
        return (1 - this->alpha) * this->q_object.get_cur_q_value() + this->alpha * (reward + this->gamma * this->q_object.get_next_q_value());
    }

    void train() {
        int train_step = 0;
        int epoch_step = 0;
        cout << train_step << endl;
        while (train_step < this->num_train) {
            cout << "======================" << endl;
            cout << "start epoch: " << epoch_step << endl;
            q_object.reset();
#if USE_CUDA
            double* q_table_cuda = q_object.get_q_table_cuda();
            int state = 0;
            dim3 block_size(BLOCK_SIZE);
            dim3 grid_size(
                (int)(ACTION_SIZE / block_size.x) + 1
            );
#endif
            while (!q_object.is_end_state()) {
                if (train_step > this->num_train) {
                    break;
                }
                cout << "----------------------" << endl;
                cout << "train step: " << train_step << endl;
                //this->q_object.get_state();
#if USE_CUDA
                cal_q_kernel<<<grid_size, block_size>>>(q_table_cuda, state);
                cudaDeviceSynchronize();
                q_object.q_table.detach();
                q_object.q_table.print_table_max();
#else
                this -> choose_action();
                this -> q_object.get_next_state();
                double reward = q_object.cal_reward();
                double q_value = this->cal_q_value(reward);
                cout << "q_value: " << q_value << endl;
                this -> q_object.set_q_value(q_value);
                this -> q_object.take_next_state();
#endif 
                train_step ++;
            }
            epoch_step ++;
        }
    }
};

class NashQLP {
    public:
    double alpha;
    double epsilon;
    double gamma;
    int num_train;
    QObject q_object;

    NashQLP(double alpha, double epsilon, double gamma, int num_train) {
        this->alpha = alpha;
        this->epsilon = epsilon;
        this->gamma = gamma;
        this->num_train = num_train;
        this->q_object = QObject();
    }

    void choose_action() {
        double rand_num = (double)rand() / RAND_MAX;
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

    double cal_q_value(double reward) {
        return (1 - this->alpha) * this->q_object.get_cur_q_value() + this->alpha * (reward + this->gamma * this->q_object.get_next_q_value());
    }

    void train() {
        int train_step = 0;
        int epoch_step = 0;
        cout << train_step << endl;
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
                double reward = q_object.cal_reward();
                double q_value = this->cal_q_value(reward);
                cout << "q_value: " << q_value << endl;
                this -> q_object.set_q_value(q_value);
                this -> q_object.take_next_state();
                train_step ++;
            }
            epoch_step ++;
        }
    }
}


void train() {
    int train_step = 0;
    int epoch_step = 0;
    cout << train_step << endl;
    while (train_step < this->num_train) {
        cout << "======================" << endl;
        cout << "start epoch: " << epoch_step << endl;

        mesh.reset();

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


        while (!q_object.is_end_state()) {
            if (train_step > this->num_train) {
                break;
            }
            cout << "----------------------" << endl;
            cout << "train step: " << train_step << endl;
            //this->q_object.get_state();

            this -> choose_action();
            this -> q_object.get_next_state();
            double reward = q_object.cal_reward();
            double q_value = this->cal_q_value(reward);
            cout << "q_value: " << q_value << endl;
            this -> q_object.set_q_value(q_value);
            this -> q_object.take_next_state();
#endif 
            train_step ++;
        }
        epoch_step ++;
    }
}

int main() {
    QLP qlp = QLP(ALPHA, EPSILON, GAMMA, NUM_TRAIN);
    cout << "training..." << endl;

    QObject q_object = QObject();
    double* q_table_cuda = q_object.get_q_table_cuda();

    int train_step = 0;
    int epoch_step = 0;
    cout << train_step << endl;
    while (train_step < NUM_TRAIN) {
#if 0
        cout << "======================" << endl;
        cout << "start epoch: " << epoch_step << endl;
#endif
        int state = 0;
        dim3 block_size(BLOCK_SIZE);
        dim3 grid_size(
            (int)(ACTION_SIZE / block_size.x) + 1
        );
        while (!is_end_state(TARGET_STATE, state)) {
            if (train_step % 10000 == 0) {
                cout << "train step: " << train_step << " coverage:" << get_coverage(q_table_cuda) << endl;

            }
            if (train_step > NUM_TRAIN) {
                break;
            }
#if 0
                cout << "----------------------" << endl;
                cout << "train step: " << train_step << endl;
                cout << "state: " << state << endl;
#endif
            cal_q_kernel<<<grid_size, block_size>>>(q_table_cuda, state);
            cudaDeviceSynchronize();

            double rand_num = (double)rand() / RAND_MAX;
            if (rand_num > EPSILON) {
                state = get_next_state_by_max(q_table_cuda, state);
#if DEBUG
                cout << "choose max action, next state: " << state << endl;
#endif
            }
            else {
                state = get_next_state_by_rand(state);
#if DEBUG
                cout << "choose rand action, next state: " << state << endl;
#endif
            }

#if DEBUG
            cout << "coverage: " << get_coverage(q_table_cuda) << endl;
            q_object.q_table.detach();
            q_object.q_table.print_table_max();
#endif
            train_step ++;
        }
#if 0
        cout << "coverage: " << get_coverage(q_table_cuda) << endl;
#endif
        epoch_step ++;
    }
    cout << "coverage: " << get_coverage(q_table_cuda) << endl;
    q_object.q_table.detach();
    q_object.q_table.print_table_max();
    //qlp.q_object.print_q_table();
    return 0;
}