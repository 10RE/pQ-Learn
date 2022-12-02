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

#define BLOCK_SIZE 1024

#define ALPHA 0.1
#define GAMMA 0.9
#define EPSILON 0.8

#define NUM_TRAIN 1000000

#define DEBUG 0

#define USE_CUDA 1

__device__ bool action_valid_cuda(int state, int action) {
    if (state < SIZE_X && action == 0) {
        return false;
    }
    if (state >= SIZE_X * (SIZE_Y - 1) && action == 1) {
        return false;
    }
    if (state % SIZE_X == 0 && action == 2) {
        return false;
    }
    if (state % SIZE_X == SIZE_X - 1 && action == 3) {
        return false;
    }
    return true;
}

__device__ int get_next_state_cuda(int state, int action){
    if (action_valid_cuda(state, action) == false) {
        return state;
    }
    switch (action) {
        case 0:
            return state - SIZE_X;
            break;
        case 1:
            return state + SIZE_X;
            break;
        case 2:
            return state - 1;
            break;
        case 3:
            return state + 1;
            break;
    }
}

__device__ double get_value_cuda(double* q_table, int state, int action){
    return q_table[state * ACTION_SIZE + action];
}

__device__ void set_value_cuda(double* q_table, int main_offset, int offset, double value){
    q_table[main_offset + offset] = value;
}

__global__ void get_max_q_value_in_action(double* max_q, double* isMAX, double* q_table, int state) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    isMAX[row] = true;
    if (get_value_cuda(q_table, state, row) < get_value_cuda(q_table, state, col)) {
        isMAX[row] = false;
    }
    if (isMAX[row]) {
        *max_q = get_value_cuda(q_table, state, row);
    }
}

__device__ double get_max_q_value_in_action_cuda(double* q_table, int state) {
    double max_q = DBL_MIN;
    for (int i = 0; i < ACTION_SIZE; i++) {
        double value = get_value_cuda(q_table, state, i);
        
        if (value > max_q) {
            max_q = value;
        }
    }
    //printf("max_value: %f\n", max_q);
    return max_q;
}

__device__ int cal_distance_cuda(int state, int target_state) {
    // int diff = target_state - state;
    int distance = 0;
    int target_x = target_state % SIZE_X;
    int target_y = target_state / SIZE_X;
    int cur_x = state % SIZE_X;
    int cur_y = state / SIZE_X;
    distance = abs(target_x - cur_x) + abs(target_y - cur_y);
    return distance;
}

__device__ double cal_reward_cuda(int state, int next_state, int target_state) {
    double next_distance = (double)cal_distance_cuda(next_state, target_state);
    double cur_distance = (double)cal_distance_cuda(state, target_state);
    double reward = cur_distance - next_distance;
    return reward;
}

__global__ void cal_q_kernel(double* q_table, int state) {
    int action = threadIdx.x + blockIdx.x * blockDim.x;
    if (action >= ACTION_SIZE) {
        return;
    }
    // double* isMAX;
    // if (cudaMalloc(&isMAX, sizeof(double) * ACTION_SIZE) != cudaSuccess) {
        
    // }
    double cur_q = get_value_cuda(q_table, state, action);
    double max_q = 0;
    int next_state = get_next_state_cuda(state, action);

    max_q = get_max_q_value_in_action_cuda(q_table, next_state);

    //get_max_q_value_in_action<<<ACTION_SIZE, ACTION_SIZE>>>(&max_q, isMAX, q_table, next_state);
    //cudaFree(isMAX);
    double reward = cal_reward_cuda(state, next_state, TARGET_STATE);
#if DEBUG
    printf("action: %d, next_state: %d, reward: %f, max_q: %f\n", action, next_state, reward, max_q);
#endif
    q_table[state * ACTION_SIZE + action] = (1 - ALPHA) * cur_q + ALPHA * (reward + GAMMA * max_q);
    //*state = next_state;
}

__global__ void get_coverage_kernel(int* coverage, double* q_table) {
    __shared__ int not_covered_count;
    if (threadIdx.x == 0) {
        not_covered_count = 0;
    }
    int state = threadIdx.x + blockIdx.x * blockDim.x;
    if (state >= STATE_SIZE) {
        return;
    }
    int main_offset = state * ACTION_SIZE;
    bool covered_flag = false;
    for (int i = 0; i < ACTION_SIZE; i++) {
        if (get_value_cuda(q_table, state, i) != 0) {
            covered_flag = true;
            break;
        }
    }
    if (!covered_flag) {
        atomicAdd(&not_covered_count, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        coverage[blockIdx.x] = not_covered_count;
    }
}

double get_coverage(double* q_table) {
    dim3 block(BLOCK_SIZE);
    int grid_size = STATE_SIZE/block.x + 1;
    dim3 grid(grid_size);
    int* cover_count = new int[grid_size];
    int* cover_count_cuda;
    if (cudaMalloc(&cover_count_cuda, sizeof(int) * grid_size) != cudaSuccess) {
        printf("cudaMalloc for coverage failed\n");
    }
    get_coverage_kernel<<<grid, block>>>(cover_count_cuda, q_table);
    cudaMemcpy(cover_count, cover_count_cuda, sizeof(int) * grid_size, cudaMemcpyDeviceToHost);
    int total = 0;
    for (int i = 0; i < grid_size; i++) {
        total += cover_count[i];
    }
    return 1 - (double)total / (double)STATE_SIZE;
}

bool action_valid(int state, int action) {
    if (state < SIZE_X && action == 0) {
        return false;
    }
    if (state >= STATE_SIZE && action == 1) {
        return false;
    }
    if (state % SIZE_X == 0 && action == 2) {
        return false;
    }
    if (state % SIZE_X == SIZE_X - 1 && action == 3) {
        return false;
    }
    return true;
}

int get_next_state(int state, int action){
    if (action_valid(state, action) == false) {
        return state;
    }
    //printf("get_next_action: %d\n", action);
    switch (action) {
        case 0:
            return state - SIZE_X;
            break;
        case 1:
            return state + SIZE_X;
            break;
        case 2:
            return state - 1;
            break;
        case 3:
            return state + 1;
            break;
    }
    return 0;
}

int get_next_state_by_max(double* q_table, int state) {
    double* actions = new double[ACTION_SIZE];
    if(cudaMemcpy(
            actions,
            &(q_table[state * ACTION_SIZE]), 
            sizeof(double) * ACTION_SIZE,
            cudaMemcpyDeviceToHost
        ) != cudaSuccess){
        cout << "Copying actions, could not copy from GPU" << endl;
    }
    double max_q = -1000000;
    int max_action = 0;
    for (int i = 0; i < ACTION_SIZE; i++) {
        if (actions[i] > max_q && action_valid(state, i)) {
            // #if DEBUG
            // cout << "action: " << i << ", value: " << actions[i] << endl;
            // #endif
            max_q = actions[i];
            max_action = i;
        }
    }
    delete[] actions;
    return get_next_state(state, max_action);;
}

int get_next_state_by_rand(int state) {
    int action = rand() % ACTION_SIZE;
    if (action_valid(state, action) == false) {
        return get_next_state_by_rand(state);
    }
    return get_next_state(state, action);
}

bool state_valid(int state) {
    if (state < 0 || state >= STATE_SIZE) {
        return false;
    }
    return true;
}

bool is_end_state(int target_state, int state) {
    if (target_state == state || state_valid(state) == false) {
        return true;
    }
    return false;
}

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

//     ~QTable(){
//         if (this -> q_table != NULL){
//             delete[] this -> q_table;
//         }
// #if USE_CUDA
//         if (this -> cuda_q_table != NULL){
//             cudaFree(this -> cuda_q_table);
//         }
// #endif
//     }

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

class QObject {
public:
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
        //cout << this -> prev_distance << endl;
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
        double max_action_val = 0;
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
        double max_action_val = 0;
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

    double cal_reward() {
        double next_distance = (double)cal_distance(this -> next_state);
        double cur_distance = (double)cal_distance(this -> cur_state);
        double reward = cur_distance - next_distance;
        // if (distance < this -> prev_distance) {
        //     this -> prev_distance = distance;
        // }
        return reward;
    };

    double get_cur_q_value() {
        return this -> q_table.get_value(this -> cur_state, this -> cur_action);
        #if DEBUG
            cout << "get cur q value - q value: " << this -> q_table.get_value(this -> cur_state, this -> cur_action) << endl;
        #endif
    }

    double get_next_q_value() {
        double max_action_val = 0;
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

    void set_q_value(double q_value) {
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

    double* get_q_table_cuda() {
        return this -> q_table.get_table_cuda();
    }
};

class QLP {
public:
    double alpha;
    double epsilon;
    double gamma;
    int num_train;
    QObject q_object;

    QLP(double alpha, double epsilon, double gamma, int num_train) {
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