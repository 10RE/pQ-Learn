#include <iostream>
#include "math.h"
#include <map>
#include <float.h>

using namespace std;

#define SIZE_X 2
#define SIZE_Y 2
#define MESH_SIZE SIZE_X*SIZE_Y
#define ACTION_SIZE 2
#define SURROUNDING_SIZE 9
#define MAX_VEHICLE_NUM 8

#define STATE_SIZE 4096 //(int)pow(MAX_VEHICLE_NUM, LANE_SIZE)
#define Q_TABLE_SIZE 8192 //STATE_SIZE * ACTION_SIZE

#define LANE_SIZE 4

#define ENV_ITEM_SIZE 15

#define ENV_SIZE MESH_SIZE*ENV_ITEM_SIZE

#define BLOCK_SIZE 1024
#define BLOCK_SQUARE_SIZE 32

#define ALPHA 0.5
#define BETA 0.1 //decay reward from surroundings
#define GAMMA 0.5
#define EPSILON 0.4
#define VEHICLE_RATE 0.3

#define VERBOSE 0

#define NUM_TRAIN 30000
#define NUM_TEST 10000

#define DEBUG 0

#define USE_CUDA 0


/*
0: cur_state_up
1: cur_state_down
2: cur_state_left
3: cur_state_right
4: cur_state_id
5: next_state_up
6: next_state_down
7: next_state_left
8: next_state_right
9: next_state_id
10: action // not used
11: in_vehicle_up
12: in_vehicle_down
13: in_vehicle_left
14: in_vehicle_right
*/
//action
__device__ int get_env(int* env, int mesh_id, int item_id) {
    return env[mesh_id * ENV_ITEM_SIZE + item_id];
}

__device__ void set_env(int* env, int mesh_id, int item_id, double value) {
    env[mesh_id * ENV_ITEM_SIZE + item_id] = value;
}

__device__ void set_env(int* env, int id, double value) {
    env[id] = value;
}


__device__ int* get_cur_state(int* env, int mesh_id) {
    return env + mesh_id * ENV_ITEM_SIZE;
}

__device__ int get_cur_state_id(int* env, int mesh_id) {
    return *(env + mesh_id * ENV_ITEM_SIZE + 4);
}

__device__ int* get_next_state(int* env, int mesh_id) {
    return env + mesh_id * ENV_ITEM_SIZE + 5;
}

__device__ void set_next_state_id(int* env, int mesh_id, int next_state_id) {
    *(env + mesh_id * ENV_ITEM_SIZE + 9) = next_state_id;
}

__device__ int get_next_state_id(int* env, int mesh_id) {
    return *(env + mesh_id * ENV_ITEM_SIZE + 9);
}

// __device__ double* get_reward(int* env, int id) {
//     return env + id * ENV_ITEM_SIZE + 8;
// }

__device__ int get_action(int* env, int mesh_id) {
    return *(env + mesh_id * ENV_ITEM_SIZE + 8);
}

__device__ int* get_in_vehicle(int* env, int mesh_id) {
    return env + mesh_id * ENV_ITEM_SIZE + 9;
}

__device__ void increase_in_vehicle(int* env, int mesh_id, int lane) {
    *(env + mesh_id * ENV_ITEM_SIZE + 9 + lane) += 1;
}

__global__ void reset_env_kernel(int* env) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= ENV_SIZE) {
        return;
    }
    set_env(env, id, 0);
}

__device__ int get_state(int* state) {
    int state_ret = 0;
    for (int i = 0; i < LANE_SIZE; i++){
        state_ret += state[i] * (int)pow(MAX_VEHICLE_NUM, i);
    }
    return state_ret;
}

__device__ int get_next_state_cuda(int* next_state, int* state, int action){
    int lane = action * 2;
    for (int i = 0; i < LANE_SIZE; i++){
        if (i == lane || i == (lane + 1)) {
            next_state[i] = state[i] > 0 ? state[i] - 1 : state[i];
        }
        else {
            next_state[i] = state[i];
        }            
    }
    return get_state(next_state);
}

__device__ double get_value_cuda(double* q_table, int id, int state, int action){
    return q_table[(state * ACTION_SIZE + action) + Q_TABLE_SIZE * id];
}

__device__ void set_value_cuda(double* q_table, int id, int state, int action, double value){
    q_table[(state * ACTION_SIZE + action) + Q_TABLE_SIZE * id] = value;
}

// __global__ void get_max_q_value_in_action(double* max_q, double* isMAX, double* q_table, int state) {
//     int row = threadIdx.x + blockIdx.x * blockDim.x;
//     int col = threadIdx.y + blockIdx.y * blockDim.y;
//     isMAX[row] = true;
//     if (get_value_cuda(q_table, state, row) < get_value_cuda(q_table, state, col)) {
//         isMAX[row] = false;
//     }
//     if (isMAX[row]) {
//         *max_q = get_value_cuda(q_table, state, row);
//     }
// }

__device__ double get_max_q_value_in_action_cuda(double* q_table, int id, int state) {
    double max_q = DBL_MIN;
    for (int i = 0; i < ACTION_SIZE; i++) {
        double value = get_value_cuda(q_table, id, state, i);
        
        if (value > max_q) {
            max_q = value;
        }
    }
    //printf("max_value: %f\n", max_q);
    return max_q;
}


/* can be a global */
__device__ double cal_reward_cuda(int* next_state, int* state, int* in_vehicle) {
    int max_next_lane_num = 0;
    for (int i = 0; i < LANE_SIZE; i++){
        if (next_state[i] > max_next_lane_num) {
            max_next_lane_num = next_state[i];
        }
    }
    int max_cur_lane_num = 0;
    for (int i = 0; i < LANE_SIZE; i++){
        if (state[i] > max_cur_lane_num) {
            max_cur_lane_num = state[i];
        }
    }
    double in_vehicle_num = 0;
    for (int i = 0; i < LANE_SIZE; i++){
        in_vehicle_num += in_vehicle[i];
    }
    double reward = max_cur_lane_num * max_cur_lane_num - max_next_lane_num * max_next_lane_num - in_vehicle_num * 0.025;

    return reward;
}

__device__ int choose_action_cuda(int mesh_id, double* q_table, int* env) {//, double* max_q) {
    int state_id = get_cur_state_id(env, mesh_id);
    double max_q_value = get_max_q_value_in_action_cuda(q_table, mesh_id, state_id);
    //*max_q = max_q_value;
    int action_id = 0;
    for (int i = 0; i < ACTION_SIZE; i++) {
        double value = get_value_cuda(q_table, mesh_id, state_id, i);
        if (value == max_q_value) {
            action_id = i;
            break;
        }
    }
    return action_id;
}

__device__ void cal_vehicle_out (int* environment, int mesh_id, int action) {
    if (action == 0) {
        set_env(environment, mesh_id, 13, get_cur_state(environment, mesh_id)[0] > 0 ? 1 : 0);
        set_env(environment, mesh_id, 14, get_cur_state(environment, mesh_id)[1] > 0 ? 1 : 0);
    }
    else {
        set_env(environment, mesh_id, 11, get_cur_state(environment, mesh_id)[2] > 0 ? 1 : 0);
        set_env(environment, mesh_id, 12, get_cur_state(environment, mesh_id)[3] > 0 ? 1 : 0);
    }
}

__device__ void update_vehicle_in_with_out (int* environment, int mesh_id, int action) {
    if (action == 0) {
        set_env(environment, mesh_id, 11, get_in_vehicle(environment, mesh_id)[2]);
        set_env(environment, mesh_id, 12, get_in_vehicle(environment, mesh_id)[3]);
        set_env(environment, mesh_id, 13, 0);
        set_env(environment, mesh_id, 14, 0);
    }
    else {
         set_env(environment, mesh_id, 13, get_in_vehicle(environment, mesh_id)[0]);
        set_env(environment, mesh_id, 14, get_in_vehicle(environment, mesh_id)[1]);
        set_env(environment, mesh_id, 11, 0);
        set_env(environment, mesh_id, 12, 0);
    }
}

__device__ void update_vehicle_in (int* environment, int mesh_id) {
    for (int i = 0; i < LANE_SIZE; i++) {
        set_env(environment, mesh_id, 0 + i, get_in_vehicle(environment, mesh_id)[i]);
        set_env(environment, mesh_id, 11 + i, 0);
    }
}

__global__ void update_env_pre_kernel(int* environment, double* q_table) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;
    if (id > MESH_SIZE) {
        return;
    }
    int action = choose_action_cuda(id, q_table, environment);
    int* state = get_cur_state(environment, id);
    int next_state_id = get_next_state_cuda(get_next_state(environment, id), state, action);  // update next state, get next state id
    set_next_state_id(environment, id, next_state_id);
    cal_vehicle_out(environment, id, action);
    update_vehicle_in_with_out(environment, id, action);
}

__global__ void update_reward_kernel(int* environment, double* reward) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int mesh_id = x + y * blockDim.x * gridDim.x;
    if (mesh_id > MESH_SIZE) {
        return;
    }

    int max_next_lane_num = 0;
    for (int i = 0; i < LANE_SIZE; i++){
        int* next_state = get_next_state(environment, mesh_id);
        if (next_state[i] > max_next_lane_num) {
            max_next_lane_num = next_state[i];
        }
    }
    int max_cur_lane_num = 0;
    for (int i = 0; i < LANE_SIZE; i++){
        int* cur_state = get_cur_state(environment, mesh_id);
        if (cur_state[i] > max_cur_lane_num) {
            max_cur_lane_num = cur_state[i];
        }
    }

    double in_vehicle_num = 0;
    for (int i = 0; i < LANE_SIZE; i++){
        in_vehicle_num += get_in_vehicle(environment, mesh_id)[i];
    }
    reward[mesh_id] = max_cur_lane_num * max_cur_lane_num - max_next_lane_num * max_next_lane_num - in_vehicle_num * 0.025;
}

__global__ void cal_q_kernel_single(int* environment, double* reward, double* q_table, int mesh_id) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int action = threadIdx.y;
    if (id >= SURROUNDING_SIZE) {
        return;
    }
    int* state = get_cur_state(environment, id);
    int* actual_state = new int[LANE_SIZE];
    int* next_state = new int[LANE_SIZE];
    for (int i = 0; i < LANE_SIZE; i++){
        if ((id - 2) / 2 == i) {
            int tmp_state = state[i] + (id % 2 == 0 ? 1 : -1);
            actual_state[i] = tmp_state > 0 ? (tmp_state < MAX_VEHICLE_NUM ? tmp_state : MAX_VEHICLE_NUM) : 0;
        }
        else {
            actual_state[i] = (int)state[i];
        }
    }

    int state_int = get_state(actual_state);
    
    // double* isMAX;
    // if (cudaMalloc(&isMAX, sizeof(double) * ACTION_SIZE) != cudaSuccess) {
        
    // }
    double cur_q = get_value_cuda(q_table, mesh_id, state_int, action);
    int next_state_int = get_next_state_cuda(next_state, actual_state, action);

    double next_q = 0;
    next_q = get_max_q_value_in_action_cuda(q_table, mesh_id, next_state_int);

    //get_max_q_value_in_action<<<ACTION_SIZE, ACTION_SIZE>>>(&max_q, isMAX, q_table, next_state);
    //cudaFree(isMAX);
    double local_reward = reward[mesh_id];

    double surrounding_reward = 0;

    int s_id = mesh_id - SIZE_X;
    if (s_id > 0) {
        surrounding_reward += reward[s_id];
    }
    s_id = mesh_id + SIZE_X;
    if (s_id < MESH_SIZE) {
        surrounding_reward += reward[s_id];
    }
    s_id = mesh_id - 1;
    if (mesh_id % SIZE_X != 0) {
        surrounding_reward += reward[s_id];
    }
    s_id = mesh_id + 1;
    if (mesh_id % SIZE_X != SIZE_X - 1) {
        surrounding_reward += reward[s_id];
    }

    double q_value = 0;
    q_value += (1 - ALPHA) * cur_q;
    q_value += ALPHA * local_reward + BETA * surrounding_reward;
    q_value += ALPHA * next_q;
    set_value_cuda(q_table, mesh_id, state_int, action, q_value);
}

__device__ void generate_new_vehicel(int* environment, int mesh_id) {
    if (mesh_id < SIZE_X) {
        double rand_num = (double)rand() / RAND_MAX;
        if (rand_num < VEHICLE_RATE) {
            increase_in_vehicle(environment, mesh_id, 0);
        }
    }
    if (mesh_id >= MESH_SIZE - SIZE_X) {
        double rand_num = (double)rand() / RAND_MAX;
        if (rand_num < VEHICLE_RATE) {
            increase_in_vehicle(environment, mesh_id, 1);
        }
    }
    if (mesh_id % SIZE_X == 0) {
        double rand_num = (double)rand() / RAND_MAX;
        if (rand_num < VEHICLE_RATE) {
            increase_in_vehicle(environment, mesh_id, 2);
        }
    }
    if (mesh_id % SIZE_X == SIZE_X - 1) {
        double rand_num = (double)rand() / RAND_MAX;
        if (rand_num < VEHICLE_RATE) {
            increase_in_vehicle(environment, mesh_id, 3);
        }
    }
}

__device__ void take_next_step(int* environment, int mesh_id) {
    for (int i = 0; i < LANE_SIZE; i++) {
        set_env(environment, mesh_id, i, get_next_state(environment, mesh_id)[i]);
    }
    set_env(environment, mesh_id, 4, get_next_state_id(environment, mesh_id));
}

__global__ void update_env_after_kernel(int* environment, double* q_table) {
    int mesh_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (mesh_id >= MESH_SIZE) {
        return;
    }
    generate_new_vehicel(environment, mesh_id);
    update_vehicle_in(environment, mesh_id);
    take_next_step(environment, mesh_id);
}

// __global__ void get_coverage_kernel(int* coverage, double* q_table) {
//     __shared__ int not_covered_count;
//     if (threadIdx.x == 0) {
//         not_covered_count = 0;
//     }
//     int state = threadIdx.x + blockIdx.x * blockDim.x;
//     if (state >= STATE_SIZE) {
//         return;
//     }
//     int main_offset = state * ACTION_SIZE;
//     bool covered_flag = false;
//     for (int i = 0; i < ACTION_SIZE; i++) {
//         if (get_value_cuda(q_table,  state, i) != 0) {
//             covered_flag = true;
//             break;
//         }
//     }
//     if (!covered_flag) {
//         atomicAdd(&not_covered_count, 1);
//     }
//     __syncthreads();
//     if (threadIdx.x == 0) {
//         coverage[blockIdx.x] = not_covered_count;
//     }
// }


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
    int* cur_state; // 0 - 8 number of cars in each direction top, top-turn, right, right-turn, bottom, bottom-turn, left, left-turn, 9 - 12 number of new cars in each lane
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
        double max_action_val = DBL_MIN;
        int state_offset = this -> cur_state_int;
        this -> cur_action = 0;
        for (int i = 0; i < this -> action_size; i++) {
            int val = this -> q_table.get_value(state_offset, i);
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
        // double next_reward = 0;
        // int larger_val_h = this -> next_state[2] > this -> next_state[3] ? this -> next_state[2] : this -> next_state[3];
        // int larger_val_v = this -> next_state[0] > this -> next_state[1] ? this -> next_state[0] : this -> next_state[1];
        // if (this -> cur_action == 0) {
        //     next_reward += larger_val_v * 0.3;
        //     next_reward -= larger_val_h * larger_val_h / this -> max_vehicle_num;
        // }
        // else {
        //     next_reward += larger_val_h * 0.3;
        //     next_reward -= larger_val_v * larger_val_v / this -> max_vehicle_num;
        // }

        int max_next_lane_num = 0;
        for (int i = 0; i < this -> lane_size; i++){
            if (this -> next_state[i] > max_next_lane_num) {
                max_next_lane_num = this -> next_state[i];
            }
        }
        int max_cur_lane_num = 0;
        for (int i = 0; i < this -> lane_size; i++){
            if (this -> cur_state[i] > max_cur_lane_num) {
                max_cur_lane_num = this -> cur_state[i];
            }
        }
        double in_vehicle_num = 0;
        for (int i = 0; i < this -> lane_size; i++){
            in_vehicle_num += this -> in_vehicle[i];
        }
        double reward = max_cur_lane_num * max_cur_lane_num - max_next_lane_num * max_next_lane_num - in_vehicle_num * 0.025;

        // int next_vehicle_num = this -> cal_vehicle_num(this -> next_state);
        // int cur_vehicle_num = this -> cal_vehicle_num(this -> cur_state);
        //reward = cur_vehicle_num - next_vehicle_num;
        return reward;
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
            double val = this -> q_table.get_value(this -> next_state_int, i);
            if (val > max_action_val) {
                max_action_val = val;
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
        cout << "action: " << this -> cur_action << " q_val: " << this -> q_table.get_value(this -> cur_state_int, 0) << " " << this -> q_table.get_value(this -> cur_state_int, 1)<< endl;
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
            //this -> mesh[i].update_with_in_vehicle();
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
            q_value += this -> alpha * (reward[i]) + this -> beta * surrounding_reward;
            q_value += this -> alpha * this -> mesh[i].get_next_q_value();
            #if DEBUG
                cout << "update q value - intersection: " << i << " reward: " << reward[i] << " surrounding: " << surrounding_reward << " q value: " << q_value << endl;
            #endif
            this -> mesh[i].set_q_value(q_value);
        }
    }

    void update_in_vehicle() {
        for (int i = 0; i < this -> mesh_size; i++) {
            this -> mesh[i].update_with_in_vehicle();
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
        cout << "================" << endl;
        for (int i = 0; i < this -> mesh_size; i++) {
            //cout << "intersection: " << i << endl;
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
                this -> update_in_vehicle();

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
            if (train_step > NUM_TEST) {
                cout << "Run succeed" << endl;
                break;
            }

            #if DEBUG
                for (int i = 0; i < this -> mesh_size; i++) {
                    cout << "train - intersection: " << i << " cur_state: " << this -> mesh[i].cur_state_int << endl;
                }
            #endif

            cout << "step: " << train_step << endl;
            for (int i = 0; i < this -> mesh_size; i++) {
                this -> mesh[i].print_status();
            }

            this -> choose_max_action();
            this -> get_next_state();
            this -> update_out_vehicle();
            this -> sync();
            //this -> update_q_value();
            this -> update_in_vehicle();

            this -> generate_new_vehicle();
            this -> take_next_state();

            train_step ++;
        }
        cout << "Steps: " << train_step << endl;
    }
};

void malloc_q_table(double* q_table, int size) {
    if(cudaMalloc(&q_table, sizeof(double) * size) != cudaSuccess){
        cout << "Could not allocate q_table on GPU" << endl;
    }
}

void train() {
    int train_step = 0;
    int epoch_step = 0;

    int state_size = ((int)pow(MAX_VEHICLE_NUM, LANE_SIZE));
    int q_table_size = state_size * ACTION_SIZE;

    cout << train_step << endl;
    //QTable qtable = QTable();
    double* qtable;
    if (cudaMalloc(&qtable, sizeof(double) * MESH_SIZE * q_table_size) != cudaSuccess) {
        cout << "qtable malloc failed" << endl;
    }

    int* environment;
    if (cudaMalloc(&environment, sizeof(int) * ENV_SIZE) != cudaSuccess) {
        cout << "env malloc failed" << endl;
    }
    double* reward;
    if (cudaMalloc(&reward, sizeof(double) * MESH_SIZE) != cudaSuccess) {
        cout << "reward malloc failed" << endl;
    }
    dim3 env_block(ENV_SIZE < BLOCK_SIZE ? (ENV_SIZE > 32 ? ENV_SIZE : 32) : BLOCK_SIZE);
    dim3 env_grid(ENV_SIZE / env_block.x + 1);
    dim3 mesh_block(SIZE_X < BLOCK_SQUARE_SIZE ? SIZE_X : BLOCK_SQUARE_SIZE, SIZE_Y < BLOCK_SQUARE_SIZE ? SIZE_Y : BLOCK_SQUARE_SIZE);
    dim3 mesh_grid(SIZE_X / mesh_block.x + 1, SIZE_Y / mesh_block.y + 1);
    dim3 q_block(SURROUNDING_SIZE < BLOCK_SIZE / ACTION_SIZE ? SURROUNDING_SIZE : BLOCK_SIZE / ACTION_SIZE, 2);
    dim3 q_grid(SURROUNDING_SIZE / q_block.x + 1);
    while (train_step < NUM_TRAIN) {
        cout << "======================" << endl;
        cout << "start epoch: " << epoch_step << endl;
        reset_env_kernel<<<env_grid, env_block>>>(environment);
        update_env_pre_kernel<<<mesh_grid, mesh_block>>>(environment, qtable);
        update_reward_kernel<<<mesh_grid, mesh_block>>>(environment, reward);
        for (int i = 0; i < MESH_SIZE; i++) {
            cal_q_kernel_single<<<q_grid, q_block>>>(environment, qtable, reward, i);
        }
        update_env_after_kernel<<<mesh_grid, mesh_block>>>(environment, qtable);
    }
}

int main() {
    // IntersectionMesh mesh = IntersectionMesh();
    // mesh.train();
    // //mesh.print();
    // mesh.run();
    train();
    return 0;
}