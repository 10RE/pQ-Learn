#include <iostream>
#include "math.h"
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

using namespace std;

#define SIZE_X 8
#define SIZE_Y 8
#define MESH_SIZE SIZE_X*SIZE_Y
#define ACTION_SIZE 2
#define SURROUNDING_SIZE 9
#define Q_KERNEL_SIZE SURROUNDING_SIZE * MESH_SIZE
#define MAX_VEHICLE_NUM 8

#define LANE_SIZE 4

#define STATE_SIZE (int)pow(MAX_VEHICLE_NUM, LANE_SIZE)
#define Q_TABLE_SIZE STATE_SIZE * ACTION_SIZE
#define TOTAL_Q_TABLE_SIZE Q_TABLE_SIZE * MESH_SIZE

#define ENV_ITEM_SIZE 15

#define ENV_SIZE MESH_SIZE*ENV_ITEM_SIZE

#define BLOCK_SIZE 1024
#define BLOCK_SQUARE_SIZE 32

#define ALPHA 0.5
#define BETA 0.1 //decay reward from surroundings
#define GAMMA 0.1
#define EPSILON 0.4
#define VEHICLE_RATE 0.3

#define VERBOSE 0

#define NUM_TRAIN 1000000
#define NUM_TEST 10000
#define NUM_TEST_ITER 20

#define NO_SURROUNDING 0

#define DEBUG 0

#define SHOW_STATE 0

#define USE_CUDA 0

__global__ void get_coverage_kernel(int* coverage, double* q_table) {
    __shared__ int not_covered_count;
    if (threadIdx.x == 0) {
        not_covered_count = 0;
    }
    int state = threadIdx.x + blockIdx.x * blockDim.x;
    if (state >= (TOTAL_Q_TABLE_SIZE / 2)) {
        return;
    }
    int main_offset = state * ACTION_SIZE;
    bool covered_flag = false;
    for (int i = 0; i < ACTION_SIZE; i++) {
        if (q_table[main_offset + i] != 0) {
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
    int grid_size = TOTAL_Q_TABLE_SIZE / ACTION_SIZE / block.x;
    dim3 grid(grid_size);
    int* cover_count = new int[grid_size];
    int* cover_count_cuda;
    if (cudaMalloc(&cover_count_cuda, sizeof(int) * grid_size) != cudaSuccess) {
        printf("cudaMalloc for coverage failed\n");
    }
    get_coverage_kernel<<<grid, block>>>(cover_count_cuda, q_table);
    cudaMemcpy(cover_count, cover_count_cuda, sizeof(int) * grid_size, cudaMemcpyDeviceToHost);
    int total = 0;
    //cout << TOTAL_Q_TABLE_SIZE << " " << grid_size << endl;
    for (int i = 0; i < grid_size; i++) {
        //cout << cover_count[i] << " ";
        total += cover_count[i];
    }
    //cout << total << endl;
    cudaFree(cover_count_cuda);
    return 1 - (double)total / (double)(TOTAL_Q_TABLE_SIZE / ACTION_SIZE);
}

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

__device__ void set_cur_state_id(int* env, int mesh_id, int cur_state_id) {
    *(env + mesh_id * ENV_ITEM_SIZE + 4) = cur_state_id;
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
    return *(env + mesh_id * ENV_ITEM_SIZE + 10);
}

__device__ void set_action(int* env, int mesh_id, int action) {
    *(env + mesh_id * ENV_ITEM_SIZE + 10) = action;
}

__device__ int* get_in_vehicle(int* env, int mesh_id) {
    return env + mesh_id * ENV_ITEM_SIZE + 11;
}

__device__ void increase_in_vehicle(int* env, int mesh_id, int lane) {
    *(env + mesh_id * ENV_ITEM_SIZE + 11 + lane) += 1;
}

__global__ void reset_env_kernel(int* env, bool* is_end_state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= ENV_SIZE) {
        return;
    }
    if (*is_end_state) {
        set_env(env, id, 0);
        return;
    }
    
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
    return q_table[Q_TABLE_SIZE * id + state * ACTION_SIZE + action];
}

__device__ void set_value_cuda(double* q_table, int id, int state, int action, double value){
    q_table[Q_TABLE_SIZE * id + state * ACTION_SIZE + action] = value;
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
    double max_q = -999;
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

__device__ int choose_max_action_cuda(int mesh_id, double* q_table, int* env) {//, double* max_q) {
    int state_id = get_cur_state_id(env, mesh_id);
    double max_q = -999;
    int action_id = 0;
    #if DEBUG
        double value0 = get_value_cuda(q_table, mesh_id, state_id, 0);
        double value1 = get_value_cuda(q_table, mesh_id, state_id, 1);
    #endif

    for (int i = 0; i < ACTION_SIZE; i++) {
        double value = get_value_cuda(q_table, mesh_id, state_id, i);
        
        if (value > max_q) {
            max_q = value;
            action_id = i;
        }
    }
    #if DEBUG
    printf("max_action %d, %d, %d: %f, %f\n", mesh_id, action_id, state_id, value0, value1);
    #endif
    //printf("max_value: %f\n", max_q);
    return action_id;
}

__device__ int choose_random_action_cuda(int mesh_id, curandState* rand_state) {
    double rand_num = curand_uniform(&rand_state[mesh_id]);
    return (int) (rand_num * ACTION_SIZE);
}

__device__ int choose_action_cuda(int mesh_id, double* q_table, int* env, curandState* rand_state) {
    double rand_num = curand_uniform(&rand_state[mesh_id]);
    if (rand_num < EPSILON) {
        return choose_random_action_cuda(mesh_id, rand_state);
    }
    else {
        return choose_max_action_cuda(mesh_id, q_table, env);
    }
}

__device__ void cal_vehicle_out (int* environment, int mesh_id, int action) {
    if (action == 0) {
        if (mesh_id - SIZE_X >= 0) {
            //up out vehicle
            set_env(environment, mesh_id - SIZE_X, 12, get_cur_state(environment, mesh_id)[1] > 0 ? 1 : 0);
        }
        if (mesh_id + SIZE_X < MESH_SIZE) {
            //down out vehicle
            set_env(environment, mesh_id + SIZE_X, 11, get_cur_state(environment, mesh_id)[0] > 0 ? 1 : 0);
        }
    }
    else {
        if (mesh_id % SIZE_X != 0) {
            //left out vehicle
            set_env(environment, mesh_id - 1, 14, get_cur_state(environment, mesh_id)[3] > 0 ? 1 : 0);
        }
        if (mesh_id % SIZE_X != SIZE_X - 1) {
            //down out vehicle
            set_env(environment, mesh_id + 1, 13, get_cur_state(environment, mesh_id)[2] > 0 ? 1 : 0);
        }
    }
    #if DEBUG
        __syncthreads();
        printf("out_vehicle: %d, %d, %d, %d, %d\n", mesh_id, get_in_vehicle(environment, mesh_id)[0], get_in_vehicle(environment, mesh_id)[1], get_in_vehicle(environment, mesh_id)[2], get_in_vehicle(environment, mesh_id)[3]);
    #endif
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
        set_env(environment, mesh_id, 5 + i, get_next_state(environment, mesh_id)[i] + get_in_vehicle(environment, mesh_id)[i]);
        //set_env(environment, mesh_id, 11 + i, 0);
    }
    #if DEBUG
        printf("in_vehicle: %d, %d, %d, %d, %d\n", mesh_id, get_in_vehicle(environment, mesh_id)[0], get_in_vehicle(environment, mesh_id)[1], get_in_vehicle(environment, mesh_id)[2], get_in_vehicle(environment, mesh_id)[3]);
        int* state = get_cur_state(environment, mesh_id);
        int* next_state = get_next_state(environment, mesh_id);
        printf("update__vehicle_in: %d, cur_state: %d, %d, %d, %d, next_state: %d, %d, %d, %d \n", mesh_id, state[0], state[1], state[2], state[3], next_state[0], next_state[1], next_state[2], next_state[3]);
    #endif
}

__device__ void reset_vehicle_in (int* environment, int mesh_id) {
    for (int i = 0; i < LANE_SIZE; i++) {
        set_env(environment, mesh_id, 11 + i, 0);
    }
}

__global__ void update_env_pre_kernel(int* environment, double* q_table, curandState* rand_state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= MESH_SIZE) {
        return;
    }
    int* state = get_cur_state(environment, id);
    int cur_state_id = get_state(state);
    set_cur_state_id(environment, id, cur_state_id);

    int action = choose_action_cuda(id, q_table, environment, rand_state);

    set_action(environment, id, action);
    
    int next_state_id = get_next_state_cuda(get_next_state(environment, id), state, action);  // update next state, get next state id
    #if SHOW_STATE
        printf("env_prev: %d, action: %d, cur_state: %d, %d, %d, %d, next_state: %d, %d, %d, %d \n", id, action, state[0], state[1], state[2], state[3], get_next_state(environment, id)[0], get_next_state(environment, id)[1], get_next_state(environment, id)[2], get_next_state(environment, id)[3]);
    #endif
    reset_vehicle_in(environment, id);
    set_next_state_id(environment, id, next_state_id);
    //__syncthreads();
    cal_vehicle_out(environment, id, action);
    //__syncthreads();
    //update_vehicle_in_with_out(environment, id, action);
}

__global__ void update_env_pre_run_kernel(int* environment, double* q_table, curandState* rand_state, bool show_state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= MESH_SIZE) {
        return;
    }
    int* state = get_cur_state(environment, id);
    int cur_state_id = get_state(state);
    set_cur_state_id(environment, id, cur_state_id);

    int action = choose_max_action_cuda(id, q_table, environment);
    
    set_action(environment, id, action);

    int next_state_id = get_next_state_cuda(get_next_state(environment, id), state, action);  // update next state, get next state id
    if (show_state) {
        printf("env_prev: %d, action: %d, cur_state: %d, %d, %d, %d, next_state: %d, %d, %d, %d, q_0: %f, q_1: %f \n", id, action, state[0], state[1], state[2], state[3], get_next_state(environment, id)[0], get_next_state(environment, id)[1], get_next_state(environment, id)[2], get_next_state(environment, id)[3], get_value_cuda(q_table, id, get_cur_state_id(environment, id), 0), get_value_cuda(q_table, id, get_cur_state_id(environment, id), 1));
    }
    reset_vehicle_in(environment, id);
    set_next_state_id(environment, id, next_state_id);
    //__syncthreads();
    cal_vehicle_out(environment, id, action);
    //__syncthreads();
    //update_vehicle_in_with_out(environment, id, action);
}

__global__ void update_reward_kernel(int* environment, double* reward) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int mesh_id = x;
    if (mesh_id >= MESH_SIZE) {
        return;
    }

    int max_next_lane_num = 0;
    int next_lane_sum = 0;
    int* next_state = get_next_state(environment, mesh_id);
    // if (action_id == get_action(environment, mesh_id)) {
    //     next_state = get_next_state(environment, mesh_id);
    // } else {
    //     next_state = ;
    // }
    for (int i = 0; i < LANE_SIZE; i++){
        
        next_lane_sum += next_state[i];
        if (next_state[i] > max_next_lane_num) {
            max_next_lane_num = next_state[i];
        }
    }
    int max_cur_lane_num = 0;
    int cur_lane_sum = 0;
    int* cur_state = get_cur_state(environment, mesh_id);
    for (int i = 0; i < LANE_SIZE; i++){
        cur_lane_sum += cur_state[i];
        if (cur_state[i] > max_cur_lane_num) {
            max_cur_lane_num = cur_state[i];
        }
    }

    double in_vehicle_num = 0;
    for (int i = 0; i < LANE_SIZE; i++){
        in_vehicle_num += get_in_vehicle(environment, mesh_id)[i];
    }
    reward[mesh_id] = 0.25 * (cur_lane_sum - next_lane_sum) + max_cur_lane_num * max_cur_lane_num - max_next_lane_num * max_next_lane_num - in_vehicle_num * 0.025;
    //reward[mesh_id] = max_cur_lane_num * max_cur_lane_num - max_next_lane_num * max_next_lane_num - in_vehicle_num * 0.025;
    #if DEBUG
        printf("reward: %d, %f\n", mesh_id, reward[mesh_id]);
    #endif
}

__device__ double cal_reward(int* environment, int mesh_id, int* cur_state, int* next_state) {
    int max_next_lane_num = 0;
    int next_lane_sum = 0;
    for (int i = 0; i < LANE_SIZE; i++){
        next_lane_sum += next_state[i];
        if (next_state[i] > max_next_lane_num) {
            max_next_lane_num = next_state[i];
        }
    }
    int max_cur_lane_num = 0;
    int cur_lane_sum = 0;
    for (int i = 0; i < LANE_SIZE; i++){
        cur_lane_sum += cur_state[i];
        if (cur_state[i] > max_cur_lane_num) {
            max_cur_lane_num = cur_state[i];
        }
    }

    double in_vehicle_num = 0;
    for (int i = 0; i < LANE_SIZE; i++){
        in_vehicle_num += get_in_vehicle(environment, mesh_id)[i];
    }
    return 0.25 * (cur_lane_sum - next_lane_sum) + max_cur_lane_num * max_cur_lane_num - max_next_lane_num * max_next_lane_num - in_vehicle_num * 0.025;
}

__global__ void debug_reward_kernel(double* reward) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int mesh_id = x + y * blockDim.x * gridDim.x;
    if (mesh_id >= MESH_SIZE) {
        return;
    }
    printf("reward_debug: %d, %f\n", mesh_id, reward[mesh_id]);
}

__global__ void cal_q_kernel_single(int* environment, double* reward, double* q_table) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int action = threadIdx.y;

    int surrounding_id = id % SURROUNDING_SIZE;
    #if NO_SURROUNDING
    if (surrounding_id != 0) {
        return;
    }
    #endif
    int mesh_id = id / SURROUNDING_SIZE;
    #if NO_SURROUNDING 
    if (action != get_action(environment, mesh_id)) {
        return;
    }
    #endif
    if (id >= Q_KERNEL_SIZE) {
        return;
    }
    if (action >= ACTION_SIZE) {
        return;
    }
    
    
    int* state = get_cur_state(environment, mesh_id);
    int actual_state[LANE_SIZE] = {0, 0, 0, 0};//new int[LANE_SIZE];
    int next_state[LANE_SIZE] = {0, 0, 0, 0};
    if (surrounding_id == 0) {
        for (int i = 0; i < LANE_SIZE; i++){
            actual_state[i] = (int)state[i];
        }
    }
    else {
        for (int i = 0; i < LANE_SIZE; i++){
            if ((surrounding_id - 1) / 2 == i) {
                int tmp_state = state[i] + (surrounding_id % 2 == 0 ? 1 : -1);
                if (tmp_state < 0 || tmp_state > MAX_VEHICLE_NUM) {
                    return;
                }
                actual_state[i] = tmp_state > 0 ? (tmp_state < MAX_VEHICLE_NUM ? tmp_state : MAX_VEHICLE_NUM) : 0;
            }
            else {
                actual_state[i] = (int)state[i];
            }
        }
    }

    #if NO_SURROUNDING
        int state_int = get_state(actual_state);
        int state_int = get_cur_state_id(environment, mesh_id);

        double cur_q = get_value_cuda(q_table, mesh_id, state_int, action);
        int next_state_int = get_next_state_id(environment, mesh_id);

        double local_reward = reward[mesh_id];

    #else
        int state_int = get_state(actual_state);
    
        double cur_q = get_value_cuda(q_table, mesh_id, state_int, action);
        int next_state_int = get_next_state_cuda(next_state, actual_state, action);

        double local_reward = cal_reward(environment, mesh_id, actual_state, next_state);
    #endif 

    double next_q = 0;
    next_q = get_max_q_value_in_action_cuda(q_table, mesh_id, next_state_int);

    //get_max_q_value_in_action<<<ACTION_SIZE, ACTION_SIZE>>>(&max_q, isMAX, q_table, next_state);
    //cudaFree(isMAX);
    //double local_reward = reward[mesh_id];
    
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
    if (mesh_id % SIZE_X != (SIZE_X - 1)) {
        surrounding_reward += reward[s_id];
    }

    double q_value = 0;
    q_value += (1 - ALPHA) * cur_q;
    q_value += ALPHA * local_reward + BETA * surrounding_reward;
    q_value += ALPHA * GAMMA * next_q;
    //__syncthreads();
    set_value_cuda(q_table, mesh_id, state_int, action, q_value);

    #if DEBUG
    __syncthreads();
    printf("q_value %d state: %d, %d, %d, %d, %d, action: %d, next_state: %d, %d, %d, %d, q_val: %f, reward: %f, surrounding: %f, q_val_0: %f, q_val_1: %f\n", mesh_id, state_int, actual_state[0], actual_state[1], actual_state[2], actual_state[3], action, next_state[0], next_state[1], next_state[2], next_state[3], next_q, local_reward, surrounding_reward, get_value_cuda(q_table, mesh_id, state_int, 0), get_value_cuda(q_table, mesh_id, state_int, 1));
    #endif
}

__device__ void generate_new_vehicel(int* environment, int mesh_id, curandState* rand_state) {
    if (mesh_id < SIZE_X) {
        double rand_num = curand_uniform(&rand_state[mesh_id]);
        if (rand_num < VEHICLE_RATE) {
            increase_in_vehicle(environment, mesh_id, 0);
        }
    }
    if (mesh_id >= MESH_SIZE - SIZE_X) {
        double rand_num = curand_uniform(&rand_state[mesh_id]);
        if (rand_num < VEHICLE_RATE) {
            increase_in_vehicle(environment, mesh_id, 1);
        }
    }
    if (mesh_id % SIZE_X == 0) {
        double rand_num = curand_uniform(&rand_state[mesh_id]);
        if (rand_num < VEHICLE_RATE) {
            increase_in_vehicle(environment, mesh_id, 2);
        }
    }
    if (mesh_id % SIZE_X == SIZE_X - 1) {
        double rand_num = curand_uniform(&rand_state[mesh_id]);
        if (rand_num < VEHICLE_RATE) {
            increase_in_vehicle(environment, mesh_id, 3);
        }
    }
}

__device__ void take_next_step(int* environment, int mesh_id) {
    for (int i = 0; i < LANE_SIZE; i++) {
        set_env(environment, mesh_id, i, get_next_state(environment, mesh_id)[i]);
    }
    //set_env(environment, mesh_id, 4, get_next_state_id(environment, mesh_id));
}

__global__ void update_env_after_kernel(int* environment, double* q_table, curandState* rand_state) {
    int mesh_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (mesh_id >= MESH_SIZE) {
        return;
    }
    generate_new_vehicel(environment, mesh_id, rand_state);
    update_vehicle_in(environment, mesh_id);
    take_next_step(environment, mesh_id);
}

__global__ void is_end_state_kernel(int* environment, bool* is_end) {
    int mesh_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (mesh_id >= MESH_SIZE) {
        return;
    }
    int* state = get_cur_state(environment, mesh_id);
    for (int i = 0; i < LANE_SIZE; i++) {
        if (state[i] > MAX_VEHICLE_NUM) {
            *is_end = true;
            #if DEBUG
                printf("Is end state: %d state: %d %d %d %d\n", mesh_id, state[0], state[1], state[2], state[3]);
            #endif
            return;
        }
    }
    #if DEBUG
        printf("Is not end state: %d state: %d %d %d %d\n", mesh_id, state[0], state[1], state[2], state[3]);
    #endif
    return;
}

__global__ void reset_is_end_state_kernel(bool* is_end) {
    *is_end = false;
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

void malloc_q_table(double* q_table, int size) {
    if(cudaMalloc(&q_table, sizeof(double) * size) != cudaSuccess){
        cout << "Could not allocate q_table on GPU" << endl;
    }
}

__global__ void rand_setup_kernel(curandState* rand_state)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(clock64(), tid, 0, &rand_state[tid]);
}

void train(double* qtable) {
    int train_step = 0;
    int epoch_step = 0;

    curandState* rand_states;
    if (cudaMalloc(&rand_states, sizeof(curandState) * MESH_SIZE) != cudaSuccess) {
        cout << "curandState malloc failed" << endl;
    }
    rand_setup_kernel<<<MESH_SIZE, 1>>>(rand_states);

    // cout << train_step << endl;
    //QTable qtable = QTable();

    int* environment;
    if (cudaMalloc(&environment, sizeof(int) * ENV_SIZE) != cudaSuccess) {
        cout << "env malloc failed" << endl;
    }
    double* reward;
    if (cudaMalloc(&reward, sizeof(double) * MESH_SIZE * ACTION_SIZE) != cudaSuccess) {
        cout << "reward malloc failed" << endl;
    }
    bool* is_end_state_cuda;
    bool is_end_state = false;
    if (cudaMalloc(&is_end_state_cuda, sizeof(bool)) != cudaSuccess) {
        cout << "is_end_state malloc failed" << endl;
    }
    int* environment_cpu = (int*)malloc(sizeof(int) * ENV_SIZE);
    dim3 env_block(ENV_SIZE < BLOCK_SIZE ? (ENV_SIZE > 32 ? ENV_SIZE : 32) : BLOCK_SIZE);
    dim3 env_grid(ENV_SIZE / env_block.x + 1);
    dim3 mesh_block(MESH_SIZE < BLOCK_SQUARE_SIZE ? MESH_SIZE : BLOCK_SQUARE_SIZE);
    dim3 mesh_grid(MESH_SIZE / mesh_block.x + 1);
    dim3 reward_block(MESH_SIZE < BLOCK_SIZE ? MESH_SIZE : BLOCK_SIZE);
    dim3 reward_grid(MESH_SIZE / mesh_block.x + 1);
    dim3 q_block(Q_KERNEL_SIZE < BLOCK_SIZE / ACTION_SIZE ? Q_KERNEL_SIZE : BLOCK_SIZE / ACTION_SIZE, ACTION_SIZE);
    dim3 q_grid(Q_KERNEL_SIZE / q_block.x + 1);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    while (train_step < NUM_TRAIN) {
        // cout << "======================" << endl;
        // cout << "start epoch: " << epoch_step << endl;
            
        #if DEBUG
            // if (cudaMemcpy(
            //     environment_cpu, 
            //     environment, 
            //     sizeof(int) * ENV_SIZE,
            //     cudaMemcpyDeviceToHost
            // ) != cudaSuccess) {
            //     cout << "Could not copy to CPU" << endl;
            // }
            // for (int i = 0; i < MESH_SIZE; i++) {
            //     cout << "intersection: " << i << endl;
            //     for (int j = 0; j < LANE_SIZE; j++) {
            //         cout << environment_cpu[i * ENV_ITEM_SIZE + j] << " ";
            //     }
            //     cout << endl;
            // }
        #endif
        
        is_end_state_kernel<<<mesh_grid, mesh_block>>>(environment, is_end_state_cuda);
        //cudaEventRecord(start, 0);
        reset_env_kernel<<<env_grid, env_block>>>(environment, is_end_state_cuda);
        reset_is_end_state_kernel<<<1, 1>>>(is_end_state_cuda);

        // cout << "----------------------" << endl;
        // cout << "train step: " << train_step << endl;
        
        //debug_reward_kernel<<<mesh_grid, mesh_block>>>(reward);
        
        update_env_pre_kernel<<<mesh_grid, mesh_block>>>(environment, qtable, rand_states);
        update_reward_kernel<<<reward_grid, reward_block>>>(environment, reward);
        cal_q_kernel_single<<<q_grid, q_block>>>(environment, reward, qtable);
        update_env_after_kernel<<<mesh_grid, mesh_block>>>(environment, qtable, rand_states);
        
        
        //cudaDeviceSynchronize();
        train_step ++;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout << "Time: " << time << endl;

    cudaFree(environment);
    cudaFree(reward);
    cudaFree(is_end_state_cuda);
    cudaFree(rand_states);

    // double* qtable_cpu = (double*)malloc(sizeof(double) * MESH_SIZE * q_table_size);
    // if (cudaMemcpy(
    //     qtable_cpu, 
    //     qtable, 
    //     sizeof(double) * MESH_SIZE * q_table_size,
    //     cudaMemcpyDeviceToHost
    // ) != cudaSuccess){
    //     cout << "Could not copy to CPU" << endl;
    // }
    // for (int i = 0; i < MESH_SIZE; i++) {
    //     cout << "mesh: " << i << endl;
    //     for (int j = 0; j < state_size; j++) {
    //         cout << "state: " << j << endl;
    //         for (int k = 0; k < ACTION_SIZE; k++) {
    //             cout << qtable_cpu[i * q_table_size + j * ACTION_SIZE + k] << " ";
    //         }
    //         cout << endl;
    //     }
    // }

}

int run(double* qtable) {
    int run_step = 0;

    curandState* rand_states;
    if (cudaMalloc(&rand_states, sizeof(curandState) * MESH_SIZE) != cudaSuccess) {
        cout << "curandState malloc failed" << endl;
    }
    rand_setup_kernel<<<MESH_SIZE, 1>>>(rand_states);
    
    //cout << run_step << endl;

    int* environment;
    if (cudaMalloc(&environment, sizeof(int) * ENV_SIZE) != cudaSuccess) {
        cout << "env malloc failed" << endl;
    }
    double* reward;
    if (cudaMalloc(&reward, sizeof(double) * MESH_SIZE) != cudaSuccess) {
        cout << "reward malloc failed" << endl;
    }
    bool* is_end_state_cuda;
    bool is_end_state = false;
    if (cudaMalloc(&is_end_state_cuda, sizeof(bool)) != cudaSuccess) {
        cout << "is_end_state malloc failed" << endl;
    }
    int* environment_cpu = (int*)malloc(sizeof(int) * ENV_SIZE);
    dim3 env_block(ENV_SIZE < BLOCK_SIZE ? (ENV_SIZE > BLOCK_SQUARE_SIZE ? ENV_SIZE : BLOCK_SQUARE_SIZE) : BLOCK_SIZE);
    dim3 env_grid(ENV_SIZE / env_block.x + 1);
    dim3 mesh_block(MESH_SIZE < BLOCK_SIZE ? MESH_SIZE : BLOCK_SIZE);
    dim3 mesh_grid(MESH_SIZE / mesh_block.x + 1);
    dim3 q_block(SURROUNDING_SIZE < BLOCK_SIZE / ACTION_SIZE ? SURROUNDING_SIZE : BLOCK_SIZE / ACTION_SIZE, ACTION_SIZE);
    dim3 q_grid(SURROUNDING_SIZE / q_block.x + 1);
    reset_is_end_state_kernel<<<1, 1>>>(is_end_state_cuda);
    reset_env_kernel<<<env_grid, env_block>>>(environment, is_end_state_cuda);
    /*
    while (run_step < NUM_TEST) {
        // cout << "======================" << endl;
        // cout << "start epoch: " << epoch_step << endl;
            
        #if DEBUG
            // if (cudaMemcpy(
            //     environment_cpu, 
            //     environment, 
            //     sizeof(int) * ENV_SIZE,
            //     cudaMemcpyDeviceToHost
            // ) != cudaSuccess) {
            //     cout << "Could not copy to CPU" << endl;
            // }
            // for (int i = 0; i < MESH_SIZE; i++) {
            //     cout << "intersection: " << i << endl;
            //     for (int j = 0; j < LANE_SIZE; j++) {
            //         cout << environment_cpu[i * ENV_ITEM_SIZE + j] << " ";
            //     }
            //     cout << endl;
            // }
        #endif
        
        is_end_state_kernel<<<mesh_grid, mesh_block>>>(environment, is_end_state_cuda);
        //cudaEventRecord(start, 0);
        
        if(cudaMemcpy(
            &is_end_state, 
            is_end_state_cuda, 
            sizeof(bool),
            cudaMemcpyDeviceToHost
        ) != cudaSuccess){
            cout << "Could not copy to CPU" << endl;
        }
        if (is_end_state) {
            break;
        }
        else if (run_step > NUM_TEST) {
            break;
        }

        update_env_pre_kernel<<<mesh_grid, mesh_block>>>(environment, qtable, rand_states);
        update_env_pre_run_kernel<<<mesh_grid, mesh_block>>>(environment, qtable, rand_states, false);
        update_env_after_kernel<<<mesh_grid, mesh_block>>>(environment, qtable, rand_states);
        run_step ++;
    }
    */
    
    reset_is_end_state_kernel<<<1, 1>>>(is_end_state_cuda);
    reset_env_kernel<<<env_grid, env_block>>>(environment, is_end_state_cuda);
    while (1) {

        #if DEBUG
            // if (cudaMemcpy(
            //     environment_cpu, 
            //     environment, 
            //     sizeof(int) * ENV_SIZE,
            //     cudaMemcpyDeviceToHost
            // ) != cudaSuccess) {
            //     cout << "Could not copy to CPU" << endl;
            // }
            // for (int i = 0; i < MESH_SIZE; i++) {
            //     cout << "intersection: " << i << endl;
            //     for (int j = 0; j < LANE_SIZE; j++) {
            //         cout << environment_cpu[i * ENV_ITEM_SIZE + j] << " ";
            //     }
            //     cout << endl;
            // }
        #endif

        is_end_state_kernel<<<mesh_grid, mesh_block>>>(environment, is_end_state_cuda);
        cudaDeviceSynchronize();
        if(cudaMemcpy(
            &is_end_state, 
            is_end_state_cuda, 
            sizeof(bool),
            cudaMemcpyDeviceToHost
        ) != cudaSuccess){
            cout << "Could not copy to CPU" << endl;
        }
        if (is_end_state) {
            // cout << "test failed at step: " << run_step << endl;
            //reset_env_kernel<<<env_grid, env_block>>>(environment);
            //reset_is_end_state_kernel<<<1, 1>>>(is_end_state_cuda);
            break;
        }
        else if (run_step > NUM_TEST) {
            // cout << "test success after step: " << run_step << endl;
            break;
        }
        reset_is_end_state_kernel<<<1, 1>>>(is_end_state_cuda);
        //cout << "----------------------" << endl;
        //cout << "run step: " << run_step << endl;
        update_env_pre_run_kernel<<<mesh_grid, mesh_block>>>(environment, qtable, rand_states, false);
        update_env_after_kernel<<<mesh_grid, mesh_block>>>(environment, qtable, rand_states);
        cudaDeviceSynchronize();
        run_step ++;
    }
    cudaFree(environment);
    cudaFree(reward);
    cudaFree(is_end_state_cuda);
    cudaFree(rand_states);
    return run_step;

}

int main() {

    // IntersectionMesh mesh = IntersectionMesh();
    // mesh.train();
    // //mesh.print();
    // mesh.run();

    double* qtable;
    int state_size = ((int)pow(MAX_VEHICLE_NUM, LANE_SIZE));
    int q_table_size = state_size * ACTION_SIZE;
    if (cudaMalloc(&qtable, sizeof(double) * MESH_SIZE * q_table_size) != cudaSuccess) {
        cout << "qtable malloc failed" << endl;
    }

    train(qtable);
    
    cout << "Coverage: " << get_coverage(qtable) << endl;
    int avg_train = 0;
    int success_count = 0;
    for (int i = 0; i < NUM_TEST_ITER; i++) {
        int run_step = run(qtable);
        if (run_step == NUM_TEST + 1) {
            success_count ++;
        }
        avg_train += run_step;
    }
    cout << "Success rate: " << success_count << endl;
    cout << "Average steps: " << avg_train / NUM_TEST_ITER << endl;
    cudaFree(qtable);
    return 0;
}
