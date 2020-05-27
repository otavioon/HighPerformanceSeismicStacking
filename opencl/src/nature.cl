#define F_FAC 0.85f
#define CR 0.5f

typedef unsigned int uint32_t;
typedef enum { V, A, B, N } nature_prmtr_t;
typedef enum { SEMBL, STACK } nature_rslt_t;

float random_next(__global uint32_t* seed) {

    // Lehmer random number generator
    // https://en.wikipedia.org/wiki/Lehmer_random_number_generator
    *seed = ((*seed) * 16807 ) % 2147483647;
    return  (float)(*seed) * 4.6566129e-10;
}

__kernel
void start_all_gpu( __global float* x_v,
                    __global float* x_a,
                    __global float* x_b,
                    __global __read_only float* min,
                    __global __read_only float* max,
                    __global uint32_t* st) {

    uint32_t offset = get_group_id(0) * get_local_size(0);
    uint32_t i = offset + get_local_id(0);

    float p = random_next(&st[i]);

    uint32_t prmtr = V;
    x_v[i] = min[prmtr] + p * (max[prmtr] - min[prmtr]);

    if (x_a) {
        p = random_next(&st[i]);
        prmtr = A;
        x_a[i] = min[prmtr] + p * (max[prmtr] - min[prmtr]);
    }

    if (x_b) {
        p = random_next(&st[i]);
        prmtr = B;
        x_b[i] = min[prmtr] + p * (max[prmtr] - min[prmtr]);
    }
}

__kernel
void mutate_all_gpu(__global float* v_v,
                    __global float* v_a,
                    __global float* v_b,
                    __global const float* x_v,
                    __global const float* x_a,
                    __global const float* x_b,
                    __global __read_only float* min,
                    __global __read_only float* max,
                    __global uint32_t* st) {

    uint32_t offset = get_group_id(0) * get_local_size(0);
    uint32_t i = offset + get_local_id(0);

    float   p1 = random_next(&st[i]),
            p2 = random_next(&st[i]),
            p3 = random_next(&st[i]);

    uint32_t    r1 = offset + (uint32_t)(p1 * (float)(get_local_size(0) - 1)),
                r2 = offset + (uint32_t)(p2 * (float)(get_local_size(0) - 1)),
                r3 = offset + (uint32_t)(p3 * (float)(get_local_size(0) - 1));

    v_v[i] = x_v[r1] + F_FAC * (x_v[r2] - x_v[r3]);

    if (v_v[i] > max[V]) {
        v_v[i] = max[V];
    }

    if (v_v[i] < min[V]) {
        v_v[i] = min[V];
    }

    if (v_a) {
        v_a[i] = x_a[r1] + F_FAC * (x_a[r2] - x_a[r3]);

        if (v_a[i] > max[A]) {
            v_a[i] = max[A];
        }

        if (v_a[i] < min[A]) {
            v_a[i] = min[A];
        }
    }

    if (v_b) {
        v_b[i] = x_b[r1] + F_FAC * (x_b[r2] - x_b[r3]);

        if (v_b[i] > max[B]) {
            v_b[i] = max[B];
        }

        if (v_b[i] < min[B]) {
            v_b[i] = min[B];
        };
    }
}

__kernel
void crossover_all_gpu( __global float* u_v,
                        __global float* u_a,
                        __global float* u_b,
                        __global const float* x_v,
                        __global const float* x_a,
                        __global const float* x_b,
                        __global const float* v_v,
                        __global const float* v_a,
                        __global const float* v_b,
                        uint32_t d,
                        __global uint32_t* st) {

    uint32_t offset = get_group_id(0) * get_local_size(0);
    uint32_t i = offset + get_local_id(0);

    uint32_t l = (uint32_t)(random_next(&st[i]) * (float)(d - 1));
    float r = random_next(&st[i]);

    if (r > CR && l != V) {
        u_v[i] = x_v[i];
    }
    else {
        u_v[i] = v_v[i];
    }

    if (u_a) {
        r = random_next(&st[i]);

        if (r > CR && l != A) {
            u_a[i] = x_a[i];
        }
        else {
            u_a[i] = v_a[i];
        }
    }

    if (u_b) {
        r = random_next(&st[i]);

        if (r > CR && l != B) {
            u_b[i] = x_b[i];
        }
        else {
            u_b[i] = v_b[i];
        }
    }
}

__kernel
void select_all_gpu(__global float* x_v,
                    __global float* x_a,
                    __global float* x_b,
                    __global float* f_x_sembl,
                    __global float* f_x_stack,
                    __global const float* u_v,
                    __global const float* u_a,
                    __global const float* u_b,
                    __global const float* f_u_sembl,
                    __global const float* f_u_stack) {

    uint32_t offset = get_group_id(0) * get_local_size(0);
    uint32_t i = offset + get_local_id(0);

    if (f_u_sembl[i] > f_x_sembl[i]) {
        x_v[i] = u_v[i];
        f_x_sembl[i] = f_u_sembl[i];
        f_x_stack[i] = f_u_stack[i];

        if (x_a) {
            x_a[i] = u_a[i];
        }

        if (x_b) {
            x_b[i] = u_b[i];
        }
    }
}

__kernel
void best_all_gpu(  __global const float* x_v,
                    __global const float* x_a,
                    __global const float* x_b,
                    __global const float* f_x_sembl,
                    __global const float* f_x_stack,
                    __global float* rslt_sembl,
                    __global float* rslt_stack,
                    __global float* rslt_v,
                    __global float* rslt_a,
                    __global float* rslt_b,
                    __local float* tmp_sembl,
                    __local float* tmp_stack,
                    __local float* tmp_v,
                    __local float* tmp_a,
                    __local float* tmp_b) {

    uint32_t offset = get_group_id(0) * get_local_size(0);
    uint32_t i = offset + get_local_id(0);
    uint32_t t = get_local_id(0);

    tmp_sembl[t] = f_x_sembl[i];
    tmp_stack[t] = f_x_stack[i];
    tmp_v[t] = x_v[i];

    if (x_a) {
        tmp_a[t] = x_a[i];
    }

    if (x_b) {
        tmp_b[t] = x_b[i];
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    /* Reduce the best results */
    for (uint32_t s = get_local_size(0) / 2; s > 0; s = s >> 1) {
        if (t < s) {
            if (tmp_sembl[t] < tmp_sembl[t + s]) {
                tmp_sembl[t] = tmp_sembl[t + s];
                tmp_stack[t] = tmp_stack[t + s];
                tmp_v[t] = tmp_v[t + s];
                tmp_a[t] = tmp_a[t + s];
                tmp_b[t] = tmp_b[t + s];
            }
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (t == 0) {
        rslt_sembl[get_group_id(0)] = tmp_sembl[0];
        rslt_stack[get_group_id(0)] = tmp_stack[0];
        rslt_v[get_group_id(0)] = tmp_v[0];

        if (rslt_a) {
            rslt_a[get_group_id(0)] = tmp_a[0];
        }

        if (rslt_b) {
            rslt_b[get_group_id(0)] = tmp_b[0];
        }
    }
}