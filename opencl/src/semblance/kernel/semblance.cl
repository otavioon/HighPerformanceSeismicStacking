
#define uint32_t unsigned int
#define uint8_t unsigned char

typedef enum {
  CMP,
  CRS,
  CRP
} method_t;

typedef struct {
    float m0, h0;
    float t0, v, a, b;
} mthd_pmtr_t;

int compute_crp_mh(float h, float t0, mthd_pmtr_t prmtrs, float* mh) {

    if (!prmtrs.v) {
        return -1;
    }

    float   theta = prmtrs.t0 * prmtrs.a,
            gamma = 2 * sqrt(h * h + prmtrs.h0 * prmtrs.h0),
            tn0_sq = prmtrs.t0 * prmtrs.t0 - 4 * prmtrs.h0 * prmtrs.h0 / (prmtrs.v * prmtrs.v);

    // Auxiliar parameters
    float   theta_sq = theta * theta,
            gamma_sq = gamma * gamma,
            tn0_quad = tn0_sq * tn0_sq,
            sqrt_arg = tn0_quad * tn0_quad + tn0_quad * theta_sq * gamma_sq  +
                                16 * h * h * prmtrs.h0 * prmtrs.h0 * theta_sq * theta_sq;

    if (sqrt_arg < 0) {
        return -1;
    }

    sqrt_arg = theta_sq * gamma_sq + 2 * tn0_quad + 2 * sqrt(sqrt_arg);

    if (sqrt_arg == 0) {
        return -1;
    }

    *mh = prmtrs.m0 + 2 * theta * (h * h - prmtrs.h0 * prmtrs.h0) / sqrt(sqrt_arg);

    return 0;
}

bool use_trace(float m, float h, float t0, float apm, float aph0, mthd_pmtr_t prmtrs, method_t mthd, __private void* extr) {

    float mh = 0;
    switch (mthd) {
        case CMP:
           return true;
        case CRS:
            return true;
        case CRP:
            if (!compute_crp_mh(h, t0, prmtrs, &mh)) {
                if (extr && fabs(m - mh) <= apm) {
                    *((float *) extr) = mh;
                    return true;
                }
            }
            break;
    }

    return false;
};

bool use_midpoint(float m, float apm, float m0, method_t mthd) {

    switch (mthd) {
        case CMP:
            if (m == m0) {
                return true;
            }
            break;
        case CRS:
            if (fabs(m - m0) <= apm) {
                return true;
            }
            break;
        case CRP:
            return true;
    }

    return false;
}

float compute_time_crp(float m, float h, float t0, mthd_pmtr_t prmtrs, void *extr) {

    if (prmtrs.v != 0 && extr) {

        float tn0_sq, tn_sq, w_sqrt_1, w_sqrt_2, u, sqrt_arg, ah, th, mh;

        mh = *((float*) extr);

        tn0_sq = t0 * t0 - 4 * prmtrs.h0 * prmtrs.h0 / (prmtrs.v * prmtrs.v);

        w_sqrt_1 = (h + prmtrs.h0) * (h + prmtrs.h0) - (mh - prmtrs.m0) * (mh - prmtrs.m0);
        w_sqrt_2 = (h - prmtrs.h0) * (h - prmtrs.h0) - (mh - prmtrs.m0) * (mh - prmtrs.m0);

        if (w_sqrt_1 < 0 || w_sqrt_2 < 0) {
            return -1.0f;
        }

        u = sqrt(w_sqrt_1) + sqrt(w_sqrt_2);

        th = t0;
        if (fabs(h) > fabs(prmtrs.h0)) {

            if (!u) {
                return -1.0f;
            }

            sqrt_arg = 4 * h * h / (prmtrs.v * prmtrs.v) + 4 * h * h / (u * u) * tn0_sq;
            if (sqrt_arg < 0) {
                return -1.0f;
            }
            th = sqrt(sqrt_arg);
        }
        else if (fabs(h) < fabs(prmtrs.h0)) {

            if (!prmtrs.h0) {
                return -1.0f;
            }

            sqrt_arg = 4 * h * h / (prmtrs.v * prmtrs.v) + u * u / (4 * prmtrs.h0 * prmtrs.h0) * tn0_sq;
            if (sqrt_arg < 0) {
                return -1.0f;
            }
            th = sqrt(sqrt_arg);
        }

        tn_sq = th * th - 4 * h * h / (prmtrs.v * prmtrs.v);

        if (!th || !tn0_sq) {
            return -1.0f;
        }

        ah = (t0 * tn_sq) / (th * tn0_sq) * prmtrs.a;

        return th + ah * (m - mh);
    }

    return -1.0f;
}

__kernel
void search_for_traces_crp( __global __read_only float* midpnt,
                            __global __read_only float* hlffset,
                            __global uint8_t* used_tr,
                            float apm, float m0, float h0, uint32_t nsmpl_per_trce, float dt_s,
                            __global __read_only float *v,
                            __global __read_only float* a,
                            uint32_t prmt_c,  uint32_t offst, uint8_t shr, uint32_t ntraces) {

    uint32_t t_idx = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (t_idx < ntraces) {

        used_tr[t_idx] = 0;

        float mh;
        mthd_pmtr_t prmtrs;
        prmtrs.m0 = m0;
        prmtrs.h0 = h0;

        uint32_t t = 0;
        while (!used_tr[t_idx] && t < nsmpl_per_trce) {

            prmtrs.t0 = t * dt_s;

            uint32_t i = 0;
            while (i < prmt_c) {

                uint32_t p_idx = shr ? offst + i : offst + t * prmt_c + i;

                prmtrs.v = v[p_idx];
                prmtrs.a = a[p_idx];

                if (use_trace(midpnt[t_idx], hlffset[t_idx], prmtrs.t0, apm, 0, prmtrs, CRP, &mh)) {
                    used_tr[t_idx] = 1;
                    break;
                }

                i++;
            }
            t++;
        }
    }
}

__kernel
void search_for_traces_cmp_crs( __global __read_only float* midpnt,
                                __global __read_only float* hlffset,
                                __global uint8_t* used_tr,
                                method_t method, float apm, float m0, uint32_t ntraces) {

    uint32_t t_idx = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (t_idx < ntraces) {
        used_tr[t_idx] = use_midpoint(midpnt[t_idx], apm, m0, method) ? 1 : 0;
    }
}

float compute_time(float m, float h, float t0, mthd_pmtr_t prmtrs, method_t mthd, void* extr) {

    if (mthd == CMP) {
        if (prmtrs.v != 0) {
            return sqrt(t0 * t0 + 4 * h * h / (prmtrs.v * prmtrs.v));
        }
    }
    else if (mthd == CRS) {
        if (prmtrs.v != 0) {

            float tmp, dm;

            dm = m - prmtrs.m0;
            tmp = t0 + prmtrs.a * dm,
            tmp = tmp * tmp +  prmtrs.b * dm * dm + 4 * h * h / (prmtrs.v * prmtrs.v);
            if (tmp >= 0) {
                return sqrt(tmp);
            }
        }
    }
    else if (mthd == CRP) {
        return compute_time_crp(m, h, t0, prmtrs, extr);
    }

    return -1.0f;
}

int compute_semblance(__global const float *samples, float t, float dt_s, __private float* num, __private float* den, __private float* lin,
                            int idx_tau, int w, uint32_t nsamples) {

    float idx_t, dt;
    int k_t;

    idx_t = t / dt_s;
    k_t = (int) idx_t;
    dt = idx_t - (float) k_t;

    if ((k_t - idx_tau >= 0) && (k_t + idx_tau + 1 < nsamples)) {

        for (int j = 0; j < w; j++) {

            int k = k_t - idx_tau + j;

            float y1 = samples[k + 1],
                  y0 = samples[k],
                  u = (y1 - y0) * dt + y0;

            num[j] += u;
            *lin += u;
            *den += u * u;
        }

        return 0;
    }

    return -1;
}

__kernel
void compute_semblance_gpu(/* Data arrays */
                        __global __read_only float *samples,
                        __global __read_only float *midpoint,
                        __global __read_only float *halfoffset,
                        uint32_t tr_c,
                        /* Parameter arrays */
                        __global __read_only float *prmt_v,
                        __global __read_only float *prmt_a,
                        __global __read_only float *prmt_b,
                        /* Data boundaries */
                        uint32_t nsamples, float m0,
                        /* Data thresholds */
                        float dt_s, float aph0, float h0, float apm, int idx_tau,
                        method_t mthd,
                        /* Output arrays */
                        __global float *sembl,
                        __global float *stack,
                        __global __write_only float *max_v,
                        __global __write_only float *max_a,
                        __global __write_only float *max_b,
                        /* Missed traces array */
                        __global float *missed,
                        __local float* thrd_sembl,
                        __local float* thrd_stck,
                        __local float* trhd_prmtr_v,
                        __local float* trhd_prmtr_a,
                        __local float* trhd_prmtr_b) {

    size_t prmtr_idx = get_local_id(0);
    size_t t0_idx = get_group_id(0);
    uint32_t missed_idx = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (t0_idx < nsamples) {

        /* Copy and initialize shared memory */
        thrd_sembl[prmtr_idx] = thrd_stck[prmtr_idx] = 0;

        uint32_t missed_c = 0;

        mthd_pmtr_t prmtrs;
        prmtrs.t0 = t0_idx * dt_s;
        prmtrs.m0 = m0;
        prmtrs.h0 = h0;

        trhd_prmtr_v[prmtr_idx] = prmtrs.v = prmt_v[prmtr_idx];

        if (prmt_a) {
            trhd_prmtr_a[prmtr_idx] = prmtrs.a = prmt_a[prmtr_idx];
        }

        if (prmt_b) {
            trhd_prmtr_b[prmtr_idx] = prmtrs.b = prmt_b[prmtr_idx];
        }

        int usd_trcs = 0, bndrr_err = 0;
        int w = 2 * idx_tau + 1;

        float num[64], den = 0, lin = 0;
        for (uint32_t i = 0; i < 64; i++) {
            num[i] = 0;
        }

        for (uint32_t idx_trce = 0; idx_trce < tr_c; idx_trce++) {

            float h = halfoffset[idx_trce];
            float m = midpoint[idx_trce];

            float mh;
            if (mthd == CRP && !use_trace(m, h, prmtrs.t0, apm, 0, prmtrs, CRP, &mh)) {
                missed_c++;
                continue;
            }

            __global const float *tr_samples = &samples[idx_trce * nsamples];

            float t = compute_time(m, h, prmtrs.t0, prmtrs, mthd, &mh);

            if (!compute_semblance(tr_samples, t, dt_s, num, &den, &lin, idx_tau, w, nsamples)) {
                usd_trcs++;
            }
            else {
                bndrr_err++;
            }
        }

        if (usd_trcs) {

            float sum_num = 0;
            for (int j = 0; j < w; j++) {
                sum_num += num[j] * num[j];
            }

            thrd_sembl[prmtr_idx] = sum_num / (usd_trcs * den);
            thrd_stck[prmtr_idx] = lin / (usd_trcs * w);
        }

        missed[missed_idx] += missed_c;

        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        /* Reduce the best results */
        for (uint32_t s = get_local_size(0) / 2; s > 0; s = s >> 1) {
            if (prmtr_idx < s) {
                if (thrd_sembl[prmtr_idx] < thrd_sembl[prmtr_idx + s]) {
                    thrd_sembl[prmtr_idx] = thrd_sembl[prmtr_idx + s];
                    thrd_stck[prmtr_idx] = thrd_stck[prmtr_idx + s];
                    trhd_prmtr_v[prmtr_idx] = trhd_prmtr_v[prmtr_idx + s];
                    trhd_prmtr_a[prmtr_idx] = trhd_prmtr_a[prmtr_idx + s];
                    trhd_prmtr_b[prmtr_idx] = trhd_prmtr_b[prmtr_idx + s];
                }
            }

            work_group_barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (prmtr_idx == 0 && thrd_sembl[0] > sembl[t0_idx]) {
            sembl[t0_idx] = thrd_sembl[0];
            stack[t0_idx] = thrd_stck[0];
            max_v[t0_idx] = trhd_prmtr_v[0];

            if (max_a) {
                max_a[t0_idx] = trhd_prmtr_a[0];
            }

            if (max_b) {
                max_b[t0_idx] = trhd_prmtr_b[0];
            }
       }
    }
}

__kernel
void compute_semblance_ga_gpu( /* Data arrays */
                                __global __read_only float *samples,
                                __global __read_only float *midpoint,
                                __global __read_only float *halfoffset,
                                uint32_t tr_c,
                                /* Parameter arrays */
                                __global __read_only float *prmt_v,
                                __global __read_only float *prmt_a,
                                __global __read_only float *prmt_b,
                                /* Data boundaries */
                                uint32_t nsamples, float m0,
                                /* Data thresholds */
                                float dt_s, float aph0, float h0, float apm, int idx_tau,
                                method_t mthd,
                                /* Output arrays */
                                __global float *sembl,
                                __global float *stack,
                                /* Missed traces array */
                                __global float *missed) {

    uint32_t t0_idx = get_group_id(0);
    uint32_t prmtr_idx = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (t0_idx < nsamples) {

        sembl[prmtr_idx] = stack[prmtr_idx] = 0;

        uint32_t missed_c = 0;

        mthd_pmtr_t prmtrs;

        prmtrs.t0 = t0_idx * dt_s;
        prmtrs.m0 = m0;
        prmtrs.h0 = h0;

        prmtrs.v = prmt_v[prmtr_idx];

        if (prmt_a) {
            prmtrs.a = prmt_a[prmtr_idx];
        }

        if (prmt_b) {
            prmtrs.b = prmt_b[prmtr_idx];
        }

        int usd_trcs = 0, bndrr_err = 0;

        int w = 2 * idx_tau + 1;

        float num[64], den = 0, lin = 0;
        for (uint32_t i = 0; i < 64; i++) {
            num[i] = 0;
        }

        for (uint32_t idx_trce = 0; idx_trce < tr_c; idx_trce++) {

            float h = halfoffset[idx_trce];
            float m = midpoint[idx_trce];

            float mh;
            if (mthd == CRP && !use_trace(m, h, prmtrs.t0, apm, 0, prmtrs, CRP, &mh)) {
                missed_c++;
                continue;
            }

            __global const float *tr_samples = &samples[idx_trce * nsamples];

            float t = compute_time(m, h, prmtrs.t0, prmtrs, mthd, &mh);

            if (!compute_semblance(tr_samples, t, dt_s, num, &den, &lin, idx_tau, w, nsamples)) {
                usd_trcs++;
            }
            else {
                bndrr_err++;
            }
        }

        if (usd_trcs) {

            float sum_num = 0;
            for (int j = 0; j < w; j++) {
                sum_num += num[j] * num[j];
            }

            sembl[prmtr_idx] = sum_num / (usd_trcs * den);
            stack[prmtr_idx] = lin / (usd_trcs * w);
        }

        missed[prmtr_idx] += missed_c;
    }
}

__kernel
void compute_strech_free_sembl_gpu( /* Data arrays */
                                    __global __read_only float *samples,
                                    __global __read_only float *midpoint,
                                    __global __read_only float *halfoffset,
                                    uint32_t tr_c,
                                    /* Parameter arrays */
                                    __global __read_only float *prmt_v,
                                    __global __read_only float *prmt_a,
                                    __global __read_only float *prmt_b,
                                    /* Data boundaries */
                                    uint32_t nsamples, float m0,
                                    /* Data thresholds */
                                    float dt_s, float aph0, float apm, int idx_tau,
                                    method_t mthd,
                                    /* Output arrays and others */
                                    __global float *sembl,
                                    __global float *stack,
                                    uint32_t m0_idx,
                                    __global __read_only float *inpt_prmtrs,
                                    __global float *outpt_prmtrs,
                                    /* Missed traces array */
                                    __global float *missed,
                                    __local float* thrd_sembl,
                                    __local float* thrd_stck,
                                    __local float* thrd_n) {

    uint32_t prmtr_idx = get_local_id(0);
    uint32_t t0_idx = get_group_id(0);
    uint32_t missed_idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    uint32_t outpt_idx = m0_idx * nsamples + t0_idx;

    if (t0_idx < nsamples) {

        /* Copy and initialize shared memory */
        thrd_sembl[prmtr_idx] = thrd_stck[prmtr_idx] = 0;

        uint32_t missed_c = 0;

        int n = (int) inpt_prmtrs[prmtr_idx];
        thrd_n[prmtr_idx] = inpt_prmtrs[prmtr_idx];

        if (((int) t0_idx - n) >= 0 && ((int) t0_idx - n) < nsamples) {

            mthd_pmtr_t strech, prmtrs;

            prmtrs.t0 = t0_idx * dt_s;
            strech.t0 = (t0_idx - n) * dt_s;

            prmtrs.m0 = strech.m0 = midpoint[m0_idx];

            prmtrs.v = prmt_v[outpt_idx];
            strech.v = prmt_v[outpt_idx - n];

            if (prmt_a) {
                prmtrs.a = prmt_a[outpt_idx];
                strech.a = prmt_a[outpt_idx - n];
            }

            if (prmt_b) {
                prmtrs.b = prmt_b[outpt_idx];
                strech.b = prmt_b[outpt_idx - n];
            }

            prmtrs.h0 = strech.h0 = 0;

            int usd_trcs = 0, bndrr_err = 0;

            int w = 2 * idx_tau + 1;

            float num[64], den = 0, lin = 0;
            for (uint32_t i = 0; i < 64; i++) {
                num[i] = 0;
            }

            for (uint32_t idx_trce = 0; idx_trce < tr_c; idx_trce++) {

                float h = halfoffset[idx_trce];
                float m = midpoint[idx_trce];

                float mh;
                if (mthd == CRP && !use_trace(m, h, prmtrs.t0, apm, 0, prmtrs, CRP, &mh)) {
                    missed_c++;
                    continue;
                }

                __global const float *tr_samples = &samples[idx_trce * nsamples];

                float t = prmtrs.t0 + compute_time(m, h, strech.t0, strech, mthd, &mh) - strech.t0;

                if (!compute_semblance(tr_samples, t, dt_s, num, &den, &lin, idx_tau, w, nsamples)) {
                    usd_trcs++;
                }
                else {
                    bndrr_err++;
                }
            }

            if (usd_trcs) {

                float sum_num = 0;
                for (int j = 0; j < w; j++) {
                    sum_num += num[j] * num[j];
                }

                thrd_sembl[prmtr_idx] = sum_num / (usd_trcs * den);
                thrd_stck[prmtr_idx] = lin / (usd_trcs * w);
            }

            missed[missed_idx] += missed_c;

            work_group_barrier(CLK_LOCAL_MEM_FENCE);

            /* Reduce the best results */
            for (uint32_t s = get_local_size(0) / 2; s > 0; s = s >> 1) {
                if (prmtr_idx < s) {
                    if (thrd_sembl[prmtr_idx] < thrd_sembl[prmtr_idx + s]) {
                        thrd_sembl[prmtr_idx] = thrd_sembl[prmtr_idx + s];
                        thrd_stck[prmtr_idx] = thrd_stck[prmtr_idx + s];
                        thrd_n[prmtr_idx] = thrd_n[prmtr_idx + s];
                    }
                }
                work_group_barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (prmtr_idx == 0 && thrd_sembl[0] > sembl[outpt_idx]) {
                sembl[outpt_idx] = thrd_sembl[0];
                stack[outpt_idx] = thrd_stck[0];
                outpt_prmtrs[outpt_idx] = thrd_n[0];
            }
        }
    }
}