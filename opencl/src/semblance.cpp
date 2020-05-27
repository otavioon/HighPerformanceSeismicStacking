#include "semblance.h"
#include <CL/cl.h>
#include <vector>

SemblanceOpenClParameter::~SemblanceOpenClParameter() {

    for (auto& hlpr : dv_intr_data) {
        clReleaseMemObject(hlpr);
    }

    uint32_t t = TO_UINT(Parameter::CNT);
    for (uint32_t p = 0; p < t; p++) {
        if (is_internal[p]) {
            clReleaseMemObject(dvc_prmt[p]);
        }
    }
}

int SemblanceOpenClParameter::copy_to_device(const Gather& gather) {

    cl_int ret;

    /* Instantiate data arrays */
    std::vector<float> tmp_mdpnt(gather.traces.size());
    std::vector<float> tmp_hlffset(gather.traces.size());

    for (uint32_t i = 0; i < gather.traces.size(); i++) {
        tmp_mdpnt[i] = gather.traces[i].midpoint;
        tmp_hlffset[i] = gather.traces[i].halfoffset;
    }

    dv_intr_data[TO_UINT(Helpers::MDPNT)] =
        clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_ONLY, gather.traces.size() * sizeof(float), NULL, &ret);

    clEnqueueWriteBuffer(cl_ctxt.cmd_queue, dv_intr_data[TO_UINT(Helpers::MDPNT)], CL_TRUE,
            0, gather.traces.size() * sizeof(float), tmp_mdpnt.data(),
            0, NULL, NULL);

    dv_intr_data[TO_UINT(Helpers::HLFOFFST)] =
        clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_ONLY, gather.traces.size() * sizeof(float), NULL, &ret);

    clEnqueueWriteBuffer(cl_ctxt.cmd_queue, dv_intr_data[TO_UINT(Helpers::HLFOFFST)], CL_TRUE,
            0, gather.traces.size() * sizeof(float), tmp_hlffset.data(),
            0, NULL, NULL);

    return 0;
}

int SemblanceOpenClParameter::filter_traces(const Gather& gather, const Arguments& arguments, uint32_t c, uint32_t offst, bool share, uint32_t idx_m0) {

    std::chrono::steady_clock::time_point tm = std::chrono::steady_clock::now();

    cl_int ret;
    uint32_t p_cnt = 0;

    std::vector<uint8_t> used_tr(gather.traces.size());

    float m0 = gather.cdps[idx_m0].midpoint;

    cl_mem dv_used_tr = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, gather.traces.size() * sizeof(uint8_t), NULL, &ret);

    size_t blockDim = (TO_UINT(gather.traces.size()) / THREADS_PER_BLOCK + 1) * THREADS_PER_BLOCK;
    size_t threadDim = THREADS_PER_BLOCK;

    if (arguments.chosen_method == Method::CMP || arguments.chosen_method == Method::CRS) {

        ret = clSetKernelArg(cl_ctxt.sembl_ft_cmp_crs, p_cnt++, sizeof(cl_mem), &dv_intr_data[TO_UINT(Helpers::MDPNT)]);
        ret = clSetKernelArg(cl_ctxt.sembl_ft_cmp_crs, p_cnt++, sizeof(cl_mem), &dv_intr_data[TO_UINT(Helpers::HLFOFFST)]);
        ret = clSetKernelArg(cl_ctxt.sembl_ft_cmp_crs, p_cnt++, sizeof(cl_mem), &dv_used_tr);

        uint32_t mth = TO_UINT(arguments.chosen_method);
        ret = clSetKernelArg(cl_ctxt.sembl_ft_cmp_crs, p_cnt++, sizeof(uint32_t), &mth);
        ret = clSetKernelArg(cl_ctxt.sembl_ft_cmp_crs, p_cnt++, sizeof(float), &arguments.apm);
        ret = clSetKernelArg(cl_ctxt.sembl_ft_cmp_crs, p_cnt++, sizeof(float), &m0);

        uint32_t ntraces = TO_UINT(gather.traces.size());
        ret = clSetKernelArg(cl_ctxt.sembl_ft_cmp_crs, p_cnt++, sizeof(uint32_t), &ntraces);

        ret = clEnqueueNDRangeKernel(cl_ctxt.cmd_queue, cl_ctxt.sembl_ft_cmp_crs, 1, NULL, &blockDim, &threadDim, 0, NULL, NULL);

        if (ret != CL_SUCCESS) {
            LOGI("%d %d %d", ret, blockDim, threadDim);
        }

        ret = clFinish(cl_ctxt.cmd_queue);
        if (ret != CL_SUCCESS) {
            LOGI("clFinish %d", ret);
        }

        ret = clEnqueueReadBuffer(cl_ctxt.cmd_queue, dv_used_tr, CL_TRUE, 0, used_tr.size() * sizeof(uint8_t), used_tr.data(), 0, NULL, NULL);

        u_tr_c = std::accumulate(used_tr.begin(), used_tr.end(), 0);
    }
    else if (arguments.chosen_method == Method::CRP) {

        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(cl_mem), &dv_intr_data[TO_UINT(Helpers::MDPNT)]);
        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(cl_mem), &dv_intr_data[TO_UINT(Helpers::HLFOFFST)]);
        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(cl_mem), &dv_used_tr);

        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(float), &arguments.apm);
        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(float), &m0);
        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(float), &arguments.h0);

        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(uint32_t), &gather.nsmpl_per_trce);

        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(float), &gather.dt_s);

        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(cl_mem), &dvc_prmt[TO_UINT(Parameter::V)]);
        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(cl_mem), &dvc_prmt[TO_UINT(Parameter::A)]);

        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(uint32_t), &c);
        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(uint32_t), &offst);

        uint8_t s = static_cast<uint8_t>(share);
        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(uint8_t), &s);

        uint32_t ntraces = TO_UINT(gather.traces.size());
        ret = clSetKernelArg(cl_ctxt.sembl_ft_crp, p_cnt++, sizeof(uint32_t), &ntraces);

        ret = clEnqueueNDRangeKernel(cl_ctxt.cmd_queue, cl_ctxt.sembl_ft_crp, 1, NULL, &blockDim, &threadDim, 0, NULL, NULL);

        if (ret != CL_SUCCESS) {
            LOGI("%d %d %d", ret, blockDim, threadDim);
        }

        ret = clFinish(cl_ctxt.cmd_queue);
        if (ret != CL_SUCCESS) {
            LOGI("clFinish %d", ret);
        }

        ret = clEnqueueReadBuffer(cl_ctxt.cmd_queue, dv_used_tr, CL_TRUE, 0, used_tr.size() * sizeof(uint8_t), used_tr.data(), 0, NULL, NULL);

        u_tr_c = std::accumulate(used_tr.begin(), used_tr.end(), 0);
    }

    clReleaseMemObject(dv_used_tr);

    /* Reallocate filtered sample array */
    clReleaseMemObject(dv_intr_data[TO_UINT(Helpers::FILT_SAMPL)]);
    clReleaseMemObject(dv_intr_data[TO_UINT(Helpers::FILT_MDPNT)]);
    clReleaseMemObject(dv_intr_data[TO_UINT(Helpers::FILT_HLFOFFST)]);

    dv_intr_data[TO_UINT(Helpers::FILT_SAMPL)] = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_ONLY, u_tr_c * gather.nsmpl_per_trce * sizeof(float), NULL, &ret);
    dv_intr_data[TO_UINT(Helpers::FILT_MDPNT)] = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_ONLY, u_tr_c * sizeof(float), NULL, &ret);
    dv_intr_data[TO_UINT(Helpers::FILT_HLFOFFST)] = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_ONLY, u_tr_c * sizeof(float), NULL, &ret);

    uint32_t cuda_arr_s = 0, idx = 0;
    for (uint32_t i = 0; i < gather.traces.size(); i++) {
        if (used_tr[i]) {

            clEnqueueWriteBuffer(cl_ctxt.cmd_queue, dv_intr_data[TO_UINT(Helpers::FILT_SAMPL)], CL_TRUE,
                cuda_arr_s * sizeof(float), gather.nsmpl_per_trce * sizeof(float), gather.traces[i].samples.data(),
                0, NULL, NULL);

            clEnqueueWriteBuffer(cl_ctxt.cmd_queue, dv_intr_data[TO_UINT(Helpers::FILT_MDPNT)], CL_TRUE,
                idx * sizeof(float), sizeof(float), &gather.traces[i].midpoint, 0, NULL, NULL);

            clEnqueueWriteBuffer(cl_ctxt.cmd_queue, dv_intr_data[TO_UINT(Helpers::FILT_HLFOFFST)], CL_TRUE,
                idx * sizeof(float), sizeof(float), &gather.traces[i].halfoffset, 0, NULL, NULL);

            cuda_arr_s += gather.nsmpl_per_trce;
            idx++;
        }
    }

    std::chrono::duration<double> elpsd_time = std::chrono::steady_clock::now() - tm;

    LOGI("Using %d traces. (%.3fs to filter)", u_tr_c, elpsd_time.count());

    return 0;
}

int SemblanceOpenClParameter::allocate(Parameter prmtr, uint32_t size) {

    set_internal(prmtr, true);

    cl_int ret;
    uint32_t ZERO = 0;

    dvc_prmt[TO_UINT(prmtr)] =
        clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_ONLY, size * sizeof(float), NULL, &ret);

    clEnqueueFillBuffer(cl_ctxt.cmd_queue, dvc_prmt[TO_UINT(prmtr)], &ZERO, sizeof(uint32_t),
            0, size * sizeof(float), 0, NULL, NULL);

    return 0;
}

void SemblanceOpenClParameter::free_internal(Parameter prmtr) {
    uint32_t p = TO_UINT(prmtr);
    clReleaseMemObject(dvc_prmt[p]);
    dvc_prmt[p] = NULL;
}

void SemblanceOpenClParameter::send_to_device(const std::vector<float>& data, Parameter prmtr) {

    if (!dvc_prmt[TO_UINT(prmtr)]) {
        allocate(prmtr, static_cast<uint32_t>(data.size()));
    }

    cl_int ret = clEnqueueWriteBuffer(cl_ctxt.cmd_queue, dvc_prmt[TO_UINT(prmtr)], CL_TRUE,
            0, data.size() * sizeof(float), data.data(), 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        LOGE("clEnqueueWriteBuffer failed with error %d", ret);
    }
}

void SemblanceOpenClParameter::send_to_device(const std::vector<float>& data, uint32_t start, Parameter prmtr) {

    clEnqueueWriteBuffer(cl_ctxt.cmd_queue, dvc_prmt[TO_UINT(prmtr)], CL_TRUE,
            start * sizeof(float), data.size() * sizeof(float), data.data(),
            0, NULL, NULL);
}

SemblanceOpenClResult::~SemblanceOpenClResult() {
    /* Free result arrays */
    uint32_t t = TO_UINT(Result::CNT);
    for (uint32_t r = 0; r < t; r++) {
        if (is_internal[r]) {
            clReleaseMemObject(dvc_rslt[r]);
        }
    }
}

int SemblanceOpenClResult::allocate(Result rslt, uint32_t size) {

    set_internal(rslt, true);

    cl_int ret;

    dvc_rslt[TO_UINT(rslt)] =
        clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, size * sizeof(float), NULL, &ret);

    reset(rslt, size);

    return 0;
}

void SemblanceOpenClResult::reset(Result rslt, uint32_t size) {

    uint32_t ZERO = 0;

    clEnqueueFillBuffer(cl_ctxt.cmd_queue, dvc_rslt[TO_UINT(rslt)], &ZERO, sizeof(uint32_t),
            0, size * sizeof(float), 0, NULL, NULL);
}

void SemblanceOpenClResult::free_internal(Result result) {
    uint32_t r = TO_UINT(result);
    clReleaseMemObject(dvc_rslt[r]);
}

void SemblanceOpenClResult::get_from_device(std::vector<float>& data, Result rslt) const {

    clEnqueueReadBuffer(cl_ctxt.cmd_queue, dvc_rslt[TO_UINT(rslt)], CL_TRUE, 0,
            data.size() * sizeof(float), data.data(), 0, NULL, NULL);
}

void SemblanceOpenCl::compute(uint32_t idx_m0, float h0, uint32_t thrd_cnt,
                        const SemblanceOpenClParameter& prmtrs,
                        SemblanceOpenClResult& rslt, MethodStrategy st) const {
    cl_int ret;
    cl_kernel krnl;

    uint32_t p_cnt = 0;

    const Arguments& arguments = in.arguments;
    const Gather& gather = in.gather;

    float m0 = gather.cdps[idx_m0].midpoint;

    switch (st) {
        case MethodStrategy::GA:
            krnl = cl_ctxt.sembl_ga_kernel;
            break;
        case MethodStrategy::EXHAUSTIVE:
            krnl = cl_ctxt.sembl_kernel;
            break;
        case MethodStrategy::STACK:
            krnl = cl_ctxt.stack_kernel;
            break;
    }

    // Set the arguments of the kernel
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &prmtrs.dv_intr_data[TO_UINT(Helpers::FILT_SAMPL)]);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &prmtrs.dv_intr_data[TO_UINT(Helpers::FILT_MDPNT)]);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &prmtrs.dv_intr_data[TO_UINT(Helpers::FILT_HLFOFFST)]);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(uint32_t), &prmtrs.u_tr_c);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &prmtrs.dvc_prmt[TO_UINT(Parameter::V)]);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &prmtrs.dvc_prmt[TO_UINT(Parameter::A)]);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &prmtrs.dvc_prmt[TO_UINT(Parameter::B)]);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(uint32_t), &gather.nsmpl_per_trce);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(float), &m0);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(float), &gather.dt_s);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(float), &arguments.aph0);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(float), &arguments.h0);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(float), &arguments.apm);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(float), &in.idx_tau);

    uint32_t mth = TO_UINT(arguments.chosen_method);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(uint32_t), &mth);

    ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &rslt.dvc_rslt[TO_UINT(Result::SEMBL)]);
    ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &rslt.dvc_rslt[TO_UINT(Result::STACK)]);

    if (st == MethodStrategy::EXHAUSTIVE) {

        ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &rslt.dvc_rslt[TO_UINT(Result::V)]);
        ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &rslt.dvc_rslt[TO_UINT(Result::A)]);
        ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &rslt.dvc_rslt[TO_UINT(Result::B)]);
        ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &rslt.dvc_rslt[TO_UINT(Result::MISSED)]);

        ret = clSetKernelArg(krnl, p_cnt++, thrd_cnt * sizeof(float), NULL);
        ret = clSetKernelArg(krnl, p_cnt++, thrd_cnt * sizeof(float), NULL);
        ret = clSetKernelArg(krnl, p_cnt++, thrd_cnt * sizeof(float), NULL);
        ret = clSetKernelArg(krnl, p_cnt++, thrd_cnt * sizeof(float), NULL);
        ret = clSetKernelArg(krnl, p_cnt++, thrd_cnt * sizeof(float), NULL);
    }
    else if (st == MethodStrategy::GA) {
        ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &rslt.dvc_rslt[TO_UINT(Result::MISSED)]);
    }
    else if (st == MethodStrategy::STACK) {

        ret = clSetKernelArg(krnl, p_cnt++, sizeof(float), &idx_m0);

        ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &prmtrs.dvc_prmt[TO_UINT(Parameter::N)]);
        ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &rslt.dvc_rslt[TO_UINT(Result::N)]);
        ret = clSetKernelArg(krnl, p_cnt++, sizeof(cl_mem), &rslt.dvc_rslt[TO_UINT(Result::MISSED)]);

        ret = clSetKernelArg(krnl, p_cnt++, thrd_cnt * sizeof(float), NULL);
        ret = clSetKernelArg(krnl, p_cnt++, thrd_cnt * sizeof(float), NULL);
        ret = clSetKernelArg(krnl, p_cnt++, thrd_cnt * sizeof(float), NULL);
    }

    size_t blockDim = gather.nsmpl_per_trce * thrd_cnt;
    size_t threadDim = thrd_cnt;

    ret = clEnqueueNDRangeKernel(cl_ctxt.cmd_queue, krnl, 1, NULL, &blockDim, &threadDim, 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        LOGI("%d %d %d", ret, blockDim, threadDim);
    }

    ret = clFinish(cl_ctxt.cmd_queue);
    if (ret != CL_SUCCESS) {
        LOGI("clFinish %d", ret);
    }
}
