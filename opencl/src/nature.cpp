#include "log.h"
#include "nature.h"
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <vector>

NatureGpuOpenCl::NatureGpuOpenCl(uint32_t g, uint32_t p_size, uint32_t p_cnt, OpenCLDeviceContext& c, const std::string& f) :
    NatureBase(g, p_size, p_cnt), cl_ctxt(c), kl_fldr(f) {

    cl_int ret;

    FILE *fp = fopen(kl_fldr.append("/nature.cl").c_str(), "r");

    uint32_t max_sz = 2 * 1024 * 1024;

    char *source = (char*) calloc(max_sz, sizeof(char));
    size_t source_sz = fread(source, sizeof(char), max_sz, fp);
    fclose(fp);

    cl_program prgm = clCreateProgramWithSource(cl_ctxt.ctx, 1, const_cast<const char**>(&source), &source_sz, &ret);

    ret = clBuildProgram(prgm, 1, &cl_ctxt.device_id, "-I../common/include/ -D_CL_KERNEL_ -cl-fast-relaxed-math", NULL, NULL);

    if (ret != CL_SUCCESS) {

        size_t len = 0;

        clGetProgramBuildInfo(prgm, cl_ctxt.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

        char *buffer = (char*) calloc(len, sizeof(char));

        ret = clGetProgramBuildInfo(prgm, cl_ctxt.device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);

        LOGI("%s", buffer);

        free(buffer);

        throw "OpenCL program failed to build!";
    }

    free(source);

    strt_kernel = clCreateKernel(prgm, "start_all_gpu", &ret);
    repr_kernel = clCreateKernel(prgm, "reproduce_all_gpu", &ret);
    mt_kernel = clCreateKernel(prgm, "mutate_all_gpu", &ret);
    slct_kernel = clCreateKernel(prgm, "select_all_gpu", &ret);
    bst_kernel = clCreateKernel(prgm, "best_all_gpu", &ret);
    crs_kernel = clCreateKernel(prgm, "crossover_all_gpu", &ret);

    if(ret != CL_SUCCESS){
        LOGI("clCreateKernel failed");
        throw "OpenCL clCreateKernel failed!";
    }

    srand(TO_UINT(time(NULL)));

    st = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, p_cnt * pop_size * sizeof(uint32_t), NULL, &ret);
    std::vector<uint32_t> tmp(p_cnt * pop_size);

    /* Initialize seeds */
    std::generate(tmp.begin(), tmp.end(), []() {
        return rand();
    });

    clEnqueueWriteBuffer(cl_ctxt.cmd_queue, st, CL_TRUE,
            0, p_cnt * pop_size * sizeof(uint32_t), tmp.data(),
            0, NULL, NULL);

    min = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_ONLY, TO_UINT(Parameter::CNT) * sizeof(float), NULL, &ret);
    max = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_ONLY, TO_UINT(Parameter::CNT) * sizeof(float), NULL, &ret);
}

void NatureGpuOpenCl::allocate(Parameter prmtr, float mn, float mx) {

    uint32_t prmtr_idx = TO_UINT(prmtr);

    cl_int ret;

    x[prmtr_idx] = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, pop_size * pop_count * sizeof(float), NULL, &ret);
    u[prmtr_idx] = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, pop_size * pop_count * sizeof(float), NULL, &ret);
    v[prmtr_idx] = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, pop_size * pop_count * sizeof(float), NULL, &ret);

    ret = clEnqueueWriteBuffer(cl_ctxt.cmd_queue, min, CL_TRUE, prmtr_idx * sizeof(float), sizeof(float), &mn, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(cl_ctxt.cmd_queue, max, CL_TRUE, prmtr_idx * sizeof(float), sizeof(float), &mx, 0, NULL, NULL);
}

void NatureGpuOpenCl::allocate(Result rslt) {

    cl_int ret;

    uint32_t rslt_idx = TO_UINT(rslt);
    f_x[rslt_idx] = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, pop_size * pop_count * sizeof(float), NULL, &ret);
    f_u[rslt_idx] = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, pop_size * pop_count * sizeof(float), NULL, &ret);

    float ZERO = 0;
    clEnqueueFillBuffer(cl_ctxt.cmd_queue, f_x[rslt_idx], &ZERO, sizeof(float), 0, pop_size * pop_count * sizeof(float), 0, NULL, NULL);
    clEnqueueFillBuffer(cl_ctxt.cmd_queue, f_u[rslt_idx], &ZERO, sizeof(float), 0, pop_size * pop_count * sizeof(float), 0, NULL, NULL);
}

void NatureGpuOpenCl::start() {

    cl_int ret;

    // Set the arguments of the kernel
    ret = clSetKernelArg(strt_kernel, 0, sizeof(cl_mem), &x[TO_UINT(Parameter::V)]);
    ret = clSetKernelArg(strt_kernel, 1, sizeof(cl_mem), &x[TO_UINT(Parameter::A)]);
    ret = clSetKernelArg(strt_kernel, 2, sizeof(cl_mem), &x[TO_UINT(Parameter::B)]);
    ret = clSetKernelArg(strt_kernel, 3, sizeof(cl_mem), &min);
    ret = clSetKernelArg(strt_kernel, 4, sizeof(cl_mem), &max);
    ret = clSetKernelArg(strt_kernel, 5, sizeof(cl_mem), &st);

    size_t blockDim = pop_count * pop_size;
    size_t threadDim = pop_size;

    ret = clEnqueueNDRangeKernel(cl_ctxt.cmd_queue, strt_kernel, 1, NULL, &blockDim, &threadDim, 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        LOGI("%d %d %d", ret, blockDim, threadDim);
    }

    ret = clFinish(cl_ctxt.cmd_queue);
    if (ret != CL_SUCCESS) {
        LOGI("clFinish %d", ret);
    }

    const float ZERO = 0;
    clEnqueueFillBuffer(cl_ctxt.cmd_queue, f_x[TO_UINT(Result::SEMBL)], &ZERO, sizeof(float), 0, pop_size * pop_count * sizeof(float), 0, NULL, NULL);
    clEnqueueFillBuffer(cl_ctxt.cmd_queue, f_x[TO_UINT(Result::STACK)], &ZERO, sizeof(float), 0, pop_size * pop_count * sizeof(float), 0, NULL, NULL);
}

void NatureGpuOpenCl::mutate() {

    cl_int ret;

    // Set the arguments of the kernel
    ret = clSetKernelArg(mt_kernel, 0, sizeof(cl_mem), &v[TO_UINT(Parameter::V)]);
    ret = clSetKernelArg(mt_kernel, 1, sizeof(cl_mem), &v[TO_UINT(Parameter::A)]);
    ret = clSetKernelArg(mt_kernel, 2, sizeof(cl_mem), &v[TO_UINT(Parameter::B)]);
    ret = clSetKernelArg(mt_kernel, 3, sizeof(cl_mem), &x[TO_UINT(Parameter::V)]);
    ret = clSetKernelArg(mt_kernel, 4, sizeof(cl_mem), &x[TO_UINT(Parameter::A)]);
    ret = clSetKernelArg(mt_kernel, 5, sizeof(cl_mem), &x[TO_UINT(Parameter::B)]);
    ret = clSetKernelArg(mt_kernel, 6, sizeof(cl_mem), &min);
    ret = clSetKernelArg(mt_kernel, 7, sizeof(cl_mem), &max);
    ret = clSetKernelArg(mt_kernel, 8, sizeof(cl_mem), &st);

    size_t blockDim = pop_count * pop_size;
    size_t threadDim = pop_size;

    ret = clEnqueueNDRangeKernel(cl_ctxt.cmd_queue, mt_kernel, 1, NULL, &blockDim, &threadDim, 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        LOGI("%d %d %d", ret, blockDim, threadDim);
    }

    ret = clFinish(cl_ctxt.cmd_queue);
    if (ret != CL_SUCCESS) {
        LOGI("clFinish %d", ret);
    }
}

void NatureGpuOpenCl::crossover(Method m) {

    uint32_t d;

    switch (m) {
        case Method::CMP:
            d = 1;
            break;
        case Method::CRS:
            d = 3;
            break;
        case Method::CRP:
            d = 2;
            break;
    }

    cl_int ret;

    // Set the arguments of the kernel
    ret = clSetKernelArg(crs_kernel, 0, sizeof(cl_mem), &u[TO_UINT(Parameter::V)]);
    ret = clSetKernelArg(crs_kernel, 1, sizeof(cl_mem), &u[TO_UINT(Parameter::A)]);
    ret = clSetKernelArg(crs_kernel, 2, sizeof(cl_mem), &u[TO_UINT(Parameter::B)]);
    ret = clSetKernelArg(crs_kernel, 3, sizeof(cl_mem), &x[TO_UINT(Parameter::V)]);
    ret = clSetKernelArg(crs_kernel, 4, sizeof(cl_mem), &x[TO_UINT(Parameter::A)]);
    ret = clSetKernelArg(crs_kernel, 5, sizeof(cl_mem), &x[TO_UINT(Parameter::B)]);
    ret = clSetKernelArg(crs_kernel, 6, sizeof(cl_mem), &v[TO_UINT(Parameter::V)]);
    ret = clSetKernelArg(crs_kernel, 7, sizeof(cl_mem), &v[TO_UINT(Parameter::A)]);
    ret = clSetKernelArg(crs_kernel, 8, sizeof(cl_mem), &v[TO_UINT(Parameter::B)]);
    ret = clSetKernelArg(crs_kernel, 9, sizeof(uint32_t), &d);
    ret = clSetKernelArg(crs_kernel, 10, sizeof(cl_mem), &st);

    size_t blockDim = pop_count * pop_size;
    size_t threadDim = pop_size;

    ret = clEnqueueNDRangeKernel(cl_ctxt.cmd_queue, crs_kernel, 1, NULL, &blockDim, &threadDim, 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        LOGI("%d %d %d", ret, blockDim, threadDim);
    }

    ret = clFinish(cl_ctxt.cmd_queue);
    if (ret != CL_SUCCESS) {
        LOGI("clFinish %d", ret);
    }
}

void NatureGpuOpenCl::select() {

    cl_int ret;

    // Set the arguments of the kernel
    ret = clSetKernelArg(slct_kernel, 0, sizeof(cl_mem), &x[TO_UINT(Parameter::V)]);
    ret = clSetKernelArg(slct_kernel, 1, sizeof(cl_mem), &x[TO_UINT(Parameter::A)]);
    ret = clSetKernelArg(slct_kernel, 2, sizeof(cl_mem), &x[TO_UINT(Parameter::B)]);
    ret = clSetKernelArg(slct_kernel, 3, sizeof(cl_mem), &f_x[TO_UINT(Result::SEMBL)]);
    ret = clSetKernelArg(slct_kernel, 4, sizeof(cl_mem), &f_x[TO_UINT(Result::STACK)]);
    ret = clSetKernelArg(slct_kernel, 5, sizeof(cl_mem), &u[TO_UINT(Parameter::V)]);
    ret = clSetKernelArg(slct_kernel, 6, sizeof(cl_mem), &u[TO_UINT(Parameter::A)]);
    ret = clSetKernelArg(slct_kernel, 7, sizeof(cl_mem), &u[TO_UINT(Parameter::B)]);
    ret = clSetKernelArg(slct_kernel, 8, sizeof(cl_mem), &f_u[TO_UINT(Result::SEMBL)]);
    ret = clSetKernelArg(slct_kernel, 9, sizeof(cl_mem), &f_u[TO_UINT(Result::STACK)]);

    size_t blockDim = pop_count * pop_size;
    size_t threadDim = pop_size;

    ret = clEnqueueNDRangeKernel(cl_ctxt.cmd_queue, slct_kernel, 1, NULL, &blockDim, &threadDim, 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        LOGI("%d %d %d", ret, blockDim, threadDim);
    }

    ret = clFinish(cl_ctxt.cmd_queue);
    if (ret != CL_SUCCESS) {
        LOGI("clFinish %d", ret);
    }
}

void NatureGpuOpenCl::best(std::vector<float>& rslt_sembl, std::vector<float>& rslt_stack,
                           std::vector<float>& rslt_v, std::vector<float>& rslt_a,
                           std::vector<float>& rslt_b) {

    cl_int ret;
    cl_mem dvc_rslt_sembl, dvc_rslt_stack, dvc_rslt_v, dvc_rslt_a = NULL, dvc_rslt_b = NULL;

    dvc_rslt_sembl = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, pop_count * sizeof(float), NULL, &ret);
    dvc_rslt_stack = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, pop_count * sizeof(float), NULL, &ret);
    dvc_rslt_v = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, pop_count * sizeof(float), NULL, &ret);

    if (!rslt_a.empty()) {
        dvc_rslt_a = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, pop_count * sizeof(float), NULL, &ret);
    }

    if (!rslt_b.empty()) {
        dvc_rslt_b = clCreateBuffer(cl_ctxt.ctx, CL_MEM_READ_WRITE, pop_count * sizeof(float), NULL, &ret);
    }

    // Set the arguments of the kernel
    ret = clSetKernelArg(bst_kernel, 0, sizeof(cl_mem), &x[TO_UINT(Parameter::V)]);
    ret = clSetKernelArg(bst_kernel, 1, sizeof(cl_mem), &x[TO_UINT(Parameter::A)]);
    ret = clSetKernelArg(bst_kernel, 2, sizeof(cl_mem), &x[TO_UINT(Parameter::B)]);
    ret = clSetKernelArg(bst_kernel, 3, sizeof(cl_mem), &f_x[TO_UINT(Result::SEMBL)]);
    ret = clSetKernelArg(bst_kernel, 4, sizeof(cl_mem), &f_x[TO_UINT(Result::STACK)]);
    ret = clSetKernelArg(bst_kernel, 5, sizeof(cl_mem), &dvc_rslt_sembl);
    ret = clSetKernelArg(bst_kernel, 6, sizeof(cl_mem), &dvc_rslt_stack);
    ret = clSetKernelArg(bst_kernel, 7, sizeof(cl_mem), &dvc_rslt_v);
    ret = clSetKernelArg(bst_kernel, 8, sizeof(cl_mem), &dvc_rslt_a);
    ret = clSetKernelArg(bst_kernel, 9, sizeof(cl_mem), &dvc_rslt_b);
    ret = clSetKernelArg(bst_kernel, 10, pop_size * sizeof(float), NULL);
    ret = clSetKernelArg(bst_kernel, 11, pop_size * sizeof(float), NULL);
    ret = clSetKernelArg(bst_kernel, 12, pop_size * sizeof(float), NULL);
    ret = clSetKernelArg(bst_kernel, 13, pop_size * sizeof(float), NULL);
    ret = clSetKernelArg(bst_kernel, 14, pop_size * sizeof(float), NULL);

    size_t blockDim = pop_count * pop_size;
    size_t threadDim = pop_size;

    ret = clEnqueueNDRangeKernel(cl_ctxt.cmd_queue, bst_kernel, 1, NULL, &blockDim, &threadDim, 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        LOGI("clEnqueueNDRangeKernel %d", ret);
    }

    ret = clFinish(cl_ctxt.cmd_queue);
    if (ret != CL_SUCCESS) {
        LOGI("clFinish %d", ret);
    }

    clEnqueueReadBuffer(cl_ctxt.cmd_queue, dvc_rslt_sembl, CL_TRUE, 0, pop_count * sizeof(float), rslt_sembl.data(), 0, NULL, NULL);
    clEnqueueReadBuffer(cl_ctxt.cmd_queue, dvc_rslt_stack, CL_TRUE, 0, pop_count * sizeof(float), rslt_stack.data(), 0, NULL, NULL);
    clEnqueueReadBuffer(cl_ctxt.cmd_queue, dvc_rslt_v, CL_TRUE, 0, pop_count * sizeof(float), rslt_v.data(), 0, NULL, NULL);

    clReleaseMemObject(dvc_rslt_sembl);
    clReleaseMemObject(dvc_rslt_stack);
    clReleaseMemObject(dvc_rslt_v);

    if (!rslt_a.empty()) {
        clEnqueueReadBuffer(cl_ctxt.cmd_queue, dvc_rslt_a, CL_TRUE, 0, pop_count * sizeof(float), rslt_a.data(), 0, NULL, NULL);
        clReleaseMemObject(dvc_rslt_a);
    }

    if (!rslt_b.empty()) {
        clEnqueueReadBuffer(cl_ctxt.cmd_queue, dvc_rslt_b, CL_TRUE, 0, pop_count * sizeof(float), rslt_b.data(), 0, NULL, NULL);
        clReleaseMemObject(dvc_rslt_b);
    }
}


NatureGpuOpenCl::~NatureGpuOpenCl() {

    clReleaseMemObject(min);
    clReleaseMemObject(max);

    for (uint32_t i = 0; i < TO_UINT(Parameter::CNT); i++) {
        if (x[i]) {
            clReleaseMemObject(x[i]);
            clReleaseMemObject(u[i]);
            clReleaseMemObject(v[i]);
        }
    }

    for (uint32_t i = 0; i < TO_UINT(Result::CNT); i++) {
        if (f_x[i]) {
            clReleaseMemObject(f_x[i]);
            clReleaseMemObject(f_u[i]);
        }
    }
}