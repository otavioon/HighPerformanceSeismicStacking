#ifndef NATURE_H
#define NATURE_H
#include "context.h"
#include "natr_common.h"
#include <CL/cl.h>
#include <vector>
#include <stdint.h>
#include <string>

class NatureGpuOpenCl : public NatureBase<cl_mem> {

    private:
        OpenCLDeviceContext& cl_ctxt;
        cl_mem st;
        cl_kernel strt_kernel, repr_kernel, mt_kernel, slct_kernel, bst_kernel, crs_kernel;
        std::string kl_fldr;
    public:

        NatureGpuOpenCl(uint32_t g, uint32_t p_size, uint32_t p_cnt, OpenCLDeviceContext& c, const std::string& f);

        ~NatureGpuOpenCl();

        void allocate(Parameter prmtr, float min, float max);
        void allocate(Result rslt);

        void start();
        void crossover(Method m);
        void mutate();
        void select();
        void best(  std::vector<float>& rslt_sembl, std::vector<float>& rslt_stack,
                    std::vector<float>& rslt_v, std::vector<float>& rslt_a,
                    std::vector<float>& rslt_b);
};
#endif