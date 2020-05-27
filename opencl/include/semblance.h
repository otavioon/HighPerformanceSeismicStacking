#ifndef SEMBLANCE_H
#define SEMBLANCE_H
#include "context.h"
#include "parser.h"
#include "sembl_common.h"
#include "util.h"
#include <CL/cl.h>
#include <vector>

/* Specific API for CUDA */
#define THREADS_PER_BLOCK 1024

class SemblanceOpenClParameter : public SemblanceParameterSet<cl_mem> {

    private:
        OpenCLDeviceContext& cl_ctxt;
        cl_program prgm;
        cl_kernel sembl_ft_cmp_crs, sembl_ft_crp;

    public:
        SemblanceOpenClParameter(OpenCLDeviceContext& c) :
            SemblanceParameterSet(), cl_ctxt(c) {};

        ~SemblanceOpenClParameter();

        int allocate(Parameter prmtr, uint32_t size);
        int copy_to_device(const Gather& gather);
        int filter_traces(const Gather& gather, const Arguments& arguments, uint32_t c, uint32_t offst, bool share, uint32_t idx_m0);
        void free_internal(Parameter prmtr);
        void send_to_device(const std::vector<float>& data, Parameter prmtr);
        void send_to_device(const std::vector<float>& data, uint32_t start, Parameter prmtr);
};

class SemblanceOpenClResult : public SemblanceResultSet<cl_mem> {

    private:
        OpenCLDeviceContext& cl_ctxt;

    public:
        SemblanceOpenClResult(OpenCLDeviceContext& c) :
             SemblanceResultSet(), cl_ctxt(c) {};

        ~SemblanceOpenClResult();

        int allocate(Result result, uint32_t size);
        void free_internal(Result rslt);
        void get_from_device(std::vector<float>& data, Result result) const;
        void reset(Result result, uint32_t size);
};

class SemblanceOpenCl : public Semblance<SemblanceOpenClParameter, SemblanceOpenClResult> {

    private:
        OpenCLDeviceContext& cl_ctxt;

    public:

        SemblanceOpenCl(const InputData& i, uint32_t t, OpenCLDeviceContext& c) : Semblance(i, t), cl_ctxt(c) {};
        SemblanceOpenCl(const InputData& i, OpenCLDeviceContext& c) : SemblanceOpenCl(i, 0, c) {};

        void compute(uint32_t idx_m0, float h0, uint32_t thrd_cnt,
                const SemblanceOpenClParameter& prmtrs,
                SemblanceOpenClResult& rslt, MethodStrategy st) const;
};

#endif
