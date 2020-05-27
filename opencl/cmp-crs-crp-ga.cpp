#include "log.h"
#include "nature.h"
#include "sembl_common.h"
#include "semblance.h"
#include "threading.h"
#include <chrono>
#include <CL/cl.h>
#include <functional>
#include <vector>

void wkr_thread(const InputData& i, SemblanceHostResult& h, SharedQueue& q, cl_platform_id p_id, cl_device_id dv_id) {

    OpenCLDeviceContext ctx(p_id, dv_id);
    NatureGpuOpenCl nat(i.arguments.gen, i.arguments.pop_size, i.gather.nsmpl_per_trce, ctx, i.arguments.opencl_src);
    SemblanceOpenCl sembl(i, ctx);
    SemblanceOpenClParameter dv_prmtr(ctx);
    SemblanceOpenClResult dv_rslt(ctx);
    SemblanceGa<NatureGpuOpenCl, SemblanceOpenCl, SemblanceOpenClParameter, SemblanceOpenClResult> s;

    ctx.compile(i.arguments.opencl_src);
    s.prepare(nat, sembl, dv_prmtr, dv_rslt);

    uint32_t idx_m0;
    while (q.dequeue(idx_m0)) {

        s.run_single(idx_m0, nat, sembl, dv_prmtr, dv_rslt);

        h.save(s.get(Result::SEMBL), idx_m0, HostResult::SEMBL);
        h.save(s.get(Result::STACK), idx_m0, HostResult::STACK);
        h.save(s.get(Result::V), idx_m0, HostResult::V);

        if (!s.get(Result::A).empty()) {
            h.save(s.get(Result::A), idx_m0, HostResult::A);
        }

        if (!s.get(Result::B).empty()) {
            h.save(s.get(Result::B), idx_m0, HostResult::B);
        }

        h.save(s.get(HostResult::EFFICIENCY), idx_m0, HostResult::EFFICIENCY);
        h.save(s.get(HostResult::INTR_PER_SEC), idx_m0, HostResult::INTR_PER_SEC);
    }
}

int main(int argc, char *argv[]) {

    InputData in;
    in.read(argc, argv);

    SharedQueue queue;
    queue.enqueue(0, static_cast<uint32_t>(in.gather.cdps.size()));

    SemblanceHostResult hst_rslt;
    hst_rslt.create_all(in.arguments.chosen_method, MethodStrategy::GA, static_cast<uint32_t>(in.gather.cdps.size()), in.gather.nsmpl_per_trce);

    std::vector<std::thread> threads(MAX_PLATFORMS * MAX_DEVICE_PER_PLATFORM);
    std::vector<cl_platform_id> pltfrms(MAX_PLATFORMS);
    std::vector<cl_device_id> dv_per_pltfrm(MAX_DEVICE_PER_PLATFORM);

    // Get platform and device information
    uint32_t num_devices, num_platforms, devicesCount = 0;
    cl_int ret = clGetPlatformIDs(MAX_PLATFORMS, pltfrms.data(), &num_platforms);

    if (ret != CL_SUCCESS) {
        LOGI("Error in clGetPlatformIDs");
        return -1;
    }

    for (uint32_t p = 0; p < num_platforms; p++) {

        ret = clGetDeviceIDs(pltfrms[p], CL_DEVICE_TYPE_GPU, MAX_DEVICE_PER_PLATFORM, dv_per_pltfrm.data(), &num_devices);

        if (ret != CL_SUCCESS) {
            continue;
        }

        for (uint32_t d = 0; d < num_devices; d++, devicesCount++) {
            threads[devicesCount] = std::thread(wkr_thread, std::ref(in), std::ref(hst_rslt), std::ref(queue), pltfrms[p], dv_per_pltfrm[d]);
        }
    }

    std::chrono::steady_clock::time_point st = std::chrono::steady_clock::now();

    for(uint32_t i = 0; i < devicesCount; ++i) {
        threads[i].join();
    }

    std::chrono::duration<double> elpsd_time = std::chrono::steady_clock::now() - st;

    hst_rslt.write(in, elpsd_time, MethodStrategy::GA, MethodType::OPENCL);

    return 0;
}
