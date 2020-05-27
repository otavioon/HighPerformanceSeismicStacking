#define SPITZ_ENTRY_POINT
#include "context.h"
#include "log.h"
#include "parser.h"
#include "sembl_common.h"
#include "semblance.h"
#include "spitz_common.h"
#include <chrono>
#include <mutex>
#include <spitz/spitz.hpp>
#include <vector>

class worker : public spitz::worker
{
private:
    uint32_t dv_id;
    InputData in;
    OpenCLDeviceContext ctx;
    SemblanceOpenClResult dv_rslt;
    SemblanceOpenClParameter dv_prmtr;
    SemblanceOpenCl sembl;
    SemblanceLSearch<SemblanceOpenCl, SemblanceOpenClParameter, SemblanceOpenClResult> s;
public:
    worker(int argc, const char *argv[], uint32_t id) : ctx(id), dv_rslt(ctx), dv_prmtr(ctx), sembl(in, THREADS_PER_BLOCK, ctx), dv_id(id)
    {
        in.read(argc, const_cast<char**>(argv));
        ctx.compile(in.arguments.opencl_src);
        s.prepare(sembl, dv_prmtr, dv_rslt);
    }

    worker(int argc, const char *argv[]) : worker(argc, argv, 0) {}

    int run(spitz::istream& task, const spitz::pusher& result)
    {
        std::chrono::steady_clock::time_point cdp_tm = std::chrono::steady_clock::now();

        uint32_t nsamples = in.gather.nsmpl_per_trce;

        // Binary stream used to store the output
        spitz::ostream o;

        uint32_t idx_m0;
        task >> idx_m0;

        s.run_single(idx_m0, sembl, dv_prmtr, dv_rslt);

        o.write_uint(idx_m0);
        o.write_data(s.get(Result::SEMBL).data(), nsamples * sizeof(float));
        o.write_data(s.get(Result::STACK).data(), nsamples * sizeof(float));
        o.write_data(s.get(Result::V).data(), nsamples * sizeof(float));

        if (!s.get(Result::A).empty()) {
            o.write_data(s.get(Result::A).data(), nsamples * sizeof(float));
        }

        if (!s.get(Result::B).empty()) {
            o.write_data(s.get(Result::B).data(), nsamples * sizeof(float));
        }

        o.write_float(s.get(HostResult::EFFICIENCY));
        o.write_float(s.get(HostResult::INTR_PER_SEC));

        result.push(o);

        std::chrono::duration<double> cdp_elpsd_time = std::chrono::steady_clock::now() - cdp_tm;

        LOGI("[WK] Task #%d processed. (%.3fs)", idx_m0 + 1, cdp_elpsd_time.count());

        return 0;
    }
};

// The builder binds the user code with the Spitz C++ wrapper code.
class builder : public spitz::builder
{
private:
    int dv_c = 0;
    std::mutex dv_mutex;
public:
    spitz::job_manager *create_job_manager(int argc, const char *argv[],
        spitz::istream& jobinfo)
    {
        return new SemblanceJobManager(argc, argv, jobinfo);
    }

    spitz::worker *create_worker(int argc, const char *argv[])
    {
        std::unique_lock<std::mutex> mlock(dv_mutex);
        return new worker(argc, argv, dv_c++);
    }

    spitz::committer *create_committer(int argc, const char *argv[],
        spitz::istream& jobinfo)
    {
        return new SemblanceCommitter(argc, argv, jobinfo, MethodStrategy::EXHAUSTIVE);
    }
};

// Creates a builder class.
spitz::builder *spitz_factory = new builder();
