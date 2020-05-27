#ifndef CONTEXT_H
#define CONTEXT_H

#include <CL/cl.h>
#include <string>

#define MAX_PLATFORMS 8
#define MAX_DEVICE_PER_PLATFORM 8

class OpenCLDeviceContext {

    private:

    public:
        cl_platform_id platform_id;
        cl_device_id device_id;
        cl_context ctx;
        cl_command_queue cmd_queue;
        cl_program prgm;
        cl_kernel sembl_kernel, sembl_ga_kernel, stack_kernel, sembl_ft_cmp_crs, sembl_ft_crp;

        OpenCLDeviceContext(unsigned int deviceId);
        OpenCLDeviceContext(cl_platform_id p_id, cl_device_id deviceId);

        void compile(std::string pth);

        ~OpenCLDeviceContext();
};
#endif