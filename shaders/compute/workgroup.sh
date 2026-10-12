#ifndef PW_WORKGROUP_SIZE
#error PW_WORKGROUP_SIZE must be supplied by the build
#endif
#if PW_WORKGROUP_SIZE != 128 && PW_WORKGROUP_SIZE != 256
#error PW_WORKGROUP_SIZE must be 128 or 256
#endif
