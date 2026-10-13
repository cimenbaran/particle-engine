#ifndef PW_GRANULAR_GRID_SH
#define PW_GRANULAR_GRID_SH
// Cell width 1/8 m, origin zero. Key order: x, then y, then z.
#if defined(PW_BASIN_GRID)
#define PW_GRID_X 256
#define PW_GRID_Z 256
#define PW_GRID_OFFSET 128
#define PW_GRID_EXTENT 16.0
#define PW_GRID_SUPPORT 48.0
#else
#define PW_GRID_X 128
#define PW_GRID_Z 128
#define PW_GRID_OFFSET 64
#define PW_GRID_EXTENT 8.0
#define PW_GRID_SUPPORT 32.0
#endif
#define PW_GRID_Y 128
#define PW_GRID_KEYS uint(PW_GRID_X*PW_GRID_Y*PW_GRID_Z)
#endif
