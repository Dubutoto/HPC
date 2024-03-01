/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>



#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
    int    nx;            /* no. of cells in x-direction */
    int    ny;            /* no. of cells in y-direction */
    int    maxIters;      /* no. of iterations */
    int    reynolds_dim;  /* dimension for Reynolds number */
    float density;       /* density per link */
    float accel;         /* density redistribution */
    float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */

 /* Original one
 typedef struct
 {
   float speeds[NSPEEDS];
 } t_speed;
*/

typedef struct
{
    float* speed0;
    float* speed1;
    float* speed2;
    float* speed3;
    float* speed4;
    float* speed5;
    float* speed6;
    float* speed7;
    float* speed8;
} t_speed;



/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* restrict paramfile, const char* restrict obstaclefile,
               t_param* restrict params, t_speed** restrict cells_ptr, t_speed** restrict tmp_cells_ptr,
               int** restrict obstacles_ptr, float** restrict av_vels_ptr);
/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles);
int accelerate_flow(const t_param params, t_speed* restrict cells, int* restrict obstacles);
//
// int propagate(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells);
// int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
float collision(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles);
int write_values(const t_param params, t_speed* restrict cells, int* restrict obstacles, float* restrict av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* restrict params, t_speed** restrict cells_ptr, t_speed** restrict tmp_cells_ptr,
             int** restrict obstacles_ptr, float** restrict av_vels_ptr);
/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* restrict cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* restrict cells, int* restrict obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* restrict cells, int* restrict obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
    char* paramfile = NULL;    /* name of the input parameter file */
    char* obstaclefile = NULL; /* name of a the input obstacle file */
    t_param params;              /* struct to hold parameter values */
    t_speed* cells = NULL;     /* grid containing fluid densities */
    t_speed* tmp_cells = NULL; /* scratch space */
    int* obstacles = NULL;    /* grid indicating which cells are blocked */
    float* av_vels = NULL;     /* a record of the av. velocity computed for each timestep */
    struct timeval timstr;        /* structure to hold elapsed time */
    double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc;
    /* floating point numbers to calculate elapsed wallclock time */


    /* parse the command line */
    if (argc != 3)
    {
        usage(argv[0]);
    }
    else
    {
        paramfile = argv[1];
        obstaclefile = argv[2];
    }

    /* Total/init time starts here: initialise our data structures and load values from file */
    gettimeofday(&timstr, NULL);
    tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    init_tic=tot_tic;

    initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

    cells->speed0 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    cells->speed1 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    cells->speed2 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    cells->speed3 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    cells->speed4 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    cells->speed5 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    cells->speed6 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    cells->speed7 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    cells->speed8 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));

    tmp_cells->speed0 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    tmp_cells->speed1 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    tmp_cells->speed2 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    tmp_cells->speed3 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    tmp_cells->speed4 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    tmp_cells->speed5 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    tmp_cells->speed6 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    tmp_cells->speed7 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));
    tmp_cells->speed8 = (float*)aligned_alloc(64, sizeof(float) * (params.ny * params.nx));


    /* initialise densities */
    float w0 = params.density * 4.0f / 9.0f;
    float w1 = params.density      / 9.0f;
    float w2 = params.density      / 36.0f;

    //consider collapse(2) or not
#pragma omp parallel for
    for (int jj = 0; jj < params.ny; jj++) {
        for (int ii = 0; ii < params.nx; ii++) {
            int index = ii + jj * params.nx;
            /* centre */
            cells->speed0[index] = w0;
            /* axis directions */
            cells->speed1[index] = w1;
            cells->speed2[index] = w1;
            cells->speed3[index] = w1;
            cells->speed4[index] = w1;
            /* diagonals */
            cells->speed5[index] = w2;
            cells->speed6[index] = w2;
            cells->speed7[index] = w2;
            cells->speed8[index] = w2;
        }
    }

    /* Init time stops here, compute time starts*/
    gettimeofday(&timstr, NULL);
    init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    comp_tic=init_toc;

    for (int tt = 0; tt < params.maxIters; tt++) 
    {
    if (tt % 2 == 0) {
        av_vels[tt] = timestep(params, cells, tmp_cells, obstacles);
    } else {
        av_vels[tt] = timestep(params, tmp_cells, cells, obstacles);
    }

    #ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
    #endif
}

    /* Compute time stops here, collate time starts*/
    gettimeofday(&timstr, NULL);
    comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    col_tic=comp_toc;

    // Collate data from ranks here

    /* Total/collate time stops here.*/
    gettimeofday(&timstr, NULL);
    col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    tot_toc = col_toc;

    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
    write_values(params, cells, obstacles, av_vels);
    finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

    return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles)
{
    accelerate_flow(params, cells, obstacles);
    //propagate(params, cells, tmp_cells);
    //rebound(params, cells, tmp_cells, obstacles);
    const float alter = collision(params, cells, tmp_cells, obstacles);
    return alter;
}

int accelerate_flow(const t_param params, t_speed* restrict cells, int* restrict obstacles)
{
    /* compute weighting factors */
    float w1 = params.density * params.accel / 9.0f;
    float w2 = params.density * params.accel / 36.0f;

    /* modify the 2nd row of the grid */
    int jj = params.ny - 2;

#pragma omp simd
    for (int ii = 0; ii < params.nx; ii++)
    {
        /* if the cell is not occupied and
        ** we don't send a negative density */
        if (!obstacles[ii + jj*params.nx]
            && (cells->speed3[ii + jj*params.nx] - w1) > 0.0f
            && (cells->speed6[ii + jj*params.nx] - w2) > 0.0f
            && (cells->speed7[ii + jj*params.nx] - w2) > 0.0f)
        {
            /* increase 'east-side' densities */
            cells->speed1[ii + jj*params.nx] += w1;
            cells->speed5[ii + jj*params.nx] += w2;
            cells->speed8[ii + jj*params.nx] += w2;
            /* decrease 'west-side' densities */
            cells->speed3[ii + jj*params.nx] -= w1;
            cells->speed6[ii + jj*params.nx] -= w2;
            cells->speed7[ii + jj*params.nx] -= w2;
        }
    }
    return EXIT_SUCCESS;
}

float collision(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles) {
  const float c_sq = 3.0f; /* square of speed of sound */
  const float t_sq = 4.5f;
  const float h_sq = 1.5f;
  const float w0 = 4.0f / 9.0f;  /* weighting factor */
  const float w1 = 1.0f / 9.0f;  /* weighting factor */
  const float w2 = 1.0f / 36.0f; /* weighting factor */

  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u = 0.0f;          /* accumulated magnitudes of velocity for each cell */

  __assume_aligned(cells, 64);
  __assume_aligned(cells->speed0, 64);
  __assume_aligned(cells->speed1, 64);
  __assume_aligned(cells->speed2, 64);
  __assume_aligned(cells->speed3, 64);
  __assume_aligned(cells->speed4, 64);
  __assume_aligned(cells->speed5, 64);
  __assume_aligned(cells->speed6, 64);
  __assume_aligned(cells->speed7, 64);
  __assume_aligned(cells->speed8, 64);

  __assume_aligned(tmp_cells, 64);
  __assume_aligned(tmp_cells->speed0, 64);
  __assume_aligned(tmp_cells->speed1, 64);
  __assume_aligned(tmp_cells->speed2, 64);
  __assume_aligned(tmp_cells->speed3, 64);
  __assume_aligned(tmp_cells->speed4, 64);
  __assume_aligned(tmp_cells->speed5, 64);
  __assume_aligned(tmp_cells->speed6, 64);
  __assume_aligned(tmp_cells->speed7, 64);
  __assume_aligned(tmp_cells->speed8, 64);

  __assume_aligned(obstacles, 64);

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  #pragma omp parallel for reduction(+:tot_cells, tot_u)
  for (int jj = 0; jj < params.ny; jj++) {
    #pragma omp simd
    for (int ii = 0; ii < params.nx; ii++) {

      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const int y_n = (jj + 1) % params.ny;
      const int x_e = (ii + 1) % params.nx;
      const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      const float speed0 = cells->speed0[ii + jj*params.nx];
      const float speed1 = cells->speed1[x_w + jj*params.nx];
      const float speed2 = cells->speed2[ii + y_s*params.nx];
      const float speed3 = cells->speed3[x_e + jj*params.nx];
      const float speed4 = cells->speed4[ii + y_n*params.nx];
      const float speed5 = cells->speed5[x_w + y_s*params.nx];
      const float speed6 = cells->speed6[x_e + y_s*params.nx];
      const float speed7 = cells->speed7[x_e + y_n*params.nx];
      const float speed8 = cells->speed8[x_w + y_n*params.nx];
       
      /* compute local density total */
      float local_density = speed0 + speed1 + speed2 + speed3 + speed4 + speed5 + speed6 + speed7 + speed8;

      /* compute x velocity component */
      const float u_x = (speed1
                      + speed5
                      + speed8
                      - (speed3
                         + speed6
                         + speed7))
                     / local_density;
        /* compute y velocity component */
      const float u_y = (speed2
                      + speed5
                      + speed6
                      - (speed4
                         + speed7
                         + speed8))
                     / local_density;

      /* if the cell contains an obstacle */
      if (obstacles[jj*params.nx + ii]) 
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells->speed0[ii + jj*params.nx] = speed0;
        tmp_cells->speed1[ii + jj*params.nx] = speed3;
        tmp_cells->speed2[ii + jj*params.nx] = speed4;
        tmp_cells->speed3[ii + jj*params.nx] = speed1;
        tmp_cells->speed4[ii + jj*params.nx] = speed2;
        tmp_cells->speed5[ii + jj*params.nx] = speed7;
        tmp_cells->speed6[ii + jj*params.nx] = speed8;
        tmp_cells->speed7[ii + jj*params.nx] = speed5;
        tmp_cells->speed8[ii + jj*params.nx] = speed6;
      }
      /* don't consider occupied cells */
      else {
        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;
        const float u_sqh_sq = u_sq * h_sq;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq * h_sq);
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] * c_sq
                                         + (u[1] * u[1]) * t_sq
                                         - u_sqh_sq);
        d_equ[2] = w1 * local_density * (1.f + u[2] * c_sq
                                         + (u[2] * u[2]) * t_sq
                                         - u_sqh_sq);
        d_equ[3] = w1 * local_density * (1.f + u[3] * c_sq
                                         + (u[3] * u[3]) * t_sq
                                         - u_sqh_sq);
        d_equ[4] = w1 * local_density * (1.f + u[4] * c_sq
                                         + (u[4] * u[4]) * t_sq
                                         - u_sqh_sq);
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] * c_sq
                                         + (u[5] * u[5]) * t_sq
                                         - u_sqh_sq);
        d_equ[6] = w2 * local_density * (1.f + u[6] * c_sq
                                         + (u[6] * u[6]) * t_sq
                                         - u_sqh_sq);
        d_equ[7] = w2 * local_density * (1.f + u[7] * c_sq
                                         + (u[7] * u[7]) * t_sq
                                         - u_sqh_sq);
        d_equ[8] = w2 * local_density * (1.f + u[8] * c_sq
                                         + (u[8] * u[8]) * t_sq
                                         - u_sqh_sq);

        /* relaxation step */
        tmp_cells->speed0[ii + jj*params.nx] = speed0
                                                + params.omega
                                                * (d_equ[0] - speed0);
        tmp_cells->speed1[ii + jj*params.nx] = speed1
                                                + params.omega
                                                * (d_equ[1] - speed1);
        tmp_cells->speed2[ii + jj*params.nx] = speed2
                                                + params.omega
                                                * (d_equ[2] - speed2);
        tmp_cells->speed3[ii + jj*params.nx] = speed3
                                                + params.omega
                                                * (d_equ[3] - speed3);
        tmp_cells->speed4[ii + jj*params.nx] = speed4
                                                + params.omega
                                                * (d_equ[4] - speed4);
        tmp_cells->speed5[ii + jj*params.nx] = speed5
                                                + params.omega
                                                * (d_equ[5] - speed5);
        tmp_cells->speed6[ii + jj*params.nx] = speed6
                                                + params.omega
                                                * (d_equ[6] - speed6);
        tmp_cells->speed7[ii + jj*params.nx] = speed7
                                                + params.omega
                                                * (d_equ[7] - speed7);
        tmp_cells->speed8[ii + jj*params.nx] = speed8
                                                + params.omega
                                                * (d_equ[8] - speed8);

      
      }
      
      tot_u += (!obstacles[jj*params.nx + ii]) ? sqrtf((u_x * u_x) + (u_y * u_y)) : 0;
      tot_cells += (!obstacles[jj*params.nx + ii]) ? 1 : 0;
      
    }
  }

  return  tot_u / (float)tot_cells;
}


float av_velocity(const t_param params, t_speed* restrict cells, int* restrict obstacles)
{
    int    tot_cells = 0;  /* no. of cells used in calculation */
    float tot_u = 0.0f;          /* accumulated magnitudes of velocity for each cell */

    __assume_aligned(cells, 64);
    __assume_aligned(cells->speed0, 64);
    __assume_aligned(cells->speed1, 64);
    __assume_aligned(cells->speed2, 64);
    __assume_aligned(cells->speed3, 64);
    __assume_aligned(cells->speed4, 64);
    __assume_aligned(cells->speed5, 64);
    __assume_aligned(cells->speed6, 64);
    __assume_aligned(cells->speed7, 64);
    __assume_aligned(cells->speed8, 64);

    __assume_aligned(obstacles, 64);

    /* loop over all non-blocked cells */
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* ignore occupied cells */
            if (!obstacles[ii + jj*params.nx])
            {
                /* local density total */
                float local_density = cells->speed0[ii + jj*params.nx] + cells->speed1[ii + jj*params.nx] + cells->speed2[ii + jj*params.nx] 
                                    + cells->speed3[ii + jj*params.nx] + cells->speed4[ii + jj*params.nx] + cells->speed5[ii + jj*params.nx] 
                                    + cells->speed6[ii + jj*params.nx] + cells->speed7[ii + jj*params.nx] + cells->speed8[ii + jj*params.nx];


                /* x-component of velocity */
                float u_x = (cells->speed1[ii + jj*params.nx]
                             + cells->speed5[ii + jj*params.nx]
                             + cells->speed8[ii + jj*params.nx]
                             - (cells->speed3[ii + jj*params.nx]
                                + cells->speed6[ii + jj*params.nx]
                                + cells->speed7[ii + jj*params.nx]))
                            / local_density;
                /* compute y velocity component */
                float u_y = (cells->speed2[ii + jj*params.nx]
                             + cells->speed5[ii + jj*params.nx]
                             + cells->speed6[ii + jj*params.nx]
                             - (cells->speed4[ii + jj*params.nx]
                                + cells->speed7[ii + jj*params.nx]
                                + cells->speed8[ii + jj*params.nx]))
                            / local_density;
                /* accumulate the norm of x- and y- velocity components */
                tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
                /* increase counter of inspected cells */
                ++tot_cells;
            }
        }
    }

    return tot_u / (float)tot_cells;
}

int initialise(const char* restrict paramfile, const char* restrict obstaclefile, t_param* restrict params, t_speed** restrict cells_ptr, t_speed** restrict tmp_cells_ptr, int** restrict obstacles_ptr, float** restrict av_vels_ptr)
{
    char   message[1024];  /* message buffer */
    FILE*   fp;            /* file pointer */
    int    xx, yy;         /* generic array indices */
    int    blocked;        /* indicates whether a cell is blocked by an obstacle */
    int    retval;         /* to hold return value for checking */

    /* open the parameter file */
    fp = fopen(paramfile, "r");

    if (fp == NULL)
    {
        sprintf(message, "could not open input parameter file: %s", paramfile);
        die(message, __LINE__, __FILE__);
    }

    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params->nx));

    if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->ny));

    if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->maxIters));

    if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

    if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->density));

    if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->accel));

    if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->omega));

    if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

    /* and close up the file */
    fclose(fp);

    /*
    ** Allocate memory.
    **
    ** Remember C is pass-by-value, so we need to
    ** pass pointers into the initialise function.
    **
    ** NB we are allocating a 1D array, so that the
    ** memory will be contiguous.  We still want to
    ** index this memory as if it were a (row major
    ** ordered) 2D array, however.  We will perform
    ** some arithmetic using the row and column
    ** coordinates, inside the square brackets, when
    ** we want to access elements of this array.
    **
    ** Note also that we are using a structure to
    ** hold an array of 'speeds'.  We will allocate
    ** a 1D array of these structs.
    */

    /* main grid */
    *cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed), 64);

    if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);


    /* 'helper' grid, used as scratch space */
    *tmp_cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed) * (params->ny * params->nx), 64);

    if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

    /* the map of obstacles */
    *obstacles_ptr = _mm_malloc(sizeof(int) * (params->ny * params->nx), 64);

    if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

    /* first set all cells in obstacle array to zero */
#pragma omp parallel for
    for (int jj = 0; jj < params->ny; jj++)
    {
        for (int ii = 0; ii < params->nx; ii++)
        {
            (*obstacles_ptr)[ii + jj*params->nx] = 0;
        }
    }

    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");

    if (fp == NULL)
    {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }

    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
    {
        /* some checks */
        if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

        if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

        if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

        if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

        /* assign to array */
        (*obstacles_ptr)[xx + yy*params->nx] = blocked;
    }

    /* and close the file */
    fclose(fp);

    /*
    ** allocate space to hold a record of the avarage velocities computed
    ** at each timestep
    */
    *av_vels_ptr = (float*)_mm_malloc(sizeof(float) * params->maxIters, 64);

    return EXIT_SUCCESS;
}

int finalise(const t_param* restrict params, t_speed** restrict cells_ptr, t_speed** restrict tmp_cells_ptr, int** restrict obstacles_ptr, float** restrict av_vels_ptr)
{
    /*
    ** free up allocated memory
    */
    _mm_free(*cells_ptr);
    *cells_ptr = NULL;

    _mm_free(*tmp_cells_ptr);
    *tmp_cells_ptr = NULL;

    _mm_free(*obstacles_ptr);
    *obstacles_ptr = NULL;

    _mm_free(*av_vels_ptr);
    *av_vels_ptr = NULL;

    return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* restrict cells, int* restrict obstacles)
{
    const float viscosity = 1.0f / 6.0f * (2.0f / params.omega - 1.0f);

    return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}


float total_density(const t_param params, t_speed* cells)
{
    float total = 0.0f;   /* accumulator */ 

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
                total += cells->speed0[ii + jj*params.nx];
                total += cells->speed1[ii + jj*params.nx];
                total += cells->speed2[ii + jj*params.nx];
                total += cells->speed3[ii + jj*params.nx];
                total += cells->speed4[ii + jj*params.nx];
                total += cells->speed5[ii + jj*params.nx];
                total += cells->speed6[ii + jj*params.nx];
                total += cells->speed7[ii + jj*params.nx];
                total += cells->speed8[ii + jj*params.nx];
        }
    }
    return total;
}


int write_values(const t_param params, t_speed* restrict cells, int* restrict obstacles, float* restrict av_vels)
{
    FILE* fp;                     /* file pointer */
    const float c_sq = 1.0f / 3.0f; /* sq. of speed of sound */
    float local_density;         /* per grid cell sum of densities */
    float pressure;              /* fluid pressure in grid cell */
    float u_x;                   /* x-component of velocity in grid cell */
    float u_y;                   /* y-component of velocity in grid cell */
    float u;                     /* norm--root of summed squares--of u_x and u_y */

    fp = fopen(FINALSTATEFILE, "w");

    if (fp == NULL) {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* an occupied cell */
            if (obstacles[ii + jj*params.nx])
            {
                u_x = u_y = u = 0.0f;
                pressure = params.density * c_sq;
            } else {
                /* no obstacle */
                float local_density = 0.0f;

                float* speeds_ptrs[] = {cells->speed0, cells->speed1, cells->speed2, cells->speed3, cells->speed4, cells->speed5, cells->speed6, cells->speed7, cells->speed8};

                for (int i = 0; i < 9; ++i)
                {
                    local_density += speeds_ptrs[i][ii + jj * params.nx];
                }

                /* compute x velocity component */
                u_x = (cells->speed1[ii + jj*params.nx]
                       + cells->speed5[ii + jj*params.nx]
                       + cells->speed8[ii + jj*params.nx]
                       - (cells->speed3[ii + jj*params.nx]
                          + cells->speed6[ii + jj*params.nx]
                          + cells->speed7[ii + jj*params.nx]))
                      / local_density;
                /* compute y velocity component */
                u_y = (cells->speed2[ii + jj*params.nx]
                       + cells->speed5[ii + jj*params.nx]
                       + cells->speed6[ii + jj*params.nx]
                       - (cells->speed4[ii + jj*params.nx]
                          + cells->speed7[ii + jj*params.nx]
                          + cells->speed8[ii + jj*params.nx]))
                      / local_density;
                /* compute norm of velocity */
                u = sqrtf((u_x * u_x) + (u_y * u_y));
                /* compute pressure */
                pressure = local_density * c_sq;
            }

            /* write to file */
            fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
        }
    }

    fclose(fp);

    fp = fopen(AVVELSFILE, "w");

    if (fp == NULL) {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (int ii = 0; ii < params.maxIters; ii++) {
        fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
    }

    fclose(fp);

    return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
    fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
    exit(EXIT_FAILURE);
}