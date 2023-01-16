#include "ChronosCustomKernels.h"

//----------------------------------------------------------------------------------------

// Compute multiple MxV
void multi_MxV(const iReg nrows, const iReg ncols, const rExt* __restrict__  dv,
                 const rExt* __restrict__ dx, rExt* __restrict__ dy){

    iExt k;
    iExt ncols_iExt = iExt(ncols);

    switch(ncols)
    {

        case 1:
            for(iReg i=0; i<nrows; i++){
                dy[i] = dv[0] * dx[i];
            }
            break;

        case 2:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dy[i]  = dx[k+0] * dv[0];
                dy[i] += dx[k+1] * dv[1];
            }
            break;

        case 3:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dy[i]  = dx[k+0] * dv[0];
                dy[i] += dx[k+1] * dv[1];
                dy[i] += dx[k+2] * dv[2];
            }
            break;

        case 4:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dy[i]  = dx[k+0] * dv[0];
                dy[i] += dx[k+1] * dv[1];
                dy[i] += dx[k+2] * dv[2];
                dy[i] += dx[k+3] * dv[3];
            }
            break;

        case 5:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dy[i]  = dx[k+0] * dv[0];
                dy[i] += dx[k+1] * dv[1];
                dy[i] += dx[k+2] * dv[2];
                dy[i] += dx[k+3] * dv[3];
                dy[i] += dx[k+4] * dv[4];
                dy[i] += dx[k+5] * dv[5];
                dy[i] += dx[k+6] * dv[6];
                dy[i] += dx[k+7] * dv[7];
            }
            break;

        case 10:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dy[i]  = dx[k+0] * dv[0];
                dy[i] += dx[k+1] * dv[1];
                dy[i] += dx[k+2] * dv[2];
                dy[i] += dx[k+3] * dv[3];
                dy[i] += dx[k+4] * dv[4];
                dy[i] += dx[k+5] * dv[5];
                dy[i] += dx[k+6] * dv[6];
                dy[i] += dx[k+7] * dv[7];
                dy[i] += dx[k+8] * dv[8];
                dy[i] += dx[k+9] * dv[9];
            }
            break;

        case 15:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dy[i]  = dx[k+0 ] * dv[0 ];
                dy[i] += dx[k+1 ] * dv[1 ];
                dy[i] += dx[k+2 ] * dv[2 ];
                dy[i] += dx[k+3 ] * dv[3 ];
                dy[i] += dx[k+4 ] * dv[4 ];
                dy[i] += dx[k+5 ] * dv[5 ];
                dy[i] += dx[k+6 ] * dv[6 ];
                dy[i] += dx[k+7 ] * dv[7 ];
                dy[i] += dx[k+8 ] * dv[8 ];
                dy[i] += dx[k+9 ] * dv[9 ];
                dy[i] += dx[k+10] * dv[10];
                dy[i] += dx[k+11] * dv[11];
                dy[i] += dx[k+12] * dv[12];
                dy[i] += dx[k+13] * dv[13];
                dy[i] += dx[k+14] * dv[14];
            }
            break;

        default:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dy[i] = 0.0;
                for(iReg j=0; j<ncols; j++){
                    dy[i] += dx[k+iExt(j)] * dv[j];
                }
            }
            break;

    } // end switch

}

//----------------------------------------------------------------------------------------

// Performs MxV: y = x*v (obviously, x.cols << x.rows)
// IMPORTANT: v is assumed single-column!
void DDMat_MxV(DDMat &y, const DDMat &x, const VEC_CPU<rExt> &v){

    // check consistency
    if(x.get_nrows() != y.get_nrows()) throw linsol_error("DDMat_MxV","incompatible x/y");
    if(x.get_ncols() != v.size()) throw linsol_error("DDMat_MxV","incompatible x/v");
    if(y.get_ncols() != 1) throw linsol_error("DDMat_MxV","invalid y.ncols");

    // retrieve sizes
    iReg nrows = x.get_nrows();
    iReg ncols = x.get_ncols();

    // retrieve number of threads
    type_OMP_int nthreads = Chronos.Get_inpNthreads();
    nthreads = static_cast<type_OMP_int>(min(static_cast<iReg>(nthreads), nrows));
    if(nthreads == 0) nthreads = 1;

    // extract pointers
    rExt* coef_y = y.get_ptr_coef_data();
    const rExt* coef_x = x.get_ptr_coef_data();
    const rExt* ptr_v = v.data();

    #pragma omp parallel num_threads(nthreads) 
    {

        // Get thread ID
        type_OMP_int mythid = omp_get_thread_num();

        // Retrieve thread info
        iReg firstrow;

        iReg nrows_loc;
        Chronos.Get_ThreadPart(mythid, nthreads, nrows, firstrow, nrows_loc);
        iReg lastrow = firstrow + nrows_loc - 1;

        iExt off_x = iExt(ncols) * iExt(firstrow);
        iExt off_y = iExt(firstrow);

        multi_MxV(lastrow - firstrow + 1, ncols, ptr_v, coef_x + off_x, coef_y + off_y);
    }

}

//----------------------------------------------------------------------------------------
