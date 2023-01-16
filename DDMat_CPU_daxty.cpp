#include "ChronosCustomKernels.h"

//----------------------------------------------------------------------------------------

// Compute multiple daxty
void multi_daxty(const iReg nrows, const iReg ncols, const rExt* __restrict__  da,
                 const rExt* __restrict__ dx, const rExt* __restrict__ dy, rExt* __restrict__ dz){

    iExt k;
    iExt ncols_iExt = iExt(ncols);

    switch(ncols)
    {

        case 1:
            for(iReg i=0; i<nrows; i++){
                dz[i] = da[0] * dx[i] * dy[i];
            }
            break;

        case 2:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dz[k+0] = da[0] * dx[k+0] * dy[k+0];
                dz[k+1] = da[1] * dx[k+1] * dy[k+1];
            }
            break;

        case 4:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dz[k+0] = da[0] * dx[k+0] * dy[k+0];
                dz[k+1] = da[1] * dx[k+1] * dy[k+1];
                dz[k+2] = da[2] * dx[k+2] * dy[k+2];
                dz[k+3] = da[3] * dx[k+3] * dy[k+3];
            }
            break;

        case 8:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dz[k+0] = da[0] * dx[k+0] * dy[k+0];
                dz[k+1] = da[1] * dx[k+1] * dy[k+1];
                dz[k+2] = da[2] * dx[k+2] * dy[k+2];
                dz[k+3] = da[3] * dx[k+3] * dy[k+3];
                dz[k+4] = da[4] * dx[k+4] * dy[k+4];
                dz[k+5] = da[5] * dx[k+5] * dy[k+5];
                dz[k+6] = da[6] * dx[k+6] * dy[k+6];
                dz[k+7] = da[7] * dx[k+7] * dy[k+7];
            }
            break;


        default:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                for(iReg j=0; j<ncols; j++){
                    dz[k+iExt(j)] = da[j] * dx[k+iExt(j)] * dy[k+iExt(j)];
                }
            }
            break;

    } // end switch

}

//----------------------------------------------------------------------------------------

// Performs daxty: z = da*(y .* x) (da = column scaling)
void DDMat_daxty(DDMat& z, const VEC<rExt>& a, const DDMat& x, const DDMat &y){

    // check consistency
    if(x.get_nrows() != y.get_nrows()) throw linsol_error("DDMat_daxty","incompatible x/y");
    if(x.get_nrows() != z.get_nrows()) throw linsol_error("DDMat_daxty","incompatible x/z");
    if(x.get_ncols() != a.size()) throw linsol_error("DDMat_daxty","invalid x.ncols");
    if(y.get_ncols() != a.size()) throw linsol_error("DDMat_daxty","invalid y.ncols");
    if(z.get_ncols() != a.size()) throw linsol_error("DDMat_daxty","invalid z.ncols");

    // retrieve sizes
    iReg nrows = x.get_nrows();
    iReg ncols = x.get_ncols();

    // retrieve number of threads
    type_OMP_int nthreads = Chronos.Get_inpNthreads();
    nthreads = static_cast<type_OMP_int>(min(static_cast<iReg>(nthreads),nrows));
    if(nthreads == 0) nthreads = 1;

    // extract pointers
    rExt* coef_z = z.get_ptr_coef_data();
    const rExt* coef_x = x.get_ptr_coef_data();
    const rExt* coef_y = y.get_ptr_coef_data();
    const rExt* ptr_a = a.data();

    #pragma omp parallel num_threads(nthreads) 
    {

        // Get thread ID
        type_OMP_int mythid = omp_get_thread_num();

        // Retrieve thread info
        iReg firstrow;

        iReg nrows_loc;
        Chronos.Get_ThreadPart(mythid,nthreads,nrows,firstrow,nrows_loc);
        iReg lastrow = firstrow + nrows_loc - 1;

        iExt offset = iExt(ncols) * iExt(firstrow);

        multi_daxty(lastrow-firstrow+1, ncols, ptr_a, coef_x+offset, coef_y+offset, coef_z+offset);
    }

}

//----------------------------------------------------------------------------------------
