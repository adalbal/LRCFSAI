#include "ChronosCustomKernels.h"

//----------------------------------------------------------------------------------------

// Compute multiple scale
void multi_Extract(const iReg nrows, const iReg ncols, const rExt* __restrict__ dx, vector<rExt*> dsubx){

    iExt k;
    iExt ncols_iExt = iExt(ncols);

    switch(ncols)
    {

        case 1:
            for(iReg i=0; i<nrows; i++){
                dsubx[0][i] = dx[i];
            }
            break;

        case 2:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dsubx[0][i] = dx[k + 0];
                dsubx[1][i] = dx[k + 1];
            }
            break;

        case 4:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt * iExt(i);
                dsubx[0][i] = dx[k + 0];
                dsubx[1][i] = dx[k + 1];
                dsubx[2][i] = dx[k + 2];
                dsubx[3][i] = dx[k + 3];
            }
            break;

        case 8:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dsubx[0][i] = dx[k + 0];
                dsubx[1][i] = dx[k + 1];
                dsubx[2][i] = dx[k + 2];
                dsubx[3][i] = dx[k + 3];
                dsubx[4][i] = dx[k + 4];
                dsubx[5][i] = dx[k + 5];
                dsubx[6][i] = dx[k + 6];
                dsubx[7][i] = dx[k + 7];
            }
            break;


        default:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                for(iReg j=0; j<ncols; j++){
                    dsubx[j][i] = dx[k + iExt(j)];
                }
            }
            break;

    } // end switch

}

//----------------------------------------------------------------------------------------

// Extract single-column DDMats from (multi-column) DDMat
void DDMat_Extract(vector<DDMat*> &subx, const DDMat& x){

    // check consistency
    if((unsigned)x.get_ncols() != subx.size()) throw linsol_error("DDMat_Extract","incompatible x/subx");
    for(unsigned int i=0; i<subx.size(); ++i){
        if(subx[i]->get_nrows() != x.get_nrows()) throw linsol_error("DDMat_Extract","incompatible x/subx");
        if(subx[i]->get_ncols() != 1) throw linsol_error("DDMat_Extract","invalid subx[].ncols");
    }

    // retrieve sizes
    iReg nrows = x.get_nrows();
    iReg ncols = x.get_ncols();

    // retrieve number of threads
    type_OMP_int nthreads = Chronos.Get_inpNthreads();
    nthreads = static_cast<type_OMP_int>(min(static_cast<iReg>(nthreads), nrows));
    if(nthreads == 0) nthreads = 1;

    // extract pointers
    vector<rExt*> coef_subx(ncols, nullptr);
    for(int i=0; i<ncols; ++i) coef_subx[i] = subx[i]->get_ptr_coef_data();
    const rExt* coef_x = x.get_ptr_coef_data();

    #pragma omp parallel num_threads(nthreads) 
    {

        // get thread ID
        type_OMP_int mythid = omp_get_thread_num();

        // retrieve thread info
        iReg firstrow;
        iReg nrows_loc;
        Chronos.Get_ThreadPart(mythid, nthreads, nrows, firstrow, nrows_loc);
        iReg lastrow = firstrow + nrows_loc - 1;

        // compute offsets
        iExt off_x = iExt(ncols) * iExt(firstrow);
        iExt off_subx = iExt(firstrow);

        // add offset to coef_subx[]
        vector<rExt*> coef_subx_off(ncols, nullptr);
        for(int i=0; i<ncols; ++i) coef_subx_off[i] = coef_subx[i] + off_subx;

        multi_Extract(lastrow - firstrow + 1, ncols, coef_x + off_x, coef_subx_off);

    }

}

//----------------------------------------------------------------------------------------
