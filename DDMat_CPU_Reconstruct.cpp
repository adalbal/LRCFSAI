#include "ChronosCustomKernels.h"

//----------------------------------------------------------------------------------------

// Compute multiple scale
void multi_Reconstruct(const iReg nrows, const iReg ncols, vector<const rExt*> dsubx, rExt* __restrict__ dx){

    iExt k;
    iExt ncols_iExt = iExt(ncols);

    switch(ncols)
    {

        case 1:
            for(iReg i=0; i<nrows; i++){
                dx[i] = dsubx[0][i];
            }
            break;

        case 2:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dx[k + 0] = dsubx[0][i];
                dx[k + 1] = dsubx[1][i];
            }
            break;

        case 4:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt * iExt(i);
                dx[k + 0] = dsubx[0][i];
                dx[k + 1] = dsubx[1][i];
                dx[k + 2] = dsubx[2][i];
                dx[k + 3] = dsubx[3][i];
            }
            break;

        case 8:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                dx[k + 0] = dsubx[0][i];
                dx[k + 1] = dsubx[1][i];
                dx[k + 2] = dsubx[2][i];
                dx[k + 3] = dsubx[3][i];
                dx[k + 4] = dsubx[4][i];
                dx[k + 5] = dsubx[5][i];
                dx[k + 6] = dsubx[6][i];
                dx[k + 7] = dsubx[7][i];
            }
            break;


        default:
            for(iReg i=0; i<nrows; i++){
                k = ncols_iExt*iExt(i);
                for(iReg j=0; j<ncols; j++){
                    dx[k + iExt(j)] = dsubx[j][i];
                }
            }
            break;

    } // end switch

}

//----------------------------------------------------------------------------------------

// Reconstruct DDMat from single-column DDMats
void DDMat_Reconstruct(DDMat& x, const vector<DDMat*> &subx){

    // check consistency
    if((unsigned)x.get_ncols() != subx.size()) throw linsol_error("DDMat_Reconstruct","incompatible x/subx");
    for(unsigned int i=0; i<subx.size(); ++i){
        if(subx[i]->get_nrows() != x.get_nrows()) throw linsol_error("DDMat_Reconstruct","incompatible x/subx");
        if(subx[i]->get_ncols() != 1) throw linsol_error("DDMat_Reconstruct","invalid subx[].ncols");
    }

    // retrieve sizes
    iReg nrows = x.get_nrows();
    iReg ncols = x.get_ncols();

    // retrieve number of threads
    type_OMP_int nthreads = Chronos.Get_inpNthreads();
    nthreads = static_cast<type_OMP_int>(min(static_cast<iReg>(nthreads), nrows));
    if(nthreads == 0) nthreads = 1;

    // extract pointers
    rExt* coef_x = x.get_ptr_coef_data();
    vector<const rExt*> coef_subx(ncols, nullptr);
    for(int i=0; i<ncols; ++i) coef_subx[i] = subx[i]->get_ptr_coef_data();

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
        vector<const rExt*> coef_subx_off(ncols, nullptr);
        for(int i=0; i<ncols; ++i) coef_subx_off[i] = coef_subx[i] + off_subx;

        multi_Reconstruct(lastrow - firstrow + 1, ncols, coef_subx_off, coef_x + off_x);

    }

}

//----------------------------------------------------------------------------------------
