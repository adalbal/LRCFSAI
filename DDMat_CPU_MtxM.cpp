#include "ChronosCustomKernels.h"

//----------------------------------------------------------------------------------------

// Compute parallel dot product between x and y.
// This function uses only one column of x and multiple columns on y.

#ifdef MultThrCom

// DDOT WITH MULTIPLE COMM

#else

// DDOT WITH FUNNELED COMM
void multi_MtxM(const iReg nrows, const rExt* __restrict__ x, const iReg n1, const rExt* __restrict__ y, const iReg n2,
                rExt* __restrict__ xty){

    iExt n1_iExt = iExt(n1);
    iExt n2_iExt = iExt(n2);
    iExt jx = 0;
    iExt jy = 0;

    for(iReg i=0; i<nrows; i++){
        for(iReg j1=0; j1<n1; j1++){
            for(iReg j2=0; j2<n2; j2++){
                iExt ind1 = iExt(i)*n1_iExt + iExt(j1);
                iExt ind2 = iExt(i)*n2_iExt + iExt(j2);
                xty[j1*n2 + j2] += x[ind1] * y[ind2];
            }
            jy += n2_iExt;
        }
        jx += n1_iExt;
    }

}

#endif

//----------------------------------------------------------------------------------------

// Performs scalar product: x^t * y = dot
void DDMat_MtxM(VEC_CPU<rExt> &xty, const DDMat &x, const DDMat &y){

    // check consistency
    if(x.get_nrows() != y.get_nrows()) throw linsol_error("DDMat_MtxM","incompatible x/y");

    // Check DDMat_CPU
    const DDMat_CPU *ptr_x = nullptr;
    ptr_x = dynamic_cast<DDMat_CPU const *>(&x);
    if(ptr_x == nullptr) throw linsol_error("DDMat_MtxM","casting of x");

    // retrieve processor rank and size
    MPI_Comm currComm = Chronos.Get_currComm();

    // retrieve dimensions 
    iReg nrows = x.get_nrows();
    iReg n1 = x.get_ncols();
    iReg n2 = y.get_ncols();
    
    // alloc output and extract pointer
    xty.resize(n1*n2);
    rExt* coef_xty = xty.data();

    // retrieve number of threads
    type_OMP_int nthreads = Chronos.Get_inpNthreads();
    nthreads = static_cast<type_OMP_int>(min(static_cast<iReg>(nthreads),nrows));
    if(nthreads == 0) nthreads = 1;

    const rExt *coef_x = x.get_ptr_coef_data();
    const rExt *coef_y = y.get_ptr_coef_data();

    rExt *xty_loc = (rExt*) calloc(n1*n2, sizeof(rExt));

    // OpenMP reduction
    #pragma omp parallel num_threads(nthreads) reduction(+:xty_loc[0:n1*n2])
    {
        // get thread ID
        type_OMP_int tid = omp_get_thread_num();

        // retrieve thread info
        iReg firstrow;
        iReg nrows_loc;
        Chronos.Get_ThreadPart(tid, nthreads, nrows, firstrow, nrows_loc);

        // compute offsets
        iExt o1 = iExt(n1) * iExt(firstrow);
        iExt o2 = iExt(n2) * iExt(firstrow);

        // compute local xty
        multi_MtxM(nrows_loc, coef_x+o1, n1, coef_y+o2, n2, xty_loc);

    }

    // MPI reduction
    if(n1*n2 > MPI_MAX_int) throw linsol_error("DDMat_MtxM","n1*n2 > MPI_MAX_int");
    type_MPI_int ierr = MPI_Allreduce(&xty_loc[0], &coef_xty[0], type_MPI_int(n1*n2), MPI_rExt, MPI_SUM, currComm);
    if(ierr) throw mpi_error("DDMat_MtxM","MPI_Allreduce","Unknown");

    free(xty_loc);

}

//----------------------------------------------------------------------------------------
    
