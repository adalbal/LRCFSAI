#include "RestartedLanczos.h"

#include "ChronosEnv.h"

//----------------------------------------------------------------------------------------

// Creates an empty object.
RestartedLanczos::RestartedLanczos(){

    lanczos = new Lanczos();
    eig0 = new DDMat_CPU();

}

//----------------------------------------------------------------------------------------

// Deletes the object.
RestartedLanczos::~RestartedLanczos(){

    if(eig0 != nullptr){
        delete eig0;
        eig0 = nullptr;
    }

    if(lanczos != nullptr){
        delete lanczos;
        lanczos = nullptr;
    }

}

//----------------------------------------------------------------------------------------

// Computes residual.
void RestartedLanczos::get_residual(MatrixProd &MAT, VEC_CPU<rExt> &eigval, DDMat_CPU &eigvec,
                                    VEC_CPU<rExt>& rnorm){

    // retrieve dimensions
    iReg n = MAT.get_nrows();

    // alloc scratch
    DDMat_CPU* AUk = new DDMat_CPU(n, neig);
    DDMat_CPU* DUk = new DDMat_CPU();

    // alloc temporary arrays
    rnorm.resize(neig);
    VEC_CPU<rExt> minus(neig);
    minus.assign(neig, -1.0);

    // check ||A*Uk - D*Uk|| ~ 0
    *DUk = eigvec;                          //DUk = eigvec
    DUk->scal(eigval);                      //DUk = eigval*eigvec
    MAT.MxV(eigvec, *AUk, true);            //AUk = A*eigvec
    DUk->daxpy(minus, *AUk);                //DUk -= AUk
    DUk->nrm2(rnorm);

    // free scratch
    delete AUk;
    delete DUk;

}

//----------------------------------------------------------------------------------------

// Computes eigenpairs.
void RestartedLanczos::Solve(MatrixProd &MAT, DDMat &ineig0, VEC_CPU<rExt> &eigval, DDMat_CPU &eigvec, 
                             MatrixProd* const __restrict__ PREC){

    // check consistency
    if(PREC){
        if(MAT.get_nrows() != PREC->get_nrows()) throw linsol_error("RestartedLanczos::Solve","incompatible MAT/PREC");
    }
    if(MAT.get_nrows() != eigvec.get_nrows()) throw linsol_error("RestartedLanczos::Solve","incompatible MAT/eigvec");

    // set Lanczos
    lanczos->set_nEigs(neig);
    lanczos->set_maxITER(itres);
    lanczos->set_eigPart(sigma.c_str());
    lanczos->set_cptEigVecs(cptEigVecs);
    try {
        lanczos->Set_Solver(MAT);
    } catch (linsol_error) {
        throw linsol_error("RestartedLanczos::Solve", "Lanczos::Set_Solver failed");
    }

    //compute initial residual norm
    VEC_CPU<rExt> rnorm;
    get_residual(MAT, eigval, eigvec, rnorm); //TODO: what to use instead of eigvec?

    DDMat* eigini = &ineig0;
    iReg itout = 0; //outer iterations
    iReg ittot = 0; //total iterations
    bool converged = false;
    while(!converged && ittot<itmax){

        // update eigenvectors' initial guess 
        if(itout == 1) eigini = eig0;
        if(itout >= 1) *eigini = eigvec;

        // update Lanczos tolerance
        rExt tol;
        if(atol){
            // average initial residual
            rExt rnorm_avg = 0.0;
            for(int i=0; i<neig; ++i) rnorm_avg += rnorm[i];
            rnorm_avg /= (rExt)neig;
            // compute APROXIMATE atol (cannot pass different atol to each eigenpair)
            tol = min(rtol, atol*rnorm_avg);
        } else tol = rtol;
        lanczos->set_exitTOL(tol);

        // run Lanczos
        try {
            lanczos->Solve(MAT, *eigini, eigval, eigvec, PREC);
        } catch (linsol_error) {
            throw linsol_error("RestartedLanczos::Solve", "Lanczos::Solve failed");
        }

        // update residual norm
        get_residual(MAT, eigval, eigvec, rnorm);

        // print stuff
        int its = lanczos->get_ITER();
        printf("outer-iter: %d,\t inner-iter: %d,\t Lanczos residual:\n", itout, its); rnorm.print(); printf("\n");

        // check "absolute" convergence TODO: check for "relative" convergence too
        rExt maxRes = 0.0;
        for(int i=0; i<neig; ++i) maxRes = max(maxRes, rnorm[i]);
        if(maxRes<atol) converged = true;

        itout += 1;
        ittot += itres;
    }

}

//----------------------------------------------------------------------------------------

