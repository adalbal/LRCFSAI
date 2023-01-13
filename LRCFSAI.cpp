#include "LRCFSAI.h"

#include "ChronosEnv.h"

#include "../hex.h" //To access PRTRB_

//----------------------------------------------------------------------------------------

// Deletes the object.
LRCFSAI::~LRCFSAI(){

    if(fsai_inn != nullptr){
        delete fsai_inn;
        fsai_inn = nullptr;
    }

    if(neig){
        for(unsigned int d=0; d<Wk.size(); ++d){
            if(Wk[d] != nullptr){
                delete Wk[d];
                Wk[d] = nullptr;
            }
        }

        if(scr != nullptr){
            delete scr;
            scr = nullptr;
        }

        for(unsigned int d=0; d<subscr.size(); ++d){
            if(subscr[d] != nullptr){
                delete subscr[d];
                subscr[d] = nullptr;
            }
        }
    }

    MatLap = nullptr;

}

//----------------------------------------------------------------------------------------

// Compute preconditioner
void LRCFSAI::Compute(MatrixProd& inMatLap, vector<DSMat*> &subMatLap){

    // cast MatrixProd to BDMat
    try {
        MatLap = dynamic_cast<BDMat*>(&inMatLap);
    }
    catch (bad_cast const&) {
        throw linsol_error("LRCFSAI::Compute", "dynamic_cast error");
    }

    // retrieve dimensions
    iReg n = this->get_nrows();
    iReg nRHS = pow(2, MatLap->nsym);

    // check consistency
    if(neig){
        if(subMatLap.size() != (unsigned)nRHS) throw linsol_error("LRCFSAI::Compute","subMatLap.size() != nRHS");
        for(int d=0; d<nRHS; ++d){
            if(MatLap->get_nrows() != subMatLap[d]->get_nrows()) throw linsol_error("LRCFSAI::Compute","incompatible Lap/subLap");
        }
    }

    // create scratches
    if(neig){
        scr = new (nothrow) DDMat_CPU;
        if(scr == nullptr) throw linsol_error("LRCFSAI::Compute", "allocating scr");
        subscr.resize(nRHS, nullptr);
        for(int d=0; d<nRHS; ++d){
            subscr[d] = new (nothrow) DDMat_CPU;
            if(subscr[d] == nullptr) throw linsol_error("LRCFSAI::Compute", "allocating subscr[]");
        }
    }

    /* compute aFSAI of MatLap->Ainn (fsai_inn) */
    fsai_inn = new aFSAI_CPU();
    #if DENSE_FSAI //Use better fsai_inn
    FSAI_HeavyParams(fsai_inn);
    #endif
    try {
        fsai_inn->Compute(*MatLap->MatAinn);
    } catch (linsol_error) {
        throw linsol_error("LRCFSAI::Compute", "computing aFSAI of MatLap->MatAinn");
    }

    if(neig){
        /* set MatList for Yfun[] = G_inn*(Ainn + Aout[])*G_inn^t (FunY[]) */
        std::vector<MatList*> FunY(nRHS, nullptr);
        for(int d=0; d<nRHS; ++d){
            FunY[d] = new MatList;
            FunY[d]->AddRight(subMatLap[d]);
            FunY[d]->AddRight(fsai_inn->get_Upper());
            FunY[d]->AddLeft(fsai_inn->get_Lower());
        }

        /* compute FunY[]'s eigenpairs (eigval[], eigvec[]) */
        std::vector<VEC_CPU<rExt>> eigval(nRHS);
        std::vector<DDMat_CPU*> eigvec(nRHS, nullptr);
        std::vector<Lanczos*> eigsol(nRHS, nullptr);
        DDMat* eigvec0 = new DDMat_CPU();
        for(int d=0; d<nRHS; ++d){
            eigval[d].assign(neig, 0.0);
            eigvec[d] = new DDMat_CPU(n, neig);
            eigsol[d] = new Lanczos;
            eigsol[d]->set_nEigs(neig);
            eigsol[d]->set_maxITER(lanczos_maxit);
            eigsol[d]->set_exitTOL(0.1*lanczos_atol); //ADEL - dirty workaround. exitTol seems rtol...
            eigsol[d]->set_eigPart(lanczos_sigma.c_str());
            eigsol[d]->set_cptEigVecs(true);

            try {
                eigsol[d]->Set_Solver(*FunY[d]);
            } catch (linsol_error) {
                throw linsol_error("LRCFSAI::Compute", "Lanczos::Set_Solver");
            }

            try {
                eigsol[d]->Solve(*FunY[d], *eigvec0, eigval[d], *eigvec[d]);
            } catch (linsol_error) {
                throw linsol_error("LRCFSAI::Compute", "Lanczos::Solve");
            }

            // correct eigenvalues: eigval[d][] = 1 - eigval[d][] (Y = Id - Yfun)
            for(int i=0; i<neig; ++i) eigval[d][i] = 1.0 - eigval[d][i];
            if(!PRTRB_ && neig) eigval[0][0] = 0.0;

            // set initial search space for Lanczos
            if(d==0) *eigvec0 = *eigvec[d];

            delete eigsol[d];
            eigsol[d] = nullptr;
            delete FunY[d];
            FunY[d] = nullptr;
        }
        delete eigvec0;
        eigvec0 = nullptr;

        /* compute low-rank correction, Wk*Dk*Wk^t, to make fsai_inn closer to subMatLap[] (Dk[], Wk[]) */
        Dk.resize(nRHS, VEC_CPU<rExt>(neig, 0.0));
        Wk.resize(nRHS, nullptr);
        for(int d=0; d<nRHS; ++d){
            Wk[d] = new DDMat_CPU(n, neig);
            for(int i=0; i<neig; ++i) Dk[d][i] = eigval[d][i]/(1.0 - eigval[d][i]); //Dk[d] = eig[d]*(Id - eig[d])^{-1}
            fsai_inn->get_Upper()->MxV(*eigvec[d], *Wk[d], true);                   //Wk = G_Sinn^t*eigvec
            delete eigvec[d];
            eigvec[d] = nullptr;
        }
    }

}

//----------------------------------------------------------------------------------------

// Set neig.
void LRCFSAI::set_neigs(const iReg inneig){

    if(MatLap != nullptr) throw linsol_error("LRCFSAI::set_neigs","should be called before Compute()");

    neig = inneig;

}

//----------------------------------------------------------------------------------------

// Set neig.
void LRCFSAI::set_maxit(const iReg inlanczos_maxit){

    if(MatLap != nullptr) throw linsol_error("LRCFSAI::set_maxit","should be called before Compute()");

    lanczos_maxit = inlanczos_maxit;

}

//----------------------------------------------------------------------------------------

// Set atol.
void LRCFSAI::set_LanczosTol(const rExt inlanczos_atol){

    if(MatLap != nullptr) throw linsol_error("LRCFSAI::set_LanczosTol","should be called before Compute()");

    lanczos_atol = inlanczos_atol;

}

//----------------------------------------------------------------------------------------

// Prepares the data structure necessary for MxV.
void LRCFSAI::Prepare_MxV(const iReg nRHS){

    // check consistency
    if(nRHS != static_cast<iReg>(pow(2, MatLap->nsym))) throw linsol_error("LRCFSAI::Prepare_MxV","invalid nRHS");

    // retrieve current communicator
    MPI_Comm const currComm = Chronos.Get_currComm();

    // check if LRCFSAI is prepared
    if(Prepared_flag == true){
        if(nRHS <= get_nRHS_prep()){
            return;
        } else{
            UndoPrepare_MxV();
        }
    }

    // retrieve dimensions
    iReg n = this->get_nrows();

    // allocate scratch
    if(neig){
        scr->resize(n, nRHS);
        for(int d=0; d<nRHS; ++d){
            subscr[d]->resize(n, 1);
        }
    }

    // prepare blocks
    //MatLap->Prepare_MxV(nRHS);
    fsai_inn->Prepare_MxV(nRHS);

    // set flag
    Prepared_flag = true;
    nRHS_prep = nRHS;

    // Syncronize MPI processes before exit
    MPI_Barrier(currComm);

}

//----------------------------------------------------------------------------------------

// Undo the data structure necessary for MxV.
void LRCFSAI::UndoPrepare_MxV(){

    //MatLap->UndoPrepare_MxV();
    fsai_inn->UndoPrepare_MxV();

    Prepared_flag = false;

}

//----------------------------------------------------------------------------------------

// Compute MxV
void LRCFSAI::MxV(DDMat& x, DDMat& b, bool Barrier_flag){

    // check consistency
    if(x.get_nrows() != b.get_nrows()) throw linsol_error("LRCFSAI::MxV","incompatible x/b");
    if(x.get_nrows() != this->get_nrows()) throw linsol_error("LRCFSAI::MxV","invalid x.nrows");
    if(b.get_nrows() != this->get_nrows()) throw linsol_error("LRCFSAI::MxV","invalid b.nrows");
    if(x.get_ncols() != this->get_nRHS_prep()) throw linsol_error("LRCFSAI::MxV","invalid x.ncols");
    if(b.get_ncols() != this->get_nRHS_prep()) throw linsol_error("LRCFSAI::MxV","invalid b.ncols");

    // retrieve current communicator
    MPI_Comm const currComm = Chronos.Get_currComm();

    // retrieve dimensions
    iReg nRHS = this->get_nRHS_prep();

    VEC_CPU<VEC_CPU<rExt>> tmpk(nRHS);
    VEC_CPU<rExt> plus(nRHS);
    plus.assign(nRHS, 1.0);

    // b = (fsai_inn + Wk*Dk*Wkt)*x
    fsai_inn->MxV(x, b, true);                                      //b = fsai_inn*x
    if(neig){
        DDMat_Extract(subscr, x);
        for(int d=0; d<nRHS; ++d){
            DDMat_MtxM(tmpk[d], *Wk[d], *subscr[d]);                //tmpk[] = Wkt[]*subscr[]   [neig x 1]
            for(int i=0; i<neig; ++i) tmpk[d][i] *= Dk[d][i];       //tmpk[] *= Dk[]            [neig x 1]
            DDMat_MxV(*subscr[d], *Wk[d], tmpk[d]);                 //subscr[] = Wk[]*tmpk[]    [n x 1]
        }
        DDMat_Reconstruct(*scr, subscr);
        b.daxpy(plus, *scr);                                        //b += scr
    }

    // Syncronize MPI processes before exit
    if(Barrier_flag) MPI_Barrier(currComm);

}

//----------------------------------------------------------------------------------------

// Returns an estimate of the number of non-zeros of the preconditioner on this process
iExt LRCFSAI::Get_nnz() const
{

    // retrieve dimensions
    iReg nRHS = this->get_nRHS_prep();

    return fsai_inn->Get_nnz() + nRHS*neig*MatLap->get_nrows();

}

//----------------------------------------------------------------------------------------

// Returns an estimate of the number of FLOPs
long int LRCFSAI::Get_FlopEst() const
{

  return static_cast<long int>(fsai_inn->Get_FlopEst()) +   //fsai_inn
         static_cast<long int>(this->get_nrows()) +         //axpy
         2L*static_cast<long int>(this->get_nRHS_prep())*   //Wk*Wk^t
         static_cast<long int>(this->get_nrows())*
         static_cast<long int>(Wk[0]->get_ncols());

}

//----------------------------------------------------------------------------------------

// Check if the matrix is on GPU
bool LRCFSAI::isOnGPU() const {

    if (this->fsai_inn != nullptr) { 
        return this->fsai_inn->isOnGPU();
    } else {
        throw linsol_error("LRCFSAI::isOnGPU","setup of LRCFSAI not called yet");
    }

}

//----------------------------------------------------------------------------------------
