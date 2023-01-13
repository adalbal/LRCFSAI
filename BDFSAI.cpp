#include "BDFSAI.h"

#include "ChronosEnv.h"

//----------------------------------------------------------------------------------------

// Creates an empty object.
BDPrec::BDPrec(const int innblk){

    // check consistency
    if(innblk < 1) throw linsol_error("BDPrec::BDPrec","invalid nblk");

    nblk = innblk;

    subPrec.resize(nblk, nullptr);
    subscr1.resize(nblk, nullptr);
    subscr2.resize(nblk, nullptr);

}

//----------------------------------------------------------------------------------------

// Deletes the object.
BDPrec::~BDPrec(){

    if(nblk){
        for(int d=0; d<nblk; ++d){
            if(subPrec[d] != nullptr){
                delete subPrec[d];
                subPrec[d] = nullptr;
            }

            if(subscr1[d] != nullptr){
                delete subscr1[d];
                subscr1[d] = nullptr;
            }

            if(subscr2[d] != nullptr){
                delete subscr2[d];
                subscr2[d] = nullptr;
            }
        }

        subPrec.clear();
        subscr1.clear();
        subscr2.clear();
    }

}

//----------------------------------------------------------------------------------------

// Compute preconditioner
void BDPrec::Compute(vector<DSMat*> &subMatLap){

    // check consistency
    if((signed)subMatLap.size() != nblk) throw linsol_error("BDPrec::Compute","invalid number of subLaps");
    for(int d=0; d<nblk; ++d){
        if(subMatLap[d]->get_nrows() != subMatLap[0]->get_nrows()) throw linsol_error("BDPrec::Compute","incompatible subLaps");
    }

    // create scratches
    for(int d=0; d<nblk; ++d){
        subscr1[d] = new (nothrow) DDMat_CPU;
        if(subscr1[d] == nullptr) throw linsol_error("BDPrec::Compute", "allocating subscr1[]");
    }
    for(int d=0; d<nblk; ++d){
        subscr2[d] = new (nothrow) DDMat_CPU;
        if(subscr2[d] == nullptr) throw linsol_error("BDPrec::Compute", "allocating subscr2[]");
    }

    // compute aFSAI of each (decoupled) subsystem
    for(int d=0; d<nblk; ++d){
        subPrec[d] = new aFSAI_CPU();
        #if DENSE_FSAI //Use better fsai
        FSAI_HeavyParams(subPrec[d]);
        #endif
        try {
            subPrec[d]->Compute(*subMatLap[d]);
        } catch (linsol_error) {
            throw linsol_error("BDPrec::Compute", "computing aFSAI");
        }
    }

}

//----------------------------------------------------------------------------------------

// Prepares the data structure necessary for MxV.
void BDPrec::Prepare_MxV(const iReg nRHS){

    // check consistency
    if(nRHS != nblk) throw linsol_error("BDPrec::Prepare_MxV","invalid nRHS");

    // retrieve current communicator
    MPI_Comm const currComm = Chronos.Get_currComm();

    // check if BDPrec is prepared
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
    for(int d=0; d<nRHS; ++d){
        subscr1[d]->resize(n, 1);
        subscr2[d]->resize(n, 1);
    }

    // prepare blocks
    for(int d=0; d<nRHS; ++d) subPrec[d]->Prepare_MxV(1);

    // set flag
    Prepared_flag = true;
    nRHS_prep = nRHS;

    // Syncronize MPI processes before exit
    MPI_Barrier(currComm);

}

//----------------------------------------------------------------------------------------

// Undo the data structure necessary for MxV.
void BDPrec::UndoPrepare_MxV(){

    for(int d=0; d<nblk; ++d) subPrec[d]->UndoPrepare_MxV();

    Prepared_flag = false;

}

//----------------------------------------------------------------------------------------

// Compute MxV
void BDPrec::MxV(DDMat& x, DDMat& b, bool Barrier_flag){

    // check consistency
    if(x.get_nrows() != b.get_nrows()) throw linsol_error("BDPrec::MxV","incompatible x/b");
    if(x.get_nrows() != this->get_nrows()) throw linsol_error("BDPrec::MxV","invalid x.nrows");
    if(b.get_nrows() != this->get_nrows()) throw linsol_error("BDPrec::MxV","invalid b.nrows");
    if(x.get_ncols() != this->get_nRHS_prep()) throw linsol_error("BDPrec::MxV","invalid x.ncols");
    if(b.get_ncols() != this->get_nRHS_prep()) throw linsol_error("BDPrec::MxV","invalid b.ncols");

    // retrieve current communicator
    MPI_Comm const currComm = Chronos.Get_currComm();

    // retrieve dimensions
    iReg nRHS = this->get_nRHS_prep();

    // b[i] = subPrec[i]*x[:,i]
    DDMat_Extract(subscr1, x);                                  //subscr1[] = extract(x)
    for(int d=0; d<nRHS; ++d){
        subPrec[d]->MxV(*subscr1[d], *subscr2[d], true);        //subscr2[] = subPrec[]*subscr1[]
    }
    DDMat_Reconstruct(b, subscr2);                              //b = reconstruct(subscr2[])

    // Syncronize MPI processes before exit
    if(Barrier_flag) MPI_Barrier(currComm);

}

//----------------------------------------------------------------------------------------

// Returns an estimate of the number of non-zeros of the preconditioner on this process
iExt BDPrec::Get_nnz() const
{

    iExt numnnz = 0;
    for(int i=0; i<nblk; ++i) numnnz += subPrec[i]->Get_nnz();

    return numnnz;

}

//----------------------------------------------------------------------------------------

// Returns an estimate of the number of FLOPs
long int BDPrec::Get_FlopEst() const
{

    long int numflop = 0;
    for(int i=0; i<nblk; ++i) numflop += subPrec[i]->Get_FlopEst();

    return numflop;

}

//----------------------------------------------------------------------------------------

// Check if the matrix is on GPU
bool BDPrec::isOnGPU() const {

    if(subPrec[0] != nullptr) return this->subPrec[0]->isOnGPU();
    else throw linsol_error("BDPrec::isOnGPU","BDPrec::Compute() not called yet");

}

//----------------------------------------------------------------------------------------
