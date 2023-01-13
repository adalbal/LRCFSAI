#include "BDMat.h"

#include "ChronosEnv.h"
#include "linsol_error.h"
#include "DSMat_CPU.h"

//----------------------------------------------------------------------------------------

// Deletes the object.
BDMat::~BDMat(){

    if(MatAinn != nullptr){
        delete MatAinn;
        MatAinn = nullptr;
    }

    if(Aout != nullptr){
        delete Aout;
        Aout = nullptr;
    }

}

//----------------------------------------------------------------------------------------

// Get nrows.
iReg BDMat::get_nrows() const{

    return MatAinn->get_nrows();

}

//----------------------------------------------------------------------------------------

// Evaluates the total number of terms of the Block Diagonal matrix.
// Adds Aout NNZs too, and assumes 2nd order, ie, Aout is diagonal and Aout.Globalnterm() = Globalnrows.
iGlo BDMat::Get_Globalnterm() const{ 

    return MatAinn->Get_Globalnterm() + (nsym ? MatAinn->Get_Globalnrows() : 0);

}

//----------------------------------------------------------------------------------------

// Evaluates the total number of rows of the Reflection Symmetries matrix.
iGlo BDMat::Get_Globalnrows_inner() const{ 

    return MatAinn->Get_Globalnrows_inner();

}

//----------------------------------------------------------------------------------------

// Copy members.
void BDMat::copy_BDMat(BDMat const &other){

    // copy MatrixProd members
    copy_MatrixProd(other);

    // copy BDMat members
    nsym = other.nsym;
    if(nsym<0 || nsym>3) throw linsol_error("BDMat::copy_BDMat","invalid nsym");

    if(MatAinn != nullptr){
        delete MatAinn;
        MatAinn = nullptr;
    }
    MatAinn = new (nothrow) DSMat_CPU;
    if(MatAinn == nullptr) throw linsol_error("BDMat::copy_BDMat","allocating MatAinn");
    *MatAinn = *other.MatAinn;

    if(nsym){
        if(Aout != nullptr){
            delete Aout;
            Aout = nullptr;
        }
        Aout = new (nothrow) DDMat_CPU;
        if(Aout == nullptr) throw linsol_error("BDMat::copy_BDMat","allocating Aout");
        *Aout = *other.Aout;
    }

}

//----------------------------------------------------------------------------------------

// Copy operator.
BDMat &BDMat::operator=(BDMat const &other){

    try {
        copy_BDMat(other);
    } catch (linsol_error) {
        throw linsol_error("BDMat::operator=","copying BDMat");
    }

    return *this;

}

//----------------------------------------------------------------------------------------

// Overlaps current BDMat_CPU instance with the data pointed by input data. 
void BDMat::overlap(DSMat *__restrict__ inMatAinn, DDMat *__restrict__ inAout, iReg innsym){

    // check consistency
    if(innsym<0 || innsym>3) throw linsol_error("BDMat::overlap","invalid nsym");
    if(innsym){
        if(inMatAinn->get_nrows() != inAout->get_nrows()) throw linsol_error("BDMat::overlap","incompatible Ainn/Aout");
        if(inAout->get_ncols() != static_cast<iReg>(pow(2, innsym))) throw linsol_error("BDMat::overlap","Aout.ncols != 2^nsym");
    }

    nsym = innsym;

    if(MatAinn != nullptr){
        delete MatAinn;
        MatAinn = nullptr;
    }
    MatAinn = inMatAinn;

    if(nsym){
        if(Aout != nullptr){
            delete Aout;
            Aout = nullptr;
        }
        Aout = inAout;
    }

}

//----------------------------------------------------------------------------------------

// Assigns null pointers 
void BDMat::assign_nullptr(){

    MatAinn = nullptr;
    Aout    = nullptr;

    nsym    = -1;

}

//----------------------------------------------------------------------------------------

// Prepares the data structure necessary for MxV.
void BDMat::Prepare_MxV(const iReg nRHS){

    // check consistency
    if(nRHS != static_cast<iReg>(pow(2, nsym))) throw linsol_error("BDMat::Prepare_MxV","invalid nRHS");

    // retrieve current communicator
    MPI_Comm const currComm = Chronos.Get_currComm();

    // check if the matrix is prepared
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
    if(nsym) scr->resize(n, nRHS);

    // prepare blocks
    MatAinn->Prepare_MxV(nRHS);

    // set flag
    Prepared_flag = true;
    nRHS_prep = nRHS;

    // Syncronize MPI processes before exit
    MPI_Barrier(currComm);

}

//----------------------------------------------------------------------------------------

// Undo the data structure necessary for MxV.
void BDMat::UndoPrepare_MxV(){

    MatAinn->UndoPrepare_MxV();

    Prepared_flag = false;

}

//----------------------------------------------------------------------------------------

// Compute MxV
void BDMat::MxV(DDMat &x, DDMat &b, bool Barrier_flag){

    // check consistency
    if(x.get_nrows() != b.get_nrows()) throw linsol_error("BDMat::MxV","incompatible x/b");
    if(x.get_nrows() != this->get_nrows()) throw linsol_error("BDMat::MxV","invalid x.nrows");
    if(b.get_nrows() != this->get_nrows()) throw linsol_error("BDMat::MxV","invalid b.nrows");
    if(x.get_ncols() != this->get_nRHS_prep()) throw linsol_error("BDMat::MxV","invalid x.ncols");
    if(b.get_ncols() != this->get_nRHS_prep()) throw linsol_error("BDMat::MxV","invalid b.ncols");

    // retrieve current communicator
    MPI_Comm const currComm = Chronos.Get_currComm();

    // retrieve  dimensions
    iReg nRHS = this->get_nRHS_prep();

    // compute b = MatAinn*x + Aout.*x
    VEC_CPU<rExt> ones(nRHS, 1.0);
    MatAinn->MxV(x, b, true);                                       //b = Ainn*x
    if(nsym){
        DDMat_daxty(*scr, ones, *Aout, x);                          //scr = Aout.*x
        b.daxpy(ones, *scr);                                        //b += scr
    }

    // Syncronize MPI processes before exit
    if(Barrier_flag) MPI_Barrier(currComm);

}

//----------------------------------------------------------------------------------------

// Internal check that is specific to all derived types.
bool BDMat::Check_inner() const{ 

    bool check_OK = true;

    if(MatAinn == nullptr){
        linsol_error("BDMat::Check_inner","MatAinn is not assigned");
        check_OK = false;
    }

    if(nsym){
        if(Aout == nullptr){
            linsol_error("BDMat::Check_inner","Aout is not assigned");
            check_OK = false;
        }
    }

    return check_OK;

}

//----------------------------------------------------------------------------------------
