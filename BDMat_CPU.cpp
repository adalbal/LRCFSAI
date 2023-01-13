#include "BDMat_CPU.h"

#include "linsol_error.h" // to throw errors

//----------------------------------------------------------------------------------------

// Creates an empty object.
BDMat_CPU::BDMat_CPU(const int innsym){

    // check consistency
    if(innsym<0 || innsym>3) throw linsol_error("BDMat_CPU::BDMat_CPU","invalid nsym");

    nsym = innsym;

    MatAinn = new (nothrow) DSMat_CPU;
    if(MatAinn == nullptr) throw linsol_error("BDMat_CPU::BDMat_CPU","allocating MatAinn");

    if(nsym){
        Aout = new (nothrow) DDMat_CPU;
        if(Aout == nullptr) throw linsol_error("BDMat_CPU::BDMat_CPU","allocating Aout");

        scr = new (nothrow) DDMat_CPU;
        if(scr == nullptr) throw linsol_error("BDMat_CPU::BDMat_CPU","allocating scr");
    }

}

//----------------------------------------------------------------------------------------

// Creates the object copying input.
BDMat_CPU::BDMat_CPU(DSMat *__restrict__ inMatAinn, DDMat *__restrict__ inAout, iReg innsym){

    // check consistency
    if(innsym<0 || innsym>3) throw linsol_error("BDMat_CPU::BDMat_CPU","invalid nsym");
    if(innsym){
        if(inMatAinn->get_nrows() != inAout->get_nrows()) throw linsol_error("BDMat_CPU::BDMat_CPU","incompatible Ainn/Aout");
        if(inAout->get_ncols() != static_cast<iReg>(pow(2, innsym))) throw linsol_error("BDMat_CPU::BDMat_CPU","Aout.ncols != 2^nsym");
    }

    // check DSMat* == DSMat_CPU*
    const DSMat_CPU *inMatAinn_CPU = nullptr;
    inMatAinn_CPU = dynamic_cast<DSMat_CPU const *>(inMatAinn);
    if(inMatAinn_CPU == nullptr) throw linsol_error("BDMat_CPU::BDMat_CPU","casting of inMatAinn");

    // check DDMat* == DDMat_CPU*
    const DDMat_CPU *inAout_CPU = nullptr;
    if(innsym){
        inAout_CPU = dynamic_cast<DDMat_CPU const *>(inAout);
        if(inAout_CPU == nullptr) throw linsol_error("BDMat_CPU::BDMat_CPU","casting of inAout");
    }

    nsym = innsym;

    MatAinn = new (nothrow) DSMat_CPU;
    if(MatAinn == nullptr) throw linsol_error("BDMat_CPU::BDMat_CPU","allocating MatAinn");
    *MatAinn = *inMatAinn_CPU;

    if(nsym){
        Aout = new (nothrow) DDMat_CPU;
        if(Aout == nullptr) throw linsol_error("BDMat_CPU::BDMat_CPU","allocating Aout");
        *Aout = *inAout_CPU;

        scr = new (nothrow) DDMat_CPU;
        if(scr == nullptr) throw linsol_error("BDMat_CPU::BDMat_CPU","allocating scr");
    }

}

//----------------------------------------------------------------------------------------

// Copy.
void BDMat_CPU::copy(MatrixProd const &other){

    // check MatrixProd == DSMat_CPU
    const BDMat_CPU *pt_other = nullptr;
    pt_other = dynamic_cast<BDMat_CPU const *>(&other);
    if(pt_other == nullptr) throw linsol_error("BDMat_CPU::copy","casting of other");

    // copy DSMat members
    try {
        this->copy_BDMat(*pt_other);
    } catch (linsol_error) {
        throw linsol_error("BDMat_CPU::copy","copying BDMat members");
    }

}

//----------------------------------------------------------------------------------------

// Deletes the object.
BDMat_CPU::~BDMat_CPU(){

    if(scr != nullptr){
        delete scr;
        scr = nullptr;
    }

}

//----------------------------------------------------------------------------------------

// @brief Check if the matrix is on GPU.
bool BDMat_CPU::isOnGPU() const{

    return false;

}

//----------------------------------------------------------------------------------------


