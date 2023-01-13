#pragma once

#include "BDMat.h"
#include "DSMat_CPU.h" 

/**
 * class BDMat_CPU.
 * @brief This class is used to represent decoupled problems that can be split as:
 *            A = Id_{2^nsym} \otimes Ainn + Aout,
 *        where \otimes stands for the Kronecker product, and Aout=diag(aout) for
 *        a diagonal matrix stored as a DDMat with 2^nsym columns.
 */
class BDMat_CPU: virtual public BDMat{

    //-------------------------------------------------------------------------------------

    // Public members
    public:

    /**
     * @brief Create the object.
     */
    BDMat_CPU(const int innsym);

    /**
     * @brief Creates the object copying input.
     */
    BDMat_CPU(DSMat *__restrict__ inMatAinn, DDMat *__restrict__ inAout, iReg innsym);

    //-------------------------------------------------------------------------------------

    // Public members derived from virtual methods
    public:

    /**
     * @ brief Copy.
     */
    void copy(MatrixProd const &other);

    /**
     * @brief Deletes the object.
     */
    ~BDMat_CPU();

    /**
     * @brief Check if the matrix is on GPU
     */
    bool isOnGPU() const;

};

