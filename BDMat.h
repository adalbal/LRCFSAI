#pragma once

#include <vector>
#include "DSMat.h"
#include "ChronosCustomKernels.h"

/**
 * class BDMat.
 * @brief This class is used to represent Distributed Sparse Reflection Symmetries problems.
 */
class BDMat: public MatrixProd{

    //-------------------------------------------------------------------------------------

    // public members
    public:

    /**
     * @brief inner-subdomain couplings.
     */
    DSMat* MatAinn = nullptr;

    /**
     * @brief outer-subdomain couplings.
     */
    DDMat* Aout = nullptr;

    /**
     * @brief Scratch array for MxV computation.
     * @details This scratch is allocated during the first Prepare_MxV.
     */
    DDMat *scr = nullptr;

    /**
     * @brief Number of reflection symmetries.
     * @details nRHS needs to be 2^nsym (multiple RHS+symmetries is not supported)
     */
    int nsym = -1;

    //-------------------------------------------------------------------------------------

    // protected functions
    protected:

    /**
     * @brief Copy members.
     */
    void copy_BDMat(BDMat const &other);

    //-------------------------------------------------------------------------------------

    // public functions
    public:

    /**
     * @brief Copy operator.
     */
    BDMat& operator=(BDMat const &other);

    /**
     * @brief Overlap current BDMat_CPU instance with the data pointed by input data. 
     */
    void overlap(DSMat *__restrict__ inMatAinn, DDMat *__restrict__ inAout, iReg innsym);

    /**
     * @brief Assign null pointers.
     */
    void assign_nullptr();

    //-------------------------------------------------------------------------------------

    // public functions derived from virtual functions
    public:

    /**
     * @brief Prepares the data structure necessary for MxV.
     */
    void Prepare_MxV(const iReg nRHS);

    /**
     * @brief Undo the data structure necessary for MxV.
     */
    void UndoPrepare_MxV();

    /**
     * @brief Computes the Reflection Symmetry Matrix by Vector Product.
     * @param [in] x vector that is multiplied by the matrix.
     * @param [out] b vector where result is stored.
     * @param [in] Barrier_flag if true global Barrier is called before return.
     */
    void MxV(DDMat &x, DDMat &b, bool Barrier_flag);

    /**
     * @brief Get nrows.
     */
    iReg get_nrows() const; 

    /**
     * @brief Internal check that is specific to all derived types.
     */
    bool Check_inner() const;

    /**
     * @brief Evaluates the total number of terms of the Saddle Point matrix.
     * @details Adds Aout NNZs too, and assumes 2nd order, ie, Aout is diagonal and Aout.Globalnterm() = Globalnrows.
     */
    iGlo Get_Globalnterm() const;

    //-------------------------------------------------------------------------------------

    // private functions derived from virtual functions
    private:

    /**
     * Evaluates the total number of rows of the Block Diagonal matrix.
     */
    iGlo Get_Globalnrows_inner() const;

    //-------------------------------------------------------------------------------------

    // public virtual functions
    public:

    /**
     * @brief Deletes the object.
     */
    virtual ~BDMat() = 0;

    /**
     * @brief Check if the matrix is on GPU
     */
    virtual bool isOnGPU() const = 0;

};

