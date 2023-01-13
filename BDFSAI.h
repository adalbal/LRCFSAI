#pragma once

#include "Preconditioner.h" // to use: Preconditioner

#include "BDMat_CPU.h"      // to use: BDMat_CPU
#include "DSMat_CPU.h"      // to use: DSMat_CPU
#include "aFSAI_CPU.h"      // to use: aFSAI_CPU
#include "Lanczos.h"        // to use: Lanczos
#include "MatList.h"        // to use: MatList
#include <vector>           // to use: std::vector (instead of VEC_CPU)

#include "ChronosCustomKernels.h"

/**
 * class BDPrec.
 * @brief This class is used to manage the Reverse Augmented preconditioner.
 */
class BDPrec: public Preconditioner{

    //-------------------------------------------------------------------------------------

    // private members
    private:

    /**
     * @brief Vector of preconditioners, one per subsystem.
     */
    std::vector<Preconditioner*> subPrec;

    /**
     * @brief Scratch array for MxV computation.
     * @details This scratch is allocated during the first Prepare_MxV.
     */
    std::vector<DDMat*> subscr1;
    std::vector<DDMat*> subscr2;

    /**
     * @brief Vector of preconditioners, one per subsystem.
     */
    int nblk = 0;


    //-------------------------------------------------------------------------------------
    // private functions
    private:

    //-------------------------------------------------------------------------------------

    // public functions 
    public:

    /**
     * @brief Creates an empty object.
     */
    BDPrec(const int innblk);

    /**
     * @brief Computes the Block Diagonal FSAI preconditioner in a distributed memory environment.
     * @param [in] subMatLap[i] ith subsystem coefficient matrix (1 <= i <= 2^nsym).
     */
    void Compute(vector<DSMat*> &subMatLap);

    //-------------------------------------------------------------------------------------

    // public functions derived from virtual functions
    public:

    /**
     * @brief Deletes the object.
     */
    ~BDPrec();

    /**
     * @brief Copy operator.
     */
    void copy(MatrixProd const& /*other*/){
        throw linsol_error("BDPrec::copy","not implemented yet");
    };

    /**
     * @brief Computes the Low-Rank Corrected FSAI preconditioner in a distributed memory environment.
     * @param [in] mat_A_in block-diagonal coefficients matrix (with 2^nsym subsystems).
     */
    void Compute(MatrixProd& /*inMatLap*/, DDMat* const __restrict__ /*V0 = nullptr*/){
        throw linsol_error("BDPrec::Compute", "Wrong matrix format");
    };
    void Compute(DSMat& /*inMatLap*/, DDMat* const __restrict__ /*V0 = nullptr*/){
        throw linsol_error("BDPrec::Compute", "Wrong matrix format");
    };

    /**
     * @brief Prepares the data structure necessary for MxV.
     */
    void Prepare_MxV(const iReg nRHS);

    /**
     * @brief Undo the data structure necessary for MxV.
     */
    void UndoPrepare_MxV();

    /**
     * @brief Applies the preconditioner to a matrix (DSMat).
     * @param [in] Mat system matrix.
     * @param [out] PrecMat.
     */
    void MxM(DSMat_CPU& /*Mat*/, DSMat_CPU& /*PrecMat*/){
        throw linsol_error("BDPrec::copy","not implemented yet");
    };
    void MxM(DSMat& /*Mat*/, DSMat& /*PrecMat*/){
        throw linsol_error("BDPrec::copy","not implemented yet");
    };

    /**
     * @brief Computes the BDPrec Matrix by Vector product.
     * @param [in] x vector that is multiplied by the matrix.
     * @param [out] b vector where the result is stored.
     * @param [in] Barrier_flag if true global Barrier is called before return.
     */
    void MxV(DDMat& x, DDMat& b, bool Barrier_flag);

    /**
     * @brief Get nrows.
     */
    iReg get_nrows() const { return subPrec[0]->get_nrows(); };

    /**
     * @brief Prints the Preconditioner in ASCII format
     */
    void Print_ASCII(const string& /*filename*/) const{
        throw linsol_error("BDPrec::Print_ASCII","not implemented yet");
    };

    /**
     * @brief Sets the preconditioner parameters.
     * @param [in] params input parameters.
     */
    void Set_Parameters(const void* /*params*/){
        throw linsol_error("BDPrec::Set_Parameters","not implemented yet");
    };

    /**
     * @brief Returns an estimate of the number of non-zeros of the preconditioner on this process.
     */
    iExt Get_nnz() const;

    /**
     * @brief Returns an estimate of the number of FLOP needed to perform the preconditioner application on
     * this process.
     */
    long int Get_FlopEst() const;

    //-------------------------------------------------------------------------------------

    // private functions derived from virtual functions
    private:

    iGlo Get_Globalnrows_inner() const{ return subPrec[0]->Get_Globalnrows(); };

    bool Check_inner() const { return true; }; //ADEL - WTF??

    //-------------------------------------------------------------------------------------

    // public functions derived from virtual functions
    public:

    /**
     * @brief Prints the operator statistics
     * @param [in] *ofile output file.
     */
    void print_OperStats(FILE* /*ofile*/){
        throw linsol_error("BDPrec::print_OperStats","not implemented yet");
    };

    /**
     * @brief Prints the times needed for computation
     * @param [in] *ofile output file.
     */
    void print_Times(FILE* /*ofile*/){
        throw linsol_error("BDPrec::print_Times","not implemented yet");
    };

    //-------------------------------------------------------------------------------------
    private:

    /**
     * @brief Check if the matrix is on GPU
     */
    bool isOnGPU() const;

};
