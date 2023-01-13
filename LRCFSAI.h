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
 * class LRCFSAI.
 * @brief This class is used to manage the Reverse Augmented preconditioner.
 */
class LRCFSAI: public Preconditioner{

    //-------------------------------------------------------------------------------------

    // private members
    private:

    /**
     * @brief Reflection Symmetry coefficient matrix (2x2 block Lap = [A,B;Bt,C])
     */
    BDMat* MatLap = nullptr;

    /**
     * @brief Splittable SPD preconditioner (normally aFSAI) for the inner-subdomain couplings.
     */
    aFSAI_CPU* fsai_inn = nullptr;

    /**
     * @brief Low-Rank Correction for fsai_inn (fsai_LRC = fsai_inn + Wk*Wkt).
     * @TODO Make neig, lanczos_maxit, lanczos_atol and lanczos_sigma input parameters!
     */
    iReg neig = 3;
    iReg lanczos_maxit = 1000;
    rExt lanczos_atol = 1.0e-5;
    string lanczos_sigma = "SA";
    std::vector<DDMat_CPU*> Wk;
    std::vector<VEC_CPU<rExt>> Dk;

    /**
     * @brief Scratch array for MxV computation.
     * @details This scratch is allocated during the first Prepare_MxV.
     */
    DDMat *scr = nullptr;
    std::vector<DDMat*> subscr;

    //-------------------------------------------------------------------------------------
    // private functions
    private:

    //-------------------------------------------------------------------------------------

    // public functions 
    public:

    /**
     * @brief Creates an empty object.
     */
    LRCFSAI() {};

    /**
     * @brief Computes the Low-Rank Corrected FSAI preconditioner in a distributed memory environment.
     * @param [in] mat_A_in block-diagonal coefficients matrix (with 2^nsym subsystems).
     */
    void Compute(MatrixProd& inMatLap, vector<DSMat*> &subMatLap);

    /**
     * @brief Set neig.
     */
    void set_neigs(const iReg inneig);

    /**
     * @brief Set maxit.
     */
    void set_maxit(const iReg inlanczos_maxit);

    /**
     * @brief Set atol.
     */
    void set_LanczosTol(const rExt inlanczos_atol);

    /**
     * @brief Get neig.
     */
    iReg get_neigs() const { return neig; };

    /**
     * @brief Get atol.
     */
    rExt get_LanczosTol() const { return lanczos_atol; };

    //-------------------------------------------------------------------------------------

    // public functions derived from virtual functions
    public:

    /**
     * @brief Deletes the object.
     */
    ~LRCFSAI();

    /**
     * @brief Copy operator.
     */
    void copy(MatrixProd const& /*other*/){
        throw linsol_error("LRCFSAI::copy","not implemented yet");
    };

    /**
     * @brief Computes the Low-Rank Corrected FSAI preconditioner in a distributed memory environment.
     * @param [in] mat_A_in block-diagonal coefficients matrix (with 2^nsym subsystems).
     */
    void Compute(MatrixProd& /*inMatLap*/, DDMat* const __restrict__ /*V0 = nullptr*/){
        throw linsol_error("LRCFSAI::Compute", "Wrong matrix format");
    };
    void Compute(DSMat& /*inMatLap*/, DDMat* const __restrict__ /*V0 = nullptr*/){
        throw linsol_error("LRCFSAI::Compute", "Wrong matrix format");
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
        throw linsol_error("LRCFSAI::copy","not implemented yet");
    };
    void MxM(DSMat& /*Mat*/, DSMat& /*PrecMat*/){
        throw linsol_error("LRCFSAI::copy","not implemented yet");
    };

    /**
     * @brief Computes the LRCFSAI Matrix by Vector product.
     * @param [in] x vector that is multiplied by the matrix.
     * @param [out] b vector where the result is stored.
     * @param [in] Barrier_flag if true global Barrier is called before return.
     */
    void MxV(DDMat& x, DDMat& b, bool Barrier_flag);

    /**
     * @brief Get nrows.
     */
    iReg get_nrows() const { return MatLap->get_nrows(); };

    /**
     * @brief Prints the Preconditioner in ASCII format
     */
    void Print_ASCII(const string& /*filename*/) const{
        throw linsol_error("LRCFSAI::Print_ASCII","not implemented yet");
    };

    /**
     * @brief Sets the preconditioner parameters.
     * @param [in] params input parameters.
     */
    void Set_Parameters(const void* /*params*/){
        throw linsol_error("LRCFSAI::Set_Parameters","not implemented yet");
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

    iGlo Get_Globalnrows_inner() const{ return MatLap->Get_Globalnrows(); };

    bool Check_inner() const { return true; }; //ADEL - WTF??

    //-------------------------------------------------------------------------------------

    // public functions derived from virtual functions
    public:

    /**
     * @brief Prints the operator statistics
     * @param [in] *ofile output file.
     */
    void print_OperStats(FILE* /*ofile*/){
        throw linsol_error("LRCFSAI::print_OperStats","not implemented yet");
    };

    /**
     * @brief Prints the times needed for computation
     * @param [in] *ofile output file.
     */
    void print_Times(FILE* /*ofile*/){
        throw linsol_error("LRCFSAI::print_Times","not implemented yet");
    };

    //-------------------------------------------------------------------------------------
    private:

    /**
     * @brief Check if the matrix is on GPU
     */
    bool isOnGPU() const;

};
