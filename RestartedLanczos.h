#pragma once

#include "ChronosEnv.h"   // to use: Chronos
#include "Lanczos.h"      // to use: Lanczos
#include "linsol_error.h" // to throw errors

#include "../../src/global.h"

/**
 * class RestartedLanczos.
 * @brief This class is used to manage the restarted implementation of the Lanczos Method.
 */
class RestartedLanczos {

    //-------------------------------------------------------------------------------------

    // private members
    private:

    /*
     * @brief Number of eigenpairs to be computed.
     */
    int neig;

    /*
     * @brief Maximum number of iterations (regardless of restarting).
     */
    int itmax = 100000;

    /*
     * @brief Maximum number of iterations before restarting.
     */
    int itres;

    /**
     * @brief Exit tolerances.
     */
    double rtol = 1e-3;
    double atol = 1e-2;

    /**
     * @brief Determines which part of the spectrum is sought:
     *        "LA" - largest algebraic, for largest eigenvalues
     *        "SA" - smallest algebraic, for smallest eigenvalues
     *        "BE" - both ends, one more from high end if nev is odd
     */
    string sigma = "SA";

    /**
     * @brief This parameter determines whether eigenvectors are computed or not
     *        Note: the standard Lanczos procedure computes the eigenvalues first
     *        with the computation of eigenvectors unnecessary, if only the eigenvalues
     *        are required.
     */
    bool cptEigVecs = true;

    /*
     * @brief Scratch vectors.
     */
    DDMat* eig0 = nullptr;

    /*
     * @brief Lanczos solver.
     */
    Lanczos* lanczos = nullptr;

    //-------------------------------------------------------------------------------------
    // private functions
    private:

    /**
     * @brief Computes residual.
     * @param [out] res: residual norm.
     */
    void get_residual(MatrixProd &MAT, VEC_CPU<rExt>& eigval, DDMat_CPU& eigvec, VEC_CPU<rExt>& res);

    //-------------------------------------------------------------------------------------

    // public functions 
    public:

    /**
     * @brief Creates an empty object.
     */
    RestartedLanczos();

    /**
     * @brief Deletes the object.
     */
    ~RestartedLanczos();

    /**
     * @brief Sets the number of eigenpairs to be computed.
     */
    void set_neig(const int inneig) { neig = inneig; };

    /**
     * @brief Sets the maximum number of iterations (regardless of restarting).
     */
    void set_itmax(const int initmax) { itmax = initmax; };

    /**
     * @brief Sets the maximum number of iterations before restarting.
     */
    void set_itres(const int initres) { itres = initres; };
 
    /**
     * @brief Sets the exit relative tolerance.
     */
    void set_rtol(const double inrtol) { rtol = inrtol; };
 
    /**
     * @brief Sets the exit absolute tolerance.
     */
    void set_atol(const double inatol) { atol = inatol; };

    /**
     * @brief Returns the number of iterations performed in previous solution.
     */
    int get_ITER() const { return lanczos->get_ITER(); };

    /**
     * @brief Returns an handle to the Euclidean norm of the final iterative residual
     *        for each eigenvalue.
     */
    const rExt* get_ptr_normRES_eigvals_data() const { return lanczos->get_ptr_normRES_eigvals_data(); };
    rExt* get_ptr_normRES_eigvals_data() { return lanczos->get_ptr_normRES_eigvals_data(); };

    /**
     * @brief Solves the eigenvalue problem.
     * @param [in] MAT system matrix.
     * [in] eig0 initial eigenspace. If its size is smaller than neig, then it is padded with random vectors.
     */
    void Solve(MatrixProd &MAT, DDMat &eig0, VEC_CPU<rExt> &eigval, DDMat_CPU &eigvec,
               MatrixProd* const __restrict__ PREC = nullptr);

};
