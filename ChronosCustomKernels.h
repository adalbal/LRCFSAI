#include "DDMat_CPU.h"

#include "ChronosEnv.h"   // to use: Chronos
#include "linsol_error.h" // to throw erros
#include "mpi_error.h"    // to use: for MPI errors

#include <fstream>
#include <iomanip>

// z = a*(x .* y)
void DDMat_daxty(DDMat& z, const VEC<rExt>& a, const DDMat& x, const DDMat &y);

// xty = x^t*y
void DDMat_MtxM(VEC_CPU<rExt> &xty, const DDMat &x, const DDMat &y);

// y = x*v
// IMPORTANT: v is assumed single-column!
void DDMat_MxV(DDMat &y, const DDMat &x, const VEC_CPU<rExt> &v);

// Extract single-column DDMats from (multi-column) DDMat
void DDMat_Extract(vector<DDMat*> &subx, const DDMat& x);

// Reconstruct DDMat from single-column DDMats
void DDMat_Reconstruct(DDMat& x, const vector<DDMat*> &subx);
