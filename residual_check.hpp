#pragma once
#ifdef CHECK

// 残差チェック (後退誤差 ||A - Q*R||_F / ||A||_F)。CHECK ビルド時のみ有効。
// 因子化前後のタイルを rank 0 へ集約し、rank 0 で Q*R を再構成して残差を求める。
// PLASMA の dormqr/dtsmqr を PlasmaNoTrans で逆順適用して Q*R を作る。

#include "tile_matrix.hpp"

#include <plasma_core_blas.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace tileqr {

// 全 (I,J) タイルを rank 0 に集める (各タイル nb*nb、isT なら ib*nb)。全ランクが呼ぶ。
// 戻り値は rank 0 のみ非空 (I*nt+J 番目に各タイルが連続、列優先 ld=lld/ldt)。
inline std::vector<double> gather_tiles(const TileMatrix& A, bool isT) {
    const int per  = isT ? A.ib * A.nb : A.nb * A.nb;
    const int tag  = isT ? 901 : 900;
    const int rank = A.grid->rank;
    std::vector<double> full;
    if (rank == 0) full.assign(static_cast<std::size_t>(A.mt) * A.nt * per, 0.0);
    for (int I = 0; I < A.mt; ++I)
        for (int J = 0; J < A.nt; ++J) {
            const int owner = A.owner(I, J);
            const double* src = (owner == rank) ? (isT ? A.tile_T(I, J) : A.tile(I, J)) : nullptr;
            if (rank == 0) {
                double* dst = full.data() + static_cast<std::size_t>(I * A.nt + J) * per;
                if (owner == 0) std::copy(src, src + per, dst);
                else MPI_Recv(dst, per, MPI_DOUBLE, owner, tag, A.grid->comm, MPI_STATUS_IGNORE);
            } else if (owner == rank) {
                MPI_Send(const_cast<double*>(src), per, MPI_DOUBLE, 0, tag, A.grid->comm);
            }
        }
    return full;
}

// rank 0: 後退誤差 ||A0 - Q*R||_F / ||A0||_F。A0/Af/Tf は gather_tiles の結果。
inline double residual(const TileMatrix& A,
                       const std::vector<double>& A0,
                       const std::vector<double>& Af,
                       const std::vector<double>& Tf) {
    const int mt = A.mt, nt = A.nt, nb = A.nb, ib = A.ib, lld = A.lld, ldt = A.ldt;
    const int K = std::min(mt, nt);
    auto Ap = [&](const std::vector<double>& v, int I, int J) {
        return const_cast<double*>(v.data()) + static_cast<std::size_t>(I * nt + J) * nb * nb;
    };
    auto Tp = [&](const std::vector<double>& v, int I, int J) {
        return const_cast<double*>(v.data()) + static_cast<std::size_t>(I * nt + J) * ib * nb;
    };

    // B = R: 上のタイルは保持、対角タイルは上三角のみ、対角下のタイルは 0。
    std::vector<double> B = Af;
    for (int I = 0; I < mt; ++I)
        for (int J = 0; J < nt; ++J) {
            double* b = B.data() + static_cast<std::size_t>(I * nt + J) * nb * nb;
            if (I > J) std::fill(b, b + nb * nb, 0.0);
            else if (I == J)
                for (int c = 0; c < A.tile_cols(J); ++c)
                    for (int r = c + 1; r < nb; ++r)
                        b[r + static_cast<std::size_t>(c) * lld] = 0.0;
        }

    std::vector<double> work(static_cast<std::size_t>(ib) * nb);

    // Â = Q*B : 因子化の逆順 (k 降順, 各 k で i 降順の TSMQR の後に ORMQR)、NoTrans。
    for (int k = K - 1; k >= 0; --k) {
        for (int i = mt - 1; i >= k + 1; --i)
            for (int j = k; j < nt; ++j)
                plasma_core_dtsmqr(PlasmaLeft, PlasmaNoTrans,
                    nb, A.tile_cols(j), A.tile_rows(i), A.tile_cols(j), A.tile_cols(k), ib,
                    Ap(B, k, j), lld, Ap(B, i, j), lld,
                    Ap(Af, i, k), lld, Tp(Tf, i, k), ldt, work.data(), ib);
        const int kr = std::min(A.tile_rows(k), A.tile_cols(k));
        for (int j = k; j < nt; ++j)
            plasma_core_dormqr(PlasmaLeft, PlasmaNoTrans,
                A.tile_rows(k), A.tile_cols(j), kr, ib,
                Ap(Af, k, k), lld, Tp(Tf, k, k), ldt,
                Ap(B, k, j), lld, work.data(), A.tile_cols(j));
    }

    // ||A0 - Â||_F / ||A0||_F (各タイルの実寸のみ)。
    double num = 0.0, den = 0.0;
    for (int I = 0; I < mt; ++I)
        for (int J = 0; J < nt; ++J) {
            const double* a0 = Ap(A0, I, J);
            const double* b  = Ap(B, I, J);
            for (int c = 0; c < A.tile_cols(J); ++c)
                for (int r = 0; r < A.tile_rows(I); ++r) {
                    const std::size_t o = r + static_cast<std::size_t>(c) * lld;
                    const double d = a0[o] - b[o];
                    num += d * d;
                    den += a0[o] * a0[o];
                }
        }
    return std::sqrt(num) / std::sqrt(den);
}

}  // namespace tileqr

#endif  // CHECK
