#pragma once

// TileQR-MPI: Tile Algorithm による QR 分解のための分散行列記述子
//
//   ProcessGrid : 2D プロセスグリッドと自ランクの位置 (BLACS context + gridinfo 相当)
//   TileMatrix  : 行列の分散記述子 + ノード内タイルデータ (ScaLAPACK descA + ローカル配列 相当)
//
// 規約:
//   - ランク順序は行優先         rank = myrow*Q + mycol
//   - タイルは正方 (nb x nb)、端タイルのみ小さくなる
//   - タイル内は列優先 (LAPACK/BLAS をそのまま局所カーネルに使える)

#include <mpi.h>

#include <cassert>
#include <cstddef>
#include <memory>

namespace tileqr {

// numroc 相当 (ScaLAPACK)。ブロック幅=1 でタイル数を数える版。
// 大域タイル数 n_global のうち、グリッド添字 idx のプロセスが持つタイル数を返す。
inline int local_tile_count(int n_global, int idx, int src, int np) {
    const int mydist = (np + idx - src) % np;
    const int extra  = n_global % np;
    return n_global / np + (mydist < extra ? 1 : 0);
}

// ----------------------------------------------------------------------------
// ProcessGrid : 2D プロセスグリッド (P x Q) と自ランクの位置 (複数行列で共有)
// ----------------------------------------------------------------------------
struct ProcessGrid {
    MPI_Comm comm;    // ベースコミュニケータ
    int nprocs;       // 総プロセス数 = P*Q
    int rank;         // 自身のMPIランク
    int P;            // 縦のノード数 (nprow)
    int Q;            // 横のノード数 (npcol)
    int myrow;        // グリッド上の自分の行 [0,P)
    int mycol;        // グリッド上の自分の列 [0,Q)

    // P x Q のグリッドを comm 上に構成する (行優先割り当て)。
    ProcessGrid(MPI_Comm comm_, int P_, int Q_)
        : comm(comm_), P(P_), Q(Q_) {
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);
        assert(P * Q == nprocs && "P*Q がプロセス数と一致しません");
        myrow = rank / Q;   // 行優先
        mycol = rank % Q;
    }

    // グリッド座標 (pr, pc) のプロセスのランク。
    int rank_of(int pr, int pc) const { return pr * Q + pc; }
};

// ----------------------------------------------------------------------------
// TileMatrix : 2D ブロックサイクリック分散された行列 (descA 相当 + ローカルデータ)
// ----------------------------------------------------------------------------
struct TileMatrix {
    const ProcessGrid* grid;

    int M, N;         // 大域の要素サイズ (行・列)        <- descA: M_, N_
    int nb;           // タイルサイズ (正方)              <- descA: MB_=NB_
    int ib;           // inner block: 局所カーネル(GEQRT/TSQRT等)の内部ブロック幅
    int mt, nt;       // 大域タイル数 = ceil(M/nb), ceil(N/nb)
    int rsrc, csrc;   // 先頭タイルの所有プロセス座標      <- descA: RSRC_, CSRC_
    int mtloc, ntloc; // 自ノードが持つタイル数 (行・列方向)
    int lld;          // 各タイルの leading dimension (=nb) <- descA: LLD_
    int ldt;          // T タイルの leading dimension (=ib)

    // 自ノードのタイル群 (タイルグリッドは列優先、各タイルは nb*nb を確保)。
    std::size_t data_size;            // data の要素数
    std::unique_ptr<double[]> data;   // ローカルタイルデータ

    // T 行列 (Householder 反射の三角因子)。各タイルに ib*nb (= ib*ts) を確保し、
    // data と同じタイル分散・列優先 (ib 行 nb 列, ld=ldt) で持つ。
    // GEQRT/TSQRT が生成し、LARFB/SSRFB の適用で参照する。
    std::size_t T_size;               // T の要素数 (= ib*nb*mtloc*ntloc)
    std::unique_ptr<double[]> T;      // ローカル T タイルデータ

    // 送受信バッファ (非同期通信用、ノード単位)。タイル分散とは別の通信領域で、
    // パネル1行/1列ぶんを保持し、複数の送受信を同時に in-flight にできるようにする。
    //   comm_top : TSQRT/SSRFB で借用する一番上のタイル行 (k,*)。タイル列 j で添字、各 nb*nb。
    //   comm_V   : LARFB/SSRFB で受信する V。タイル行 i で添字、各 nb*nb。
    //   comm_T   : LARFB/SSRFB で受信する T。タイル行 i で添字、各 ib*nb。
    std::unique_ptr<double[]> comm_top;  // nb*nb*nt
    std::unique_ptr<double[]> comm_V;    // nb*nb*mt
    std::unique_ptr<double[]> comm_T;    // ib*nb*mt

    TileMatrix(const ProcessGrid& g, int M_, int N_, int nb_, int ib_,
               int rsrc_ = 0, int csrc_ = 0)
        : grid(&g), M(M_), N(N_), nb(nb_), ib(ib_),
          rsrc(rsrc_), csrc(csrc_), lld(nb_), ldt(ib_) {
        assert(ib > 0 && ib <= nb && "ib は 0<ib<=nb を満たす必要があります");
        mt = (M + nb - 1) / nb;
        nt = (N + nb - 1) / nb;
        mtloc = local_tile_count(mt, grid->myrow, rsrc, grid->P);
        ntloc = local_tile_count(nt, grid->mycol, csrc, grid->Q);
        data_size = static_cast<std::size_t>(mtloc) * ntloc * nb * nb;
        data = std::make_unique<double[]>(data_size);  // 0 で値初期化
        T_size = static_cast<std::size_t>(mtloc) * ntloc * ib * nb;
        T = std::make_unique<double[]>(T_size);        // 0 で値初期化
        comm_top = std::make_unique<double[]>(static_cast<std::size_t>(nb) * nb * nt);
        comm_V   = std::make_unique<double[]>(static_cast<std::size_t>(nb) * nb * mt);
        comm_T   = std::make_unique<double[]>(static_cast<std::size_t>(ib) * nb * mt);
    }

    // --- 所有関係 (大域タイル添字 I,J) ----------------------------------------
    int owner_row(int I) const { return (rsrc + I) % grid->P; }
    int owner_col(int J) const { return (csrc + J) % grid->Q; }
    int owner(int I, int J) const { return grid->rank_of(owner_row(I), owner_col(J)); }
    bool owns(int I, int J) const {
        return owner_row(I) == grid->myrow && owner_col(J) == grid->mycol;
    }

    // --- 大域タイル添字 -> ローカルタイル添字 (indxg2l, ブロック幅=1) ----------
    // ローカル位置は rsrc/csrc に依存せず I/P, J/Q となる。
    int local_row(int I) const { return I / grid->P; }
    int local_col(int J) const { return J / grid->Q; }

    // --- 端タイルの実寸 (要素数) ----------------------------------------------
    int tile_rows(int I) const { return (I == mt - 1) ? M - (mt - 1) * nb : nb; }
    int tile_cols(int J) const { return (J == nt - 1) ? N - (nt - 1) * nb : nb; }

    // --- ローカルデータへのアクセス -------------------------------------------
    // ローカルタイル添字 (il, jl) のタイル先頭 (列優先、ld=lld)。
    double* local_tile(int il, int jl) {
        const std::size_t off =
            (static_cast<std::size_t>(jl) * mtloc + il) * nb * nb;
        return data.get() + off;
    }
    const double* local_tile(int il, int jl) const {
        const std::size_t off =
            (static_cast<std::size_t>(jl) * mtloc + il) * nb * nb;
        return data.get() + off;
    }

    // 大域タイル添字 (I, J) のタイル先頭。自ノードが所有している必要がある。
    double* tile(int I, int J) {
        assert(owns(I, J) && "自ノードが所有しないタイルです");
        return local_tile(local_row(I), local_col(J));
    }
    const double* tile(int I, int J) const {
        assert(owns(I, J) && "自ノードが所有しないタイルです");
        return local_tile(local_row(I), local_col(J));
    }

    // --- T 行列 (三角因子) へのアクセス ---------------------------------------
    // ローカルタイル添字 (il, jl) の T タイル先頭 (列優先、ib 行 nb 列、ld=ldt)。
    double* local_tile_T(int il, int jl) {
        const std::size_t off =
            (static_cast<std::size_t>(jl) * mtloc + il) * ib * nb;
        return T.get() + off;
    }
    const double* local_tile_T(int il, int jl) const {
        const std::size_t off =
            (static_cast<std::size_t>(jl) * mtloc + il) * ib * nb;
        return T.get() + off;
    }

    // 大域タイル添字 (I, J) の T タイル先頭。自ノードが所有している必要がある。
    double* tile_T(int I, int J) {
        assert(owns(I, J) && "自ノードが所有しないタイルです");
        return local_tile_T(local_row(I), local_col(J));
    }
    const double* tile_T(int I, int J) const {
        assert(owns(I, J) && "自ノードが所有しないタイルです");
        return local_tile_T(local_row(I), local_col(J));
    }

    // --- 送受信バッファのタイルスロット ---------------------------------------
    // 一番上のタイル行 (k,*) の借用先: タイル列 j のスロット (nb 行 nb 列, ld=lld)。
    double* top_buf(int j) { return comm_top.get() + static_cast<std::size_t>(j) * nb * nb; }
    // 受信 V: タイル行 i のスロット (nb 行 nb 列, ld=lld)。
    double* V_buf(int i)   { return comm_V.get()   + static_cast<std::size_t>(i) * nb * nb; }
    // 受信 T: タイル行 i のスロット (ib 行 nb 列, ld=ldt)。
    double* T_buf(int i)   { return comm_T.get()   + static_cast<std::size_t>(i) * ib * nb; }
};

}  // namespace tileqr
