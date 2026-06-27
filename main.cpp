#include "tile_matrix.hpp"

#include <plasma_core_blas.h>  // plasma_core_dgeqrt/dtsqrt/dormqr/dtsmqr (+ Plasma enums)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef COUT
#include <icecream.hpp>
#endif

#ifdef CHECK
#include "residual_check.hpp"
#endif

// Tile Algorithm による QR 分解 (PLASMA core カーネル + MPI 非同期通信)。
//
// 局所カーネル (double, /opt/PLASMA):
//   GEQRT(k,k,k) : plasma_core_dgeqrt   LARFB(k,j,k) : plasma_core_dormqr
//   TSQRT(i,k,k) : plasma_core_dtsqrt   SSRFB(i,j,k) : plasma_core_dtsmqr
//
// 非同期通信 (MPI_Isend/MPI_Irecv):
//   V,T のマルチキャスト送信は Isend して送信元タイルが書き換わる前に Waitall。
//   受信は Irecv して使用直前に Wait。(k,*) の貸与/書き戻しは、借用側が
//   Irecv->kernel->Isend(遅延)、所有側が Isend->Wait->Irecv->Wait(同一バッファ競合回避)。
//   送受信バッファ (A.top_buf/V_buf/T_buf) はパネル1行/1列ぶんで複数を in-flight にできる。
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int nprocs = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 引数: global_m global_n ts ib [P Q]  (前半4つは必須、P Q は対で任意)。
    if (argc < 5 || argc == 6) {
        if (rank == 0) {
            std::fprintf(stderr,
                "Usage: %s global_m global_n ts ib [P Q]\n"
                "  global_m  大域行数 (要素, 必須)\n"
                "  global_n  大域列数 (要素, 必須)\n"
                "  ts        タイルサイズ = nb (必須)\n"
                "  ib        内部ブロック, 0 < ib <= ts (必須)\n"
                "  P Q       プロセスグリッド (任意, 対で指定; 既定 P=nprocs, Q=1; P*Q==nprocs)\n"
                "例: mpirun -np 4 %s 5120 5120 128 32 2 2\n",
                argv[0], argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const int global_m = std::atoi(argv[1]);
    const int global_n = std::atoi(argv[2]);
    const int ts       = std::atoi(argv[3]);
    const int ib       = std::atoi(argv[4]);
    const int P        = (argc > 5) ? std::atoi(argv[5]) : nprocs;
    const int Q        = (argc > 6) ? std::atoi(argv[6]) : 1;

    tileqr::ProcessGrid grid(MPI_COMM_WORLD, P, Q);
    tileqr::TileMatrix  A(grid, global_m, global_n, ts, ib);

    // 局所タイルを大域インデックスから決まる値で初期化 (列優先, ld=lld)。対角優位。
    for (int I = 0; I < A.mt; ++I)
        for (int J = 0; J < A.nt; ++J)
            if (A.owns(I, J)) {
                double* t = A.tile(I, J);
                for (int jj = 0; jj < A.tile_cols(J); ++jj)
                    for (int ii = 0; ii < A.tile_rows(I); ++ii) {
                        const long gr = static_cast<long>(I) * A.nb + ii;
                        const long gc = static_cast<long>(J) * A.nb + jj;
                        double v = 0.5 + 0.5 * std::sin(0.1 * gr + 0.2 * gc);
                        if (gr == gc) v += global_m;
                        t[ii + static_cast<std::size_t>(jj) * A.lld] = v;
                    }
            }

    // カーネル用ワークスペース (PLASMA driver と同寸: tau=nb, work=ib*nb)。
    std::vector<double> tau(A.nb);
    std::vector<double> work(static_cast<std::size_t>(A.ib) * A.nb);

    const int TAG_V  = 100, TAG_T  = 101;  // LARFB: V,T of (k,k)
    const int TAG_KK = 102;                // TSQRT: (k,k) 貸与/書き戻し
    const int TAG_VI = 103, TAG_TI = 104;  // SSRFB: V,T of (i,k)
    const int TAG_KJ = 105;                // SSRFB: (k,j) 貸与/書き戻し

    // (k,*) 書き戻し送信をスロット別に保持し、そのスロット再利用前に Wait する。
    std::vector<MPI_Request> top_back(A.nt, MPI_REQUEST_NULL);
    auto wait_req = [](MPI_Request& r) {
        if (r != MPI_REQUEST_NULL) MPI_Wait(&r, MPI_STATUS_IGNORE);
    };

    // ---- 局所カーネル (PLASMA core, double) ----
    auto k_geqrt = [&](int k, double* Akk, double* Tkk) {
        plasma_core_dgeqrt(A.tile_rows(k), A.tile_cols(k), A.ib,
                           Akk, A.lld, Tkk, A.ldt, tau.data(), work.data());
    };
    auto k_larfb = [&](int k, int j, const double* Vkk, const double* Tkk, double* Ckj) {
        const int kr = std::min(A.tile_rows(k), A.tile_cols(k));
        plasma_core_dormqr(PlasmaLeft, PlasmaTrans,
                           A.tile_rows(k), A.tile_cols(j), kr, A.ib,
                           Vkk, A.lld, Tkk, A.ldt, Ckj, A.lld,
                           work.data(), A.tile_cols(j));
    };
    auto k_tsqrt = [&](int i, int k, double* Akk, double* Aik, double* Tik) {
        plasma_core_dtsqrt(A.tile_rows(i), A.tile_cols(k), A.ib,
                           Akk, A.lld, Aik, A.lld, Tik, A.ldt, tau.data(), work.data());
    };
    auto k_ssrfb = [&](int i, int j, int k, double* Akj, double* Aij,
                       const double* Vik, const double* Tik) {
        plasma_core_dtsmqr(PlasmaLeft, PlasmaTrans,
                           A.nb, A.tile_cols(j), A.tile_rows(i), A.tile_cols(j),
                           A.tile_cols(k), A.ib,
                           Akj, A.lld, Aij, A.lld, Vik, A.lld, Tik, A.ldt,
                           work.data(), A.ib);
    };

    auto trace_comm = [&](const char* ev, int peer, int idx) {
#ifdef COUT
        const int rank = grid.rank;
        IC(rank, ev, peer, idx);
#else
        (void)ev; (void)peer; (void)idx;
#endif
    };

    auto col_has_panel = [&](int c, int k) {
        for (int j = k + 1; j < A.nt; ++j)
            if (A.owner_col(j) == c) return true;
        return false;
    };

#ifdef CHECK
    // 因子化前の元行列を rank 0 へ集約 (残差の基準)。
    const std::vector<double> A0 = tileqr::gather_tiles(A, false);
#endif

    // 性能計測: 全ランクを揃えてから因子化の経過時間を測る。
    MPI_Barrier(grid.comm);
    const auto t_start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < std::min(A.mt, A.nt); ++k) {
        const int pr = A.owner_row(k);
        const int pc = A.owner_col(k);
        const int r_kk = A.owner(k, k);

        std::vector<MPI_Request> panel_sends;  // V,T(k,k): TSQRT ループ前に Wait
        std::vector<MPI_Request> trail_sends;  // V,T(i,k): k ステップ末に Wait

        // ===== GEQRT (対角) + LARFB (行 k の後続) + V,T 非同期送信 =====
        if (grid.myrow == pr) {
            if (grid.mycol == pc) {
                k_geqrt(k, A.tile(k, k), A.tile_T(k, k));
                for (int c = 0; c < grid.Q; ++c) {
                    if (c == pc || !col_has_panel(c, k)) continue;
                    const int dst = grid.rank_of(pr, c);
                    MPI_Request s1, s2;
                    MPI_Isend(A.tile(k, k),   A.nb * A.nb, MPI_DOUBLE, dst, TAG_V, grid.comm, &s1);
                    MPI_Isend(A.tile_T(k, k), A.ib * A.nb, MPI_DOUBLE, dst, TAG_T, grid.comm, &s2);
                    panel_sends.push_back(s1);
                    panel_sends.push_back(s2);
                    trace_comm("ISEND_VT", dst, k);
                }
                // ローカル LARFB を送信と重ねて実行。
                for (int j = k + 1; j < A.nt; ++j)
                    if (A.owns(k, j)) k_larfb(k, j, A.tile(k, k), A.tile_T(k, k), A.tile(k, j));

            } else if (col_has_panel(grid.mycol, k)) {
                MPI_Request rv, rt;
                MPI_Irecv(A.V_buf(k), A.nb * A.nb, MPI_DOUBLE, r_kk, TAG_V, grid.comm, &rv);
                MPI_Irecv(A.T_buf(k), A.ib * A.nb, MPI_DOUBLE, r_kk, TAG_T, grid.comm, &rt);
                MPI_Wait(&rv, MPI_STATUS_IGNORE);
                MPI_Wait(&rt, MPI_STATUS_IGNORE);
                trace_comm("IRECV_VT", r_kk, k);
                for (int j = k + 1; j < A.nt; ++j)
                    if (A.owns(k, j)) k_larfb(k, j, A.V_buf(k), A.T_buf(k), A.tile(k, j));
            }
        }
        // V,T(k,k) の送信完了を待つ (この後 TSQRT が tile(k,k) を書き換えるため)。
        if (!panel_sends.empty())
            MPI_Waitall(static_cast<int>(panel_sends.size()), panel_sends.data(), MPI_STATUSES_IGNORE);

        // ===== TSQRT / SSRFB (列 k の消去, i を逐次処理) =====
        for (int i = k + 1; i < A.mt; ++i) {
            const int pr_i = A.owner_row(i);
            const int r_ik = A.owner(i, k);

            // --- TSQRT(i,k): (k,k) を貸与して更新し書き戻す ---
            if (grid.rank == r_ik && r_ik == r_kk) {
                k_tsqrt(i, k, A.tile(k, k), A.tile(i, k), A.tile_T(i, k));
            } else if (grid.rank == r_ik) {
                wait_req(top_back[k]);  // top_buf(k) の前回書き戻し送信を待つ
                MPI_Request rr;
                MPI_Irecv(A.top_buf(k), A.nb * A.nb, MPI_DOUBLE, r_kk, TAG_KK, grid.comm, &rr);
                MPI_Wait(&rr, MPI_STATUS_IGNORE);
                trace_comm("IRECV_KK", r_kk, k);
                k_tsqrt(i, k, A.top_buf(k), A.tile(i, k), A.tile_T(i, k));
                MPI_Isend(A.top_buf(k), A.nb * A.nb, MPI_DOUBLE, r_kk, TAG_KK, grid.comm, &top_back[k]);
                trace_comm("ISEND_KK_BACK", r_kk, k);
            } else if (grid.rank == r_kk) {
                MPI_Request sr, rr;
                MPI_Isend(A.tile(k, k), A.nb * A.nb, MPI_DOUBLE, r_ik, TAG_KK, grid.comm, &sr);
                MPI_Wait(&sr, MPI_STATUS_IGNORE);  // 同一バッファへ受信する前に送信完了
                MPI_Irecv(A.tile(k, k), A.nb * A.nb, MPI_DOUBLE, r_ik, TAG_KK, grid.comm, &rr);
                MPI_Wait(&rr, MPI_STATUS_IGNORE);
                trace_comm("KK_LEND", r_ik, k);
            }

            // --- SSRFB(a): V,T of (i,k) を行 pr_i に非同期配布 (i につき1回) ---
            if (grid.myrow == pr_i) {
                if (grid.mycol == pc) {
                    for (int c = 0; c < grid.Q; ++c) {
                        if (c == pc || !col_has_panel(c, k)) continue;
                        const int dst = grid.rank_of(pr_i, c);
                        MPI_Request s1, s2;
                        MPI_Isend(A.tile(i, k),   A.nb * A.nb, MPI_DOUBLE, dst, TAG_VI, grid.comm, &s1);
                        MPI_Isend(A.tile_T(i, k), A.ib * A.nb, MPI_DOUBLE, dst, TAG_TI, grid.comm, &s2);
                        trail_sends.push_back(s1);
                        trail_sends.push_back(s2);
                        trace_comm("ISEND_VTi", dst, i);
                    }
                } else if (col_has_panel(grid.mycol, k)) {
                    MPI_Request rv, rt;
                    MPI_Irecv(A.V_buf(i), A.nb * A.nb, MPI_DOUBLE, r_ik, TAG_VI, grid.comm, &rv);
                    MPI_Irecv(A.T_buf(i), A.ib * A.nb, MPI_DOUBLE, r_ik, TAG_TI, grid.comm, &rt);
                    MPI_Wait(&rv, MPI_STATUS_IGNORE);
                    MPI_Wait(&rt, MPI_STATUS_IGNORE);
                    trace_comm("IRECV_VTi", r_ik, i);
                }
            }

            // V,T(i,k) は行 pr_i のノードだけが使う (pc 列なら手元、他列は受信バッファ)。
            const double* Vik = nullptr;
            const double* Tik = nullptr;
            if (grid.myrow == pr_i) {
                Vik = (grid.mycol == pc) ? A.tile(i, k)   : A.V_buf(i);
                Tik = (grid.mycol == pc) ? A.tile_T(i, k) : A.T_buf(i);
            }

            // --- SSRFB(b): (k,j) を貸与して更新し書き戻す ---
            for (int j = k + 1; j < A.nt; ++j) {
                const int r_kj = A.owner(k, j);
                const int r_ij = A.owner(i, j);
                if (grid.rank == r_ij && r_ij == r_kj) {
                    k_ssrfb(i, j, k, A.tile(k, j), A.tile(i, j), Vik, Tik);
                } else if (grid.rank == r_ij) {
                    wait_req(top_back[j]);  // top_buf(j) の前回書き戻し送信を待つ
                    MPI_Request rr;
                    MPI_Irecv(A.top_buf(j), A.nb * A.nb, MPI_DOUBLE, r_kj, TAG_KJ, grid.comm, &rr);
                    MPI_Wait(&rr, MPI_STATUS_IGNORE);
                    k_ssrfb(i, j, k, A.top_buf(j), A.tile(i, j), Vik, Tik);
                    MPI_Isend(A.top_buf(j), A.nb * A.nb, MPI_DOUBLE, r_kj, TAG_KJ, grid.comm, &top_back[j]);
                } else if (grid.rank == r_kj) {
                    MPI_Request sr, rr;
                    MPI_Isend(A.tile(k, j), A.nb * A.nb, MPI_DOUBLE, r_ij, TAG_KJ, grid.comm, &sr);
                    MPI_Wait(&sr, MPI_STATUS_IGNORE);
                    MPI_Irecv(A.tile(k, j), A.nb * A.nb, MPI_DOUBLE, r_ij, TAG_KJ, grid.comm, &rr);
                    MPI_Wait(&rr, MPI_STATUS_IGNORE);
                }
            }
        }

        // k ステップ末: trailing V,T 送信と未完了の書き戻し送信を待つ。
        if (!trail_sends.empty())
            MPI_Waitall(static_cast<int>(trail_sends.size()), trail_sends.data(), MPI_STATUSES_IGNORE);
        for (int s = 0; s < A.nt; ++s) wait_req(top_back[s]);
    }

    // 性能計測: 最後に同期をかけ、rank 0 のみで経過時間と GFLOPS を出力。
    MPI_Barrier(grid.comm);
    const auto t_end = std::chrono::high_resolution_clock::now();
    if (grid.rank == 0) {
        const double sec   = std::chrono::duration<double>(t_end - t_start).count();
        const double n     = static_cast<double>(global_n);
        const double flops = 4.0 / 3.0 * n * n * n;     // QR 演算量 (4/3 n^3)
        const double gflops = flops / sec / 1.0e9;
        std::printf("n,ts,ib,time,GFLOPS\n");
        std::printf("%d,%d,%d,%.6f,%.3f\n", global_n, ts, ib, sec, gflops);
    }

#ifdef CHECK
    // 因子化後の V(+R) と T を集約し、rank 0 で残差を計算。
    {
        const std::vector<double> Af = tileqr::gather_tiles(A, false);
        const std::vector<double> Tf = tileqr::gather_tiles(A, true);
        if (grid.rank == 0) {
            const double res = tileqr::residual(A, A0, Af, Tf);
            std::printf("CHECK: m=%d n=%d ts=%d ib=%d P=%d Q=%d  "
                        "||A - QR||_F / ||A||_F = %.3e\n",
                        global_m, global_n, ts, ib, P, Q, res);
        }
    }
#endif

#ifdef COUT
    // 検証: R の対角 (対角タイルの対角要素)。分散形状に依らず一致するはず。
    for (int r = 0; r < nprocs; ++r) {
        if (grid.rank == r) {
            for (int k = 0; k < std::min(A.mt, A.nt); ++k) {
                if (!A.owns(k, k)) continue;
                const double* d = A.tile(k, k);
                for (int dd = 0; dd < std::min(A.tile_rows(k), A.tile_cols(k)); ++dd) {
                    const int gdiag = k * A.nb + dd;
                    const double rval = d[dd + static_cast<std::size_t>(dd) * A.lld];
                    IC(grid.rank, gdiag, rval);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    MPI_Finalize();
    return 0;
}
