/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sgemm_kernel_rvv.cpp

Abstract:

    This module implements an RVV kernel for the single precision matrix/matrix
    multiply operation (SGEMM) on riscv64.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>

namespace
{
// The packed B layout stays 16 columns wide to match MLAS, but each tile is
// consumed in runtime-sized RVV chunks so the kernel is not tied to a fixed
// VLEN such as 128 or 256 bits.
constexpr size_t kPackedCountN = 16;

template <size_t Lmul>
struct MlasSgemmRvvTraits;

#define MLAS_SGEMM_RVV_TRAITS(Lmul, Suffix)                                           \
    template <>                                                                                                 \
    struct MlasSgemmRvvTraits<Lmul> {                                                                           \
        using Vector = vfloat32##Suffix##_t;                                                                    \
                                                                                                                \
                                                                                                                \
        static MLAS_FORCEINLINE size_t VectorLength(size_t Count) { return __riscv_vsetvl_e32##Suffix(Count); } \
                                                                                                                \
        static MLAS_FORCEINLINE Vector Zero(size_t vl) { return __riscv_vfmv_v_f_f32##Suffix(0.0f, vl); }       \
                                                                                                                \
        static MLAS_FORCEINLINE Vector Load(const float* Address, size_t vl)                                    \
        {                                                                                                       \
            return __riscv_vle32_v_f32##Suffix(Address, vl);                                                    \
        }                                                                                                       \
                                                                                                                \
        static MLAS_FORCEINLINE void Store(float* Address, Vector Value, size_t vl)                             \
        {                                                                                                       \
            __riscv_vse32_v_f32##Suffix(Address, Value, vl);                                                    \
        }                                                                                                       \
                                                                                                                \
        static MLAS_FORCEINLINE Vector Multiply(Vector Value, float Scalar, size_t vl)                          \
        {                                                                                                       \
            return __riscv_vfmul_vf_f32##Suffix(Value, Scalar, vl);                                             \
        }                                                                                                       \
                                                                                                                \
        static MLAS_FORCEINLINE Vector Add(Vector Left, Vector Right, size_t vl)                                \
        {                                                                                                       \
            return __riscv_vfadd_vv_f32##Suffix(Left, Right, vl);                                               \
        }                                                                                                       \
                                                                                                                \
        static MLAS_FORCEINLINE Vector MultiplyAdd(Vector Accumulator, float Scalar, Vector Value, size_t vl)   \
        {                                                                                                       \
            return __riscv_vfmacc_vf_f32##Suffix(Accumulator, Scalar, Value, vl);                               \
        }                                                                                                       \
    }

MLAS_SGEMM_RVV_TRAITS(1, m1);
MLAS_SGEMM_RVV_TRAITS(2, m2);
MLAS_SGEMM_RVV_TRAITS(4, m4);

#undef MLAS_SGEMM_RVV_TRAITS

#if defined(__GNUC__)
#define MLAS_SGEMM_RVV_NOINLINE __attribute__((noinline))
#else
#define MLAS_SGEMM_RVV_NOINLINE
#endif

template <size_t Lmul, bool ZeroMode, bool AlphaIsOne>
MLAS_FORCEINLINE void
MlasStoreAccumulatorRvv(float* C, typename MlasSgemmRvvTraits<Lmul>::Vector Accumulator, size_t vl, float alpha)
{
    using Traits = MlasSgemmRvvTraits<Lmul>;

#if defined(_WIN32)

    if constexpr (AlphaIsOne) {
        UNREFERENCED_PARAMETER(alpha);
    }

#endif

    if constexpr (!AlphaIsOne) {
        Accumulator = Traits::Multiply(Accumulator, alpha, vl);
    }

    if constexpr (!ZeroMode) {
        Accumulator = Traits::Add(Accumulator, Traits::Load(C, vl), vl);
    }

    Traits::Store(C, Accumulator, vl);
}

template <size_t Lmul, bool ZeroMode, bool AlphaIsOne, size_t Rows>
MLAS_FORCEINLINE void
MlasSgemmTileRvv(
    const float* A, const float* B, float* C, size_t CountK, size_t CountN, size_t lda, size_t ldc, float alpha
)

{
    using Traits = MlasSgemmRvvTraits<Lmul>;

    static_assert(Rows >= 1 && Rows <= 4, "unsupported RVV SGEMM tile height");

#if defined(_WIN32)

    if constexpr (Rows == 1) {
        UNREFERENCED_PARAMETER(lda);
        UNREFERENCED_PARAMETER(ldc);
    }

    if constexpr (AlphaIsOne) {
        UNREFERENCED_PARAMETER(alpha);
    }

#endif

    using Vector = typename Traits::Vector;

    size_t remaining_n_block = CountN;
    size_t column_offset = 0;
    float* c = C;

    while (remaining_n_block > 0) {
        const size_t vl = Traits::VectorLength(remaining_n_block);

        Vector row0_block = Traits::Zero(vl);
        Vector row1_block;
        Vector row2_block;
        Vector row3_block;

        if constexpr (Rows >= 2) {
            row1_block = Traits::Zero(vl);
        }
        if constexpr (Rows >= 3) {
            row2_block = Traits::Zero(vl);
        }
        if constexpr (Rows >= 4) {
            row3_block = Traits::Zero(vl);
        }

        const float* a = A;
        const float* b = B + column_offset;
        size_t k = CountK;

        while (k >= 2) {
            const float row0_a0 = a[lda * 0 + 0];
            const float row0_a1 = a[lda * 0 + 1];
            Vector b_elements = Traits::Load(b, vl);
            row0_block = Traits::MultiplyAdd(row0_block, row0_a0, b_elements, vl);

            if constexpr (Rows >= 2) {
                row1_block = Traits::MultiplyAdd(row1_block, a[lda * 1 + 0], b_elements, vl);
            }
            if constexpr (Rows >= 3) {
                row2_block = Traits::MultiplyAdd(row2_block, a[lda * 2 + 0], b_elements, vl);
            }
            if constexpr (Rows >= 4) {
                row3_block = Traits::MultiplyAdd(row3_block, a[lda * 3 + 0], b_elements, vl);
            }

            b_elements = Traits::Load(b + kPackedCountN, vl);
            row0_block = Traits::MultiplyAdd(row0_block, row0_a1, b_elements, vl);

            if constexpr (Rows >= 2) {
                row1_block = Traits::MultiplyAdd(row1_block, a[lda * 1 + 1], b_elements, vl);
            }
            if constexpr (Rows >= 3) {
                row2_block = Traits::MultiplyAdd(row2_block, a[lda * 2 + 1], b_elements, vl);
            }
            if constexpr (Rows >= 4) {
                row3_block = Traits::MultiplyAdd(row3_block, a[lda * 3 + 1], b_elements, vl);
            }

            a += 2;
            b += kPackedCountN * 2;
            k -= 2;
        }

        if (k > 0) {
            Vector b_elements = Traits::Load(b, vl);
            row0_block = Traits::MultiplyAdd(row0_block, a[lda * 0 + 0], b_elements, vl);

            if constexpr (Rows >= 2) {
                row1_block = Traits::MultiplyAdd(row1_block, a[lda * 1 + 0], b_elements, vl);
            }
            if constexpr (Rows >= 3) {
                row2_block = Traits::MultiplyAdd(row2_block, a[lda * 2 + 0], b_elements, vl);
            }
            if constexpr (Rows >= 4) {
                row3_block = Traits::MultiplyAdd(row3_block, a[lda * 3 + 0], b_elements, vl);
            }
        }

        MlasStoreAccumulatorRvv<Lmul, ZeroMode, AlphaIsOne>(c, row0_block, vl, alpha);

        if constexpr (Rows >= 2) {
            MlasStoreAccumulatorRvv<Lmul, ZeroMode, AlphaIsOne>(c + ldc, row1_block, vl, alpha);
        }
        if constexpr (Rows >= 3) {
            MlasStoreAccumulatorRvv<Lmul, ZeroMode, AlphaIsOne>(c + ldc * 2, row2_block, vl, alpha);
        }
        if constexpr (Rows >= 4) {
            MlasStoreAccumulatorRvv<Lmul, ZeroMode, AlphaIsOne>(c + ldc * 3, row3_block, vl, alpha);
        }

        c += vl;
        column_offset += vl;
        remaining_n_block -= vl;
    }
}

template <size_t Lmul, bool ZeroMode, bool AlphaIsOne, size_t Rows>
MLAS_FORCEINLINE size_t
MlasSgemmKernelRvv(
    const float* A, const float* B, float* C, size_t CountK, size_t CountN, size_t lda, size_t ldc, float alpha
)
{
    const float* packed_b_block = B;
    float* c_block = C;
    size_t remaining_n_total = CountN;

    do {
        const size_t count_n_block = remaining_n_total >= kPackedCountN ? kPackedCountN : remaining_n_total;

        MlasSgemmTileRvv<Lmul, ZeroMode, AlphaIsOne, Rows>(
            A, packed_b_block, c_block, CountK, count_n_block, lda, ldc, alpha
        );

        c_block += count_n_block;
        packed_b_block += CountK * kPackedCountN;
        remaining_n_total -= count_n_block;

    } while (remaining_n_total > 0);

    return Rows;
}

template <size_t Lmul, bool ZeroMode, bool AlphaIsOne>
MLAS_FORCEINLINE size_t
MlasGemmFloatKernelRvvDispatchRows(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
)
{
    if (CountM >= 4) {
        return MlasSgemmKernelRvv<Lmul, ZeroMode, AlphaIsOne, 4>(A, B, C, CountK, CountN, lda, ldc, alpha);
    }

    if (CountM == 3) {
        return MlasSgemmKernelRvv<Lmul, ZeroMode, AlphaIsOne, 3>(A, B, C, CountK, CountN, lda, ldc, alpha);
    }

    if (CountM >= 2) {
        return MlasSgemmKernelRvv<Lmul, ZeroMode, AlphaIsOne, 2>(A, B, C, CountK, CountN, lda, ldc, alpha);
    }

    return MlasSgemmKernelRvv<Lmul, ZeroMode, AlphaIsOne, 1>(A, B, C, CountK, CountN, lda, ldc, alpha);
}

template <size_t Lmul, bool ZeroMode>
MLAS_FORCEINLINE size_t
MlasGemmFloatKernelRvvDispatchAlpha(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
)
{
    if (alpha == 1.0f) {
        return MlasGemmFloatKernelRvvDispatchRows<Lmul, ZeroMode, true>(
            A, B, C, CountK, CountM, CountN, lda, ldc, alpha
        );
    }

    return MlasGemmFloatKernelRvvDispatchRows<Lmul, ZeroMode, false>(
        A, B, C, CountK, CountM, CountN, lda, ldc, alpha
    );
}

template <size_t Lmul>
MLAS_FORCEINLINE size_t
MlasGemmFloatKernelRvvImpl(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode
)
{
    if (ZeroMode) {
        return MlasGemmFloatKernelRvvDispatchAlpha<Lmul, true>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
    }

    return MlasGemmFloatKernelRvvDispatchAlpha<Lmul, false>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}

MLAS_SGEMM_RVV_NOINLINE size_t
MlasGemmFloatKernelSingleRowRvv(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode
)
{
    return MlasGemmFloatKernelRvvImpl<4>(A, B, C, CountK, 1, CountN, lda, ldc, alpha, ZeroMode);
}

#define MLAS_SGEMM_RVV_ENTRY(EntryName, Lmul)                                                                          \
    MLAS_SGEMM_RVV_NOINLINE size_t MLASCALL EntryName(                                                                 \
        const float* A, const float* B, float* C, size_t CountK, size_t CountM, size_t CountN, size_t lda, size_t ldc, \
        float alpha, bool ZeroMode                                                                                     \
    )                                                                                                                  \
    {                                                                                                                  \
        if constexpr (Lmul != 4) {                                                                                     \
            if (CountM == 1) {                                                                                          \
                return MlasGemmFloatKernelSingleRowRvv(A, B, C, CountK, CountN, lda, ldc, alpha, ZeroMode);              \
            }                                                                                                          \
        }                                                                                                              \
        return MlasGemmFloatKernelRvvImpl<Lmul>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha, ZeroMode);           \
    }

MLAS_SGEMM_RVV_ENTRY(MlasGemmFloatKernelRvvM1, 1)
MLAS_SGEMM_RVV_ENTRY(MlasGemmFloatKernelRvvM2, 2)
MLAS_SGEMM_RVV_ENTRY(MlasGemmFloatKernelRvvM4, 4)

#undef MLAS_SGEMM_RVV_ENTRY

MLAS_FORCEINLINE size_t
MlasSgemmLmulRvv()
{
    const size_t BlockBytes = kPackedCountN * sizeof(float);
    const size_t RegisterBytes = __riscv_vlenb();

    if (RegisterBytes >= BlockBytes) {
        return 1;
    }
    if (2 * RegisterBytes >= BlockBytes) {
        return 2;
    }
    return 4;
}

}  // namespace

MLAS_GEMM_FLOAT_KERNEL*
MlasGetGemmFloatKernelRvv()
{
    switch (MlasSgemmLmulRvv()) {
        case 1:
            return MlasGemmFloatKernelRvvM1;
        case 2:
            return MlasGemmFloatKernelRvvM2;
        default:
            return MlasGemmFloatKernelRvvM4;
    }
}

size_t MLASCALL
MlasGemmFloatKernelRvv(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode
)
{
    return MlasGetGemmFloatKernelRvv()(A, B, C, CountK, CountM, CountN, lda, ldc, alpha, ZeroMode);
}

#undef MLAS_SGEMM_RVV_NOINLINE
#endif  // defined(MLAS_USE_RVV)
