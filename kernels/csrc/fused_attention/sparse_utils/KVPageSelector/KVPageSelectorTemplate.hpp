#pragma once

#include "../../common/cudaTypeUtils.cuh"
#include "../../common/memoryUtils.h"
#include "../../common/decoderMaskedMultiheadAttentionUtils.h"
#include "../../common/kvCacheUtils.h"
#include "KVPageSelector.h"

#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <assert.h>
#include <float.h>
#include <type_traits>

// Multi-block mmha kernel can only be selected when CUDA >= 11.7
#if (CUDART_VERSION >= 11070)
#define ENABLE_MULTI_BLOCK_OPTION
#endif

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

#ifdef ENABLE_MULTI_BLOCK_OPTION
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/std/bit>
#endif // ENABLE_MULTI_BLOCK_OPTION

// #define MMHA_USE_HMMA_FOR_REDUCTION

#define _MAX_INT 2147000000

// Below are knobs to extend FP32 accumulation for higher FP16 accuracy

// Does not seem to affect the accuracy that much
#define MMHA_USE_FP32_ACUM_FOR_FMA

// Seems to slightly improve the accuracy
#define MMHA_USE_FP32_ACUM_FOR_OUT

#if 0 && defined(MMHA_USE_FP32_ACUM_FOR_OUT)
 // Does not seem to improve the accuracy
 //#define MMHA_USE_FP32_ACUM_FOR_LOGITS
#endif

namespace mmha
{

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    //
    // We use the following terminology to describe the different dimensions.
    //
    // B:  Batch size (number of sequences),
    // L:  Sequence length,
    // D:  Hidden dimension,
    // H:  Number of heads,
    // Dh: Hidden dimension per head - Dh = D / H.
    //
    // The different kernels assign a threadblock for B x H pair. The grid has size (1, B, H). We use
    // 256 threads per block to maximum occupancy and performance.
    //
    // Each threadblock loads Dh values from Q and its associated bias. The kernels run a loop to
    // compute Q * K^T where K is loaded from a cache buffer -- except for the current timestep. The
    // cache buffer helps with memory accesses and contains keys with bias.
    //
    // The layout of the cache buffer for the keys/values is [B, H, L, Dh]
    // where the fastest moving dimension (contiguous data) is the rightmost one.
    // Contiguous threads will read one hidden_dimension per LDG unless we need more than 32 threads.
    //
    // The different kernels use 1 ~ 32 threads per key (THREADS_PER_KEY). The size of the LDGs
    // is always 16bytes (8 bytes for 8bit cache). Each thread sums Dh / THREADS_PER_KEY elements. At
    // the end of each iteration of the Q * K^T loop, we perform a reduction between lanes using an
    // HMMA instruction (Tensor Core). Each Q * K^T value is stored in shared memory in FP32.
    //
    // After that loop, a parallel softmax is computed across the different Q * K^T values stored in
    // shared memory.
    //
    // The kernel ends with a loop over the values in V. We use THREADS_PER_VALUE to control how many
    // timesteps are computed by loop iteration. As with the keys, the values are read from a cache
    // except for the current timestep. The layout of the cache buffer for the values is same as the key,
    // which is [B, H, L, Dh].
    //
    // Note that we have remapped key layout to make sure it shares the same pattern as value [B, H, L, Dh].
    // It helps coalescing memory access, and reducing register pressure.

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T, int Dh_MAX>
    struct Qk_vec_m_
    {
    };

    template <>
    struct Qk_vec_m_<uint16_t, 32>
    {
        using Type = uint32_t;
    };

    template <>
    struct Qk_vec_m_<uint16_t, 64>
    {
        using Type = uint32_t;
    };

    template <>
    struct Qk_vec_m_<uint16_t, 128>
    {
        using Type = uint2;
    };

    template <>
    struct Qk_vec_m_<uint16_t, 256>
    {
        using Type = uint4;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, int Dh>
    struct Qk_vec_k_
    {
        using Type = typename Qk_vec_m_<T, Dh>::Type;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, int V_VEC_SIZE>
    struct V_vec_m_
    {
    };

    template <>
    struct V_vec_m_<uint16_t, 2>
    {
        using Type = uint32_t;
    };

    template <>
    struct V_vec_m_<uint16_t, 4>
    {
        using Type = uint2;
    };

    template <>
    struct V_vec_m_<uint16_t, 8>
    {
        using Type = uint4;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, int V_VEC_SIZE>
    struct V_vec_k_
    {
        using Type = typename V_vec_m_<T, V_VEC_SIZE>::Type;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Reuse V_vec traits as key and value share the same layout.
    template <typename T, int K_VEC_SIZE>
    struct K_vec_m_
    {
        using Type = typename V_vec_m_<T, K_VEC_SIZE>::Type;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, int K_VEC_SIZE>
    struct K_vec_k_
    {
        using Type = typename K_vec_m_<T, K_VEC_SIZE>::Type;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    template <typename T>
    struct Qk_vec_acum_fp32_
    {
    };

    template <>
    struct Qk_vec_acum_fp32_<float>
    {
        using Type = float;
    };

    template <>
    struct Qk_vec_acum_fp32_<float2>
    {
        using Type = float2;
    };

    template <>
    struct Qk_vec_acum_fp32_<float4>
    {
        using Type = float4;
    };

    // template<> struct Qk_vec_acum_fp32_<uint16_t> { using Type = float;        };
    template <>
    struct Qk_vec_acum_fp32_<uint32_t>
    {
        using Type = float2;
    };

    template <>
    struct Qk_vec_acum_fp32_<uint2>
    {
        using Type = Float4_;
    };

    template <>
    struct Qk_vec_acum_fp32_<uint4>
    {
        using Type = Float8_;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct K_vec_acum_fp32_
    {
    };

    template <>
    struct K_vec_acum_fp32_<float>
    {
        using Type = float;
    };

    template <>
    struct K_vec_acum_fp32_<float2>
    {
        using Type = float2;
    };

    template <>
    struct K_vec_acum_fp32_<float4>
    {
        using Type = float4;
    };

    template <>
    struct K_vec_acum_fp32_<Float8_>
    {
        using Type = Float8_;
    };

    template <>
    struct K_vec_acum_fp32_<uint32_t>
    {
        using Type = float2;
    };

    template <>
    struct K_vec_acum_fp32_<uint2>
    {
        using Type = Float4_;
    };

    template <>
    struct K_vec_acum_fp32_<uint4>
    {
        using Type = Float8_;
    };

#endif // MMHA_USE_FP32_ACUM_FOR_FMA

    ////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    template <typename T>
    struct V_vec_acum_fp32_
    {
    };

    template <>
    struct V_vec_acum_fp32_<float>
    {
        using Type = float;
    };

    template <>
    struct V_vec_acum_fp32_<float2>
    {
        using Type = float2;
    };

    template <>
    struct V_vec_acum_fp32_<float4>
    {
        using Type = float4;
    };

    template <>
    struct V_vec_acum_fp32_<uint32_t>
    {
        using Type = float2;
    };

    template <>
    struct V_vec_acum_fp32_<uint2>
    {
        using Type = Float4_;
    };

    template <>
    struct V_vec_acum_fp32_<uint4>
    {
        using Type = Float8_;
    };
#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Tout, typename Tin>
    __inline__ __device__ constexpr Tout vec_conversion(const Tin &x)
    {
        static_assert(std::is_same<Tout, Tin>::value, "Type mismatch");
        return x;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename K_vec, typename T, int N>
    __inline__ __device__ constexpr void vec_ele_wise_max(K_vec &x, const K_vec &y)
    {
        T *x_ptr = reinterpret_cast<T*>(&x);
        const T *y_ptr = reinterpret_cast<const T*>(&y);

        // return the results to x
        for (int ii = 0; ii < N; ++ii)
        {
            x_ptr[ii] = fmaxf(x_ptr[ii], y_ptr[ii]);
        }
    }

    template <typename K_vec, typename T, int N>
    __inline__ __device__ constexpr void vec_ele_wise_min(K_vec &x, const K_vec &y)
    {
        T *x_ptr = reinterpret_cast<T*>(&x);
        const T *y_ptr = reinterpret_cast<const T*>(&y);

        // return the results to x
        for (int ii = 0; ii < N; ++ii)
        {
            x_ptr[ii] = fminf(x_ptr[ii], y_ptr[ii]);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int THREADS_PER_KEY, typename K_vec, int N>
    inline __device__ float qk_dot_(const K_vec (&q)[N], const K_vec (&k)[N])
    {
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
        using K_vec_acum = typename K_vec_acum_fp32_<K_vec>::Type;
#else
        using K_vec_acum = K_vec;
#endif
        // Compute the parallel products for Q*K^T (treat vector lanes separately).
        K_vec_acum qk_vec = mul<K_vec_acum, K_vec, K_vec>(q[0], k[0]);
#pragma unroll
        for (int ii = 1; ii < N; ++ii)
        {
            qk_vec = fma(q[ii], k[ii], qk_vec);
        }

        // Finalize the reduction across lanes.
        float qk = sum(qk_vec);
#pragma unroll
        for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2)
        {
            qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
        }
        return qk;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, int THREADS_PER_KEY>
    struct Qk_dot
    {
        template <typename K_vec, int N>
        static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N])
        {
            return qk_dot_<THREADS_PER_KEY>(q, k);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float4 hmma_fp32(const uint2 &a, uint32_t b)
    {
        float4 c;
        float zero = 0.f;
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
            "    {%0, %1, %2, %3}, \n"
            "    {%4, %5}, \n"
            "    {%6}, \n"
            "    {%7, %7, %7, %7}; \n"

            : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w)
            : "r"(a.x) "r"(a.y), "r"(b), "f"(zero));
        return c;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int N>
    inline __device__ float qk_hmma_dot_(const uint32_t (&q)[N], const uint32_t (&k)[N])
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
        using K_vec_acum = typename K_vec_acum_fp32_<uint32_t>::Type;
#else
        using K_vec_acum = uint32_t;
#endif
        K_vec_acum qk_vec = mul<K_vec_acum, uint32_t, uint32_t>(q[0], k[0]);
#pragma unroll
        for (int ii = 1; ii < N; ++ii)
        {
            qk_vec = fma(q[ii], k[ii], qk_vec);
        }
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
        uint32_t qk_vec_ = float2_to_half2(qk_vec);
        return hmma_fp32(make_uint2(qk_vec_, 0u), 0x3c003c00u).x;
#else
        return hmma_fp32(make_uint2(qk_vec, 0u), 0x3c003c00u).x;
#endif
#else
        return 0.f;
#endif
    }


    template <int THREADS_PER_KEY, typename K_vec_k>
    inline __device__ float qk_hmma_dot_simple(const K_vec_k& q, const K_vec_k& k);

    template <int THREADS_PER_KEY>
    inline __device__ float qk_hmma_dot_simple(const uint32_t& q, const uint32_t& k)
    {
        assert (0);
    }

    template <int THREADS_PER_KEY>
    inline __device__ float qk_hmma_dot_simple(const uint2& q, const uint2& k)
    {
        assert (0);
    }

    template <int THREADS_PER_KEY>
    inline __device__ float qk_hmma_dot_simple(const uint4& q, const uint4& k)
    {
        using K_vec_acum = uint32_t;
        K_vec_acum qk_vec = mul<K_vec_acum, uint32_t, uint32_t>(q.x, k.x);
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(qk_vec) : "r"(q.y), "r"(k.y), "r"(qk_vec));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(qk_vec) : "r"(q.z), "r"(k.z), "r"(qk_vec));
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(qk_vec) : "r"(q.w), "r"(k.w), "r"(qk_vec));
        // return hmma_fp32(make_uint2(qk_vec, 0u), 0x3c003c00u).x;
        half2 qk_vec_h = (half2 &)qk_vec;
        float qk = __half2float(__hadd(qk_vec_h.x, qk_vec_h.y));
#pragma unroll
        for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2)
        {
            qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
        }
        return qk;
    }

    template <int THREADS_PER_KEY>
    inline __device__ float qk_hmma_dot_min_max(const uint4& q, const uint4& k_max, const uint4& k_min)
    {
        // NOTE (Shang): uint4 contains 8 half numbers.
        half2 *q_ptr = (half2*)(&q);
        half2 *k_max_ptr = (half2*)(&k_max);
        half2 *k_min_ptr = (half2*)(&k_min);

        half2 acc_max = __hmul2(q_ptr[0], k_max_ptr[0]);
        half2 acc_min = __hmul2(q_ptr[0], k_min_ptr[0]);
        half2 acc = __hmax2(acc_max, acc_min);
#pragma unroll
        for (int ii = 1; ii < 4; ++ii)
        {
            acc_max = __hmul2(q_ptr[ii], k_max_ptr[ii]);
            acc_min = __hmul2(q_ptr[ii], k_min_ptr[ii]);
            acc = __hadd2(acc, __hmax2(acc_max, acc_min));
        }
        float qk_min_max = __half2float(__hadd(acc.x, acc.y));

#pragma unroll
        for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2)
        {
            qk_min_max += __shfl_xor_sync(uint32_t(-1), qk_min_max, mask);
        }

        return qk_min_max;
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    struct Qk_dot<uint16_t, 4>
    {
        template <typename K_vec, int N>
        static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N])
        {
            return qk_dot_<4>(q, k);
        }

        template <int N>
        static inline __device__ float dot(const uint32_t (&q)[N], const uint32_t (&k)[N])
        {
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA_FOR_REDUCTION)
            return qk_hmma_dot_(q, k);
#else
            return qk_dot_<4>(q, k);
#endif // defined MMHA_USE_HMMA_FOR_REDUCTION
        }
    };


    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int WARPS_PER_BLOCK, int WARP_SIZE = 32>
    inline __device__ float block_sum(float *red_smem, float sum)
    {

        // Decompose the thread index into warp / lane.
        int warp = threadIdx.x / WARP_SIZE;
        int lane = threadIdx.x % WARP_SIZE;

// Compute the sum per warp.
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
        {
            sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
        }

        // Warp leaders store the data to shared memory.
        if (lane == 0)
        {
            red_smem[warp] = sum;
        }

        // Make sure the data is in shared memory.
        __syncthreads();

        // The warps compute the final sums.
        if (lane < WARPS_PER_BLOCK)
        {
            sum = red_smem[lane];
        }

// Parallel reduction inside the warp.
#pragma unroll
        for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2)
        {
            sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
        }

        // Broadcast to other threads.
        return __shfl_sync(uint32_t(-1), sum, 0);
    }

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float cast_to_float(float u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 cast_to_float(float2 u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float4 cast_to_float(float4 u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float4_ cast_to_float(Float4_ u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float8_ cast_to_float(Float8_ u)
    {
        return u;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ float2 cast_to_float(uint32_t u)
    {
        return half2_to_float2(u);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float4_ cast_to_float(uint2 u)
    {
        Float4_ tmp;
        tmp.x = half2_to_float2(u.x);
        tmp.y = half2_to_float2(u.y);
        return tmp;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    inline __device__ Float8_ cast_to_float(uint4 u)
    {
        Float8_ tmp;
        tmp.x = half2_to_float2(u.x);
        tmp.y = half2_to_float2(u.y);
        tmp.z = half2_to_float2(u.z);
        tmp.w = half2_to_float2(u.w);
        return tmp;
    }

#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline __device__ __host__ T divUp(T m, T n)
    {
        return (m + n - 1) / n;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline __device__ __host__ T div(T m, T n)
    {
        return m / n;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct kernel_type_t
    {
        using Type = T;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Compute the largest supported head size (dh_max). It must be the smallest power-of-2 that is not strictly smaller
    // than the head size (dh).
    inline __device__ __host__ constexpr unsigned dh_max(unsigned dh)
    {
        return next_power_of_two(const_max(dh, 32u));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    inline __device__ __host__ constexpr unsigned threads_per_value(unsigned dh_max)
    {
        // add by JXGuo: 16bytes is 128 bits, which is the maximum number of bits that can be loaded in a single LDG.
        return dh_max * sizeof(T) / 16;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, unsigned Dh_MAX>
    inline __device__ __host__ constexpr unsigned threads_per_key()
    {
        // Since we want to perform the reduction entirely within a warp, the number of threads per key
        // is capped at 32.
        constexpr unsigned threads = (unsigned)(Dh_MAX * sizeof(T) / 16u);
        if ((threads & (threads - 1)) != 0)
        {
            assert(false); // Not a power of two.
        }
        return std::min(32u, threads);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T, typename T_VEC, unsigned VECS_PER_CHUNK> // add by JXGuo: VECS_PER_CHUNK = THREADS_PER_KEY, thus idx_chunk is the index of key
    __device__ inline constexpr uint2 chunk_index(unsigned tidx)
    {
        // The chunk associated with the thread.
        auto const idx_chunk = tidx / VECS_PER_CHUNK;

        // The position of the T_VEC vector in that chunk associated with the thread.
        static_assert(sizeof(T_VEC) % sizeof(T) == 0);
        unsigned constexpr kVecSize{sizeof(T_VEC) / sizeof(T)};
        auto const idx_vec = (tidx % VECS_PER_CHUNK) * kVecSize;

        return uint2{idx_chunk, idx_vec};
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////
    __inline__ __device__ uint32_t cast_smem_ptr_to_uint_helper(void const *const ptr)
    {
        uint32_t smem_int_ptr;

        asm("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, "
            "smem_ptr; }\n"
            : "=r"(smem_int_ptr)
            : "l"(ptr));

        return smem_int_ptr;
    }

    __inline__ __device__ void
    cp_async_helper(uint32_t smem_int_ptr, const uint4 *__restrict__ src, bool mask)
    {
        const int cp_size = 16;
        // cachehint will not impact performance.
        // clang-format off
        asm volatile("{"
                        "  .reg .pred p;"
                        "  setp.ne.b32 p, %0, 0;"
                        "  @p cp.async.cg.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;"
                        "}" ::"r"((int)mask),
                        "r"(smem_int_ptr),
                        "l"(src),
                        "n"(cp_size));
        // clang-format on
    }


    __inline__ __device__ void
    cp_async_launch(void *dst_ptr, const uint4 *__restrict__ src_ptr, bool mask)
    {
       uint32_t addr = cast_smem_ptr_to_uint_helper(dst_ptr);
       cp_async_helper(addr, src_ptr, mask);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////


    template <
        // The type of the inputs. Supported types: float, uint16_t, nv_bfloat16.
        typename T,
        // The type of the cache.
        typename Tcache,
        // Type of struct containing KV cache
        typename KVCacheBuffer,
        // The hidden dimension per head.
        unsigned Dh,
        // The number of threads in a threadblock.
        unsigned THREADS_PER_BLOCK,

        bool IS_RETRIEVAL_HEAD, 
        // Whether enable multi-block mode for long-sequence-length.
        bool DO_MULTI_BLOCK = false,
        bool DO_DYNAMIC_SPARSE = false,
        // Whether use INT4KV
        bool INT4KV = false,
        bool KV_WITH_ZEROS = false,
        bool SMEM_PRELOAD = false,
        // The number of threads per key.
        unsigned THREADS_PER_KEY = mmha::threads_per_key<T, dh_max(Dh)>(),
        // The number of threads per value.
        unsigned THREADS_PER_VALUE = mmha::threads_per_value<T>(dh_max(Dh)),
        // The unroll factor for loading from K cache.
        // unsigned K_LOOP_UNROLL = 8, // 8,
        // The unroll factor for loading from V cache.
        // Set it default to 4 for higher occupancy (by reducing registers usage).
        unsigned V_LOOP_UNROLL = 4>
    inline __device__ void masked_multihead_attention_page_selector_kernel(
        Multihead_attention_page_selector_params<T> params, KVCacheBuffer kvCacheBuffer, const int head_rank){

        assert(DO_MULTI_BLOCK == false);    // NOTE (Shang): We do not use multi-block mode for page selection now.

        // Question: Do we need MULTI_BLOCK for the selector? -- Maybe. Better parallelism.
        // NOTE (Shang): If streaming head. We can either directly return or fill the idxes with -1.
        if constexpr (!IS_RETRIEVAL_HEAD)
        {
            return; // NO need to do anything with streaming heads.
        }


        const int num_head_kv_buffer = IS_RETRIEVAL_HEAD ? params.num_retrieval_kv_heads : params.num_streaming_kv_heads;
        const int tokens_per_block = params.tokens_per_block;

        constexpr unsigned K_LOOP_UNROLL = SMEM_PRELOAD ? 8 : 4;
        using Tk = typename kernel_type_t<T>::Type;

        static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;
        static constexpr bool ENABLE_4BITS_CACHE = (INT4KV && ENABLE_8BITS_CACHE);
        static constexpr bool ENABLE_ZEROS = KV_WITH_ZEROS;

        // The size of a warp.
        constexpr unsigned WARP_SIZE{32};
        // The number of warps in a threadblock.
        constexpr unsigned WARPS_PER_BLOCK{THREADS_PER_BLOCK / WARP_SIZE};

        // The maximum hidden size per head.
        constexpr auto Dh_MAX = dh_max(Dh);
        constexpr bool IS_Dh_MAX = Dh == Dh_MAX;
        static_assert(Dh_MAX >= WARP_SIZE);
        static_assert(Dh_MAX >= Dh);

        // The maximum sequence length in the kv_cache, i.e., an upper bound on L.
        // Note that the maximum sequence length supported by the model might be greater than this.
        const auto max_seq_len = static_cast<unsigned>(params.memory_max_len);
        assert(max_seq_len > 0);
        // The current timestep (including paddings).
        // It is only used to calculate the smem stride.
        const auto timestep = static_cast<unsigned>(DO_MULTI_BLOCK ? params.timesteps_per_block : IS_RETRIEVAL_HEAD ? params.timestep : kvCacheBuffer.sinkTokenLen + kvCacheBuffer.localTokenLen);

#ifdef ENABLE_MULTI_BLOCK_OPTION
        constexpr bool MULTI_BLOCK_FLAG = DO_MULTI_BLOCK;
#else
        constexpr bool MULTI_BLOCK_FLAG = false;
#endif

        // Use smem_size_in_bytes (host, launch config) to allocate dynamic shared memory for this buffer.
        extern __shared__ char smem_[];

        // The shared memory for the Q*K^T values and partial logits in softmax.
        // auto qk_smem = reinterpret_cast<float *>(smem_);

        __shared__ float qk_current_smem[1];

        __shared__ Tk logits_current_smem[1];

        // The shared memory to do the final reduction for the output values. Reuse qk_smem.
        // Tk *out_smem = reinterpret_cast<Tk *>(smem_);

        // The shared memory buffers for the block-wide reductions. One for max, one for sum.
        __shared__ float red_smem[WARPS_PER_BLOCK * 2];

        // A vector of Q or K elements for the current timestep.
        using Qk_vec_m = typename Qk_vec_m_<T, Dh_MAX>::Type; // with memory-used precision
        using Qk_vec_k = typename Qk_vec_k_<T, Dh_MAX>::Type; // with kernel-used precision

        // Make sure the hidden dimension per head is a multiple of the number of threads per key.
        static_assert(Dh_MAX % THREADS_PER_KEY == 0); // trivially satisfied since THREADS_PER_KEY in {1, 2, 4}

        // The number of elements per vector.
        // Each thread will handle 16 bytes.
        constexpr int K_VEC_SIZE = 16u / sizeof(T);
        // Make sure the hidden size per head is a multiple of the vector size.
        static_assert(Dh_MAX % K_VEC_SIZE == 0);
        // The type of queries and keys for the math in the Q*K^T product.
        using K_vec_k = typename K_vec_k_<T, K_VEC_SIZE>::Type;
        // Only used when key cache is quantized to 4 or 8 bits.
        constexpr int K_VEC_M_SIZE = K_VEC_SIZE / (ENABLE_4BITS_CACHE ? 2 : 1);
        using K_vec_m = typename packed_type<Tcache, K_VEC_M_SIZE>::type;

        // Use alignment for safely casting the shared buffers as Qk_vec_k and K_vec_k.
        // Shared memory to store Q inputs.
        __shared__ __align__(const_max(sizeof(Qk_vec_k), sizeof(K_vec_k))) Tk q_smem[Dh_MAX];

        // Make sure the hidden dimension per head is a multiple of the number of threads per value.
        static_assert(Dh_MAX % THREADS_PER_VALUE == 0); // trivially satisfied since THREADS_PER_VALUE == Dh_MAX / p


        // The number of elements per vector.
        constexpr unsigned QK_VEC_SIZE{sizeof(Qk_vec_m) / sizeof(T)};
        // Make sure the hidden size per head is a multiple of the vector size.
        static_assert(Dh_MAX % QK_VEC_SIZE == 0);
        // We will use block wide reduction if needed
        // The number of vectors per Dh_MAX.
        constexpr unsigned QK_VECS_PER_Dh_MAX{Dh_MAX / QK_VEC_SIZE};
        static_assert(THREADS_PER_BLOCK >= QK_VECS_PER_Dh_MAX);

        // The batch/beam idx
        const auto bi = blockIdx.y;
        // half *k_scale_quant_orig_ptr = params.k_scale_quant_orig[bi];
        // half *v_scale_quant_orig_ptr = params.v_scale_quant_orig[bi];
        if (params.finished != nullptr && params.finished[bi])
        {
            return;
        }
        // The head.
        const unsigned hi{blockIdx.x};
        // The head index of keys and values adjusted for MQA/GQA.
        const int qhead_per_kv{params.num_heads / params.num_kv_heads};
        const unsigned hi_kv{hi / qhead_per_kv};
        // The number of heads.
        const auto num_heads = static_cast<unsigned>(params.num_heads);
        // The number of heads for keys and values adjusted for MQA/GQA.
        const auto num_heads_kv = static_cast<unsigned>(params.num_kv_heads);

        // const auto dynamic_sparse_page_idxes_base_ptr = DO_DYNAMIC_SPARSE ? (params.dynamic_sparse_page_idxes_ptr + ((bi * num_heads) + hi) * params.num_dynamic_sparse_pages) : nullptr;
        // printf("Local dynamic_sparse page idxes ptr: %d\n", *dynamic_sparse_page_idxes_base_ptr);




        // The thread in the block.
        const unsigned tidx{threadIdx.x};

        // The column tile along L dimension on K^T -- noted as T_c in flash-attention paper
        // const unsigned c_tile{0}; // const unsigned c_tile{MULTI_BLOCK_FLAG ? blockIdx.z : 0};
        const unsigned c_tile{MULTI_BLOCK_FLAG ? blockIdx.z : 0};
        if (!IS_RETRIEVAL_HEAD && blockIdx.z != 0)
        {
            return;
        }

        // Indicate if we need to compute the K/V cache element (add KV bias, IA3, RoPE, etc.) and update the cache.
        // For Self-Attention, it's always required.
        // For Cross-Attention, as everything is pre-computed,
        // in the context phase of the encoder, it's not needed in that kernel.
        // Therefore, handle_kv is !DO_CROSS_ATTENTION and irrelevant of timestep.
        // const bool handle_kv = true;  // const bool handle_kv{!DO_CROSS_ATTENTION};
        const bool handle_kv = false;  // const bool handle_kv{!DO_CROSS_ATTENTION};

        // While doing the product Q*K^T for the different keys we track the max.
        float qk_max = -FLT_MAX;

        float qk = 0.0F;

        // Compute relative attention bias on the fly, with relative attention table [head_num/TP, num_buckets] passed in.
        // num_buckets passed as params.relative_attention_bias_stride, max_distance passed as params.max_distance
        bool implicit_rel_attn_bias = params.max_distance != 0;
        int relative_attention_bias_stride = params.relative_attention_bias_stride; // num_buckets might be modified below, save it beforehand
        int max_distance = params.max_distance;

        // The actual sequence length excluding the paddings.
        // minus 1 because it includes the current timestep while tlength denotes the kv cache length.
        // const int tlength = DO_CROSS_ATTENTION
        //                         ? params.memory_length_per_sample[bi] - 1
        //                         : (params.length_per_sample ? (params.length_per_sample[bi] - 1) : static_cast<int>(timestep));
        const int tlength = (params.length_per_sample ? (params.length_per_sample[bi] - 1) : static_cast<int>(params.timestep));
        // The context length for beam searching optimization (all points to beam 0).
        const int input_length = params.input_lengths[bi];

        // The offset in the Q and K buffer also accounts for the batch.
        const auto qk_vec_idx = tidx * QK_VEC_SIZE;
        const auto is_valid_qk_vec = qk_vec_idx < Dh;            

        // const bool load_qkv_quant = params.qkv_scale_quant_orig != nullptr;
        const bool write_attention_quant = params.attention_out_scale_orig_quant != nullptr;

        // Quant/Dequant scales for 8bits kv cache.
        using T_scale = typename kv_cache_scale_type_t<T, Tcache>::Type;
        // T_scale kv_scale_quant_orig[2];
        // T_scale kv_scale_orig_quant[2];

        constexpr int MAX_TIMESTEP_SCALES = SMEM_PRELOAD ? 2048 : 1;
        __shared__ half k_scales_history_smem[MAX_TIMESTEP_SCALES], k_zeros_history_smem[MAX_TIMESTEP_SCALES], v_scales_history_smem[MAX_TIMESTEP_SCALES], v_zeros_history_smem[MAX_TIMESTEP_SCALES];
        
        
        // Up to QK_VECS_PER_Dh_MAX threads load Q and K + the bias values for the current timestep.
        // Trigger the loads from the Q and K buffers.
        Qk_vec_k q; //, k; //, q_bias, k_bias;
        zero(q);
        // zero(k);
        // zero(q_bias);
        // zero(k_bias);
        float rotary_embedding_base = params.rotary_embedding_base;
        float rotary_embedding_scale = params.rotary_embedding_scale;
        if (is_valid_qk_vec)
        {
            update_rotary_base_n_scale(rotary_embedding_base, rotary_embedding_scale,
                                       params.rotary_embedding_scale_type, params.rotary_embedding_dim, params.rotary_embedding_max_positions,
                                       tlength);
            // Query
            // The stride between tokens. We may be able to always use params.stride.
            uint32_t q_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads * Dh);
            // The offset.
            const auto q_offset = flat_index_strided3(bi, hi, qk_vec_idx, q_stride, Dh);

            // Note (shang): Load the current qk here. Not the quantized kv cache.
            {
                // Removed a branch for load_qkv_quant (current step qkv)
                q = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m *>(&params.q[q_offset]));
            }
            // {
                // Removed DO_CROSS_ATTENTION branch
                // Key
                // The stride between tokens. We may be able to always use params.stride.
                // uint32_t k_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
                // // The offset.
                // const auto k_offset = flat_index_strided3(bi, hi_kv, qk_vec_idx, k_stride, Dh);
                // {
                //     // Removed a branch for load_qkv_quant (current step qkv)
                //     k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m *>(&params.k[k_offset]));
                // }
            // }
        }

        // const bool do_ia3 = handle_kv && params.ia3_tasks != nullptr;
        const auto beam_width = static_cast<unsigned>(params.beam_width);
        {
            const bool do_rotary = is_valid_qk_vec && QK_VEC_SIZE * tidx < params.rotary_embedding_dim;

            T *q_smem_ = reinterpret_cast<T *>(smem_);
            // T *k_smem = q_smem_ + params.rotary_embedding_dim;

            const int half_rotary_dim = params.rotary_embedding_dim / 2;
            const int half_idx = qk_vec_idx / half_rotary_dim;
            const int intra_half_idx = qk_vec_idx % half_rotary_dim;
            const int smem_pitch = half_rotary_dim; 

            assert(half_rotary_dim % QK_VEC_SIZE == 0);

            if (do_rotary)
            {
                *reinterpret_cast<Qk_vec_k *>(q_smem_ + half_idx * smem_pitch + intra_half_idx) = q;
                // if (handle_kv)
                // {
                //     *reinterpret_cast<Qk_vec_k *>(k_smem + half_idx * smem_pitch + intra_half_idx) = k;
                // }
            }

            __syncthreads();

            const int transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
            constexpr int tidx_factor = (QK_VEC_SIZE > 1) ? QK_VEC_SIZE / 2 : 1;
            if (do_rotary)
            {
                vec_from_smem_transpose(q, q_smem_, transpose_idx, smem_pitch);
                // if (handle_kv)
                // {
                //     vec_from_smem_transpose(k, k_smem, transpose_idx, smem_pitch);

                //     apply_rotary_embedding(q, k, transpose_idx / tidx_factor, params.rotary_embedding_dim,
                //                            rotary_embedding_base, rotary_embedding_scale, tlength);

                //     write_smem_transpose(k, k_smem, transpose_idx, smem_pitch);
                // }
                // else
                {
                    apply_rotary_embedding(q, transpose_idx / tidx_factor, params.rotary_embedding_dim,
                                           rotary_embedding_base, rotary_embedding_scale, tlength);
                }
                write_smem_transpose(q, q_smem_, transpose_idx, smem_pitch);
            }

            __syncthreads();

            if (do_rotary)
            {
                q = *reinterpret_cast<Qk_vec_k *>(q_smem_ + half_idx * smem_pitch + intra_half_idx);
                // if (handle_kv)
                // {
                //     k = *reinterpret_cast<Qk_vec_k *>(k_smem + half_idx * smem_pitch + intra_half_idx);
                // }
            }

            __syncthreads();
        }



        // For the same reason as handle_kv, no compute needed in Cross-Attention's 1st step
        if (qk_vec_idx < Dh_MAX)
        {

            // Store the Q values to shared memory.
            // Set padded Dh to 0 for the correctness of QK (when Dh != Dh_Max).
            Qk_vec_k zero_q;
            zero(zero_q);

            *reinterpret_cast<Qk_vec_k *>(&q_smem[qk_vec_idx]) = is_valid_qk_vec ? q : zero_q;
        }

        // Make sure the data is in shared memory.
        __syncthreads();

        constexpr unsigned K_ELTS_PER_CHUNK{THREADS_PER_KEY * K_VEC_SIZE};

        // The positions of the cache buffer (for this B * H) and the vector within that chunk associated with this
        // thread.
        const auto k_idx = chunk_index<T, K_vec_k, THREADS_PER_KEY>(tidx);

        // The number of vectors per thread.
        constexpr unsigned K_VECS_PER_THREAD{Dh_MAX / K_ELTS_PER_CHUNK};
        static_assert(Dh_MAX == K_ELTS_PER_CHUNK * K_VECS_PER_THREAD);

        // Load the Q values from shared memory. The values are reused during the loop on K.
        K_vec_k q_vec[K_VECS_PER_THREAD];
        // if constexpr (ENABLE_4BITS_CACHE && ENABLE_ZEROS)
        // {
        //     #pragma unroll
        //     for (unsigned ii = 0; ii < K_VECS_PER_THREAD; ++ii)
        //     {
        //         q_vec[ii] = reorder_8xfp16(*reinterpret_cast<const K_vec_k *>(      // NOTE (Shang): do we really need to reorder here? for the page selector?
        //             &q_smem[flat_index2(ii, k_idx.y, K_ELTS_PER_CHUNK)]));
        //     }
        // }
        // else
        {
            #pragma unroll
            for (unsigned ii = 0; ii < K_VECS_PER_THREAD; ++ii)
            {
                q_vec[ii] = *reinterpret_cast<const K_vec_k *>(
                    &q_smem[flat_index2(ii, k_idx.y, K_ELTS_PER_CHUNK)]);
            }
        }
        // The number of timesteps loaded per iteration, i.e., (THREADS_PER_BLOCK * THREADS_PER_BLOCK) / 256 <= 256
        constexpr unsigned K_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_KEY};
        // The number of keys per warp.
        constexpr unsigned K_PER_WARP{WARP_SIZE / THREADS_PER_KEY};
        // The number of unrolled keys per warp.
        constexpr unsigned UNROLLED_K_PER_WARP = K_PER_WARP * K_LOOP_UNROLL;
        // The number of unrolled keys per ieration.
        constexpr unsigned UNROLLED_K_PER_ITER = K_PER_ITER * K_LOOP_UNROLL;

        // Base pointer for the row of pointers to k cache blocks
        void **k_cache_base_row_ptr = reinterpret_cast<void **>(kvCacheBuffer.getRowPtr(KVIdxType::K_IDX, bi));

        const auto timesteps_per_block = static_cast<unsigned>(params.timesteps_per_block);

        // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
        // Take all previous cache as context when we have no beam searching in order to batch as many LDGs as possible.
        
        const int context_length = tlength; // const int context_length = HAS_BEAMS ? input_length : tlength;

        const int n_sub_chunks = (context_length + kvCacheBuffer.tokensPerSubChunk - 1) / kvCacheBuffer.tokensPerSubChunk;
        const int padded_n_sub_chunks = (n_sub_chunks + kvCacheBuffer.SubChunkGroupSize - 1) / kvCacheBuffer.SubChunkGroupSize * kvCacheBuffer.SubChunkGroupSize;
        
        half *out_stats_ptr = (half*)(params.out) + ((bi * params.num_heads) + hi) * padded_n_sub_chunks; // NOTE (Shang): hoisting +logic_sub_chunk_now * params.num_heads;
            
        // const int block_valid_context_length = MULTI_BLOCK_FLAG ? timesteps_per_block : global_valid_context_length;
        const auto valid_n_sub_chunks_end = divUp(static_cast<unsigned>(n_sub_chunks), UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP;

        // // if (!MULTI_BLOCK_FLAG)
        // {
        //     qk_smem[block_valid_context_length] = qk_current_smem[0];
        // }

        // const int sink_local_gap = IS_RETRIEVAL_HEAD ? 0 : context_length - global_valid_context_length;
        const int sink_local_gap = 0;
        const int sink_end_idx = kvCacheBuffer.sinkTokenLen;
        const int local_start_idx = IS_RETRIEVAL_HEAD ? 0 : context_length - kvCacheBuffer.localTokenLen;   // NOTE (Shang): Removed + 1 here. Need to verify later.
        const int local_end_idx = context_length;


        // The generation ti_end.
        // const auto generation_ti_end = MULTI_BLOCK_FLAG ? divUp(timesteps_per_block, K_PER_WARP) * K_PER_WARP
        //                                                 : divUp(static_cast<unsigned>(tlength), K_PER_WARP) * K_PER_WARP;
        // const auto generation_ti_end = divUp(static_cast<unsigned>(tlength), K_PER_WARP) * K_PER_WARP;

        // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
        const auto bi_seq_len_offset = static_cast<std::size_t>(bi) * max_seq_len;

        const auto c_tile_times_timesteps_per_block = c_tile * params.timesteps_per_block; // 0 if !MULTI_BLOCK_FLAG
        const auto c_tile_times_timesteps_per_block_logic = c_tile * params.timesteps_per_block_logic;

        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        // {
        //     printf("params.timesteps_per_block_logic: %d\n", params.timesteps_per_block_logic);
        // }

        ////////////////////////////////////////////////////////////////////////////////////////////////
        // Key cache loops for dot(Q, K).

        // Handle only context key cache with beam searching.
        // Handle both context and generation key cache without beam searching.
        // Explict batching of LDGs (by K_LOOP_UNROLL) as it doesn't depend on indirection tables.

        // Now we will use kscales, kzeros, etc. so we need pipeline wait prior
        // if constexpr (SMEM_PRELOAD)
        // {
        //     __pipeline_wait_prior(0);
        // }
        // for (int ti = k_idx.x; ti < valid_context_ti_end; ti += UNROLLED_K_PER_ITER)
        for (int ti = k_idx.x; ti < valid_n_sub_chunks_end; ti += UNROLLED_K_PER_ITER)    // k_idx is chunked by (THREADS_PER_KEY, -1). We can reuse it here for the min-max keys.
        {
            // const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;
            const int physic_sub_chunk_base = ti;
            const int logic_sub_chunk_base = ti; // NOTE (Shang): Ignore multiblock for now.
            // const int logic_sub_chunk_base = ti + c_tile_times_timesteps_per_block; // add by JXGuo: c_tile_times_timesteps_per_block is 0 if !MULTI_BLOCK_FLAG

            // The keys loaded from the key cache.
            K_vec_m k_vec_cache[K_LOOP_UNROLL][K_VECS_PER_THREAD];
            // Qk_vec_m k_vec_stats_max[K_LOOP_UNROLL][K_VECS_PER_THREAD];
            // Qk_vec_m k_vec_stats_min[K_LOOP_UNROLL][K_VECS_PER_THREAD];
            K_vec_k k_vec_stats_max[K_LOOP_UNROLL][K_VECS_PER_THREAD];
            K_vec_k k_vec_stats_min[K_LOOP_UNROLL][K_VECS_PER_THREAD];

            // float k_scale_quant_orig_local[K_LOOP_UNROLL];
            // float k_zeros_local[K_LOOP_UNROLL];
#pragma unroll
            for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
            {
                // Haotian: we probably do not need this because each page also contains slots for OOB tokens
                // const int valid_time_now = min(time_now + k_loop * K_PER_ITER, context_length - 1);

                const int _logic_sub_chunk_now = logic_sub_chunk_base + k_loop * K_PER_ITER;
                const int logic_sub_chunk_now = min(_logic_sub_chunk_now, n_sub_chunks - 1);    // -1 because this is the idx
                const int logic_time_now = logic_sub_chunk_now * kvCacheBuffer.tokensPerSubChunk;

                // const int seqIdx = bi / beam_width * beam_width;
                const int seqIdx = bi;
                
                // Base pointer to k cache block fo r beam's batch
                Tcache *k_cache_batch = reinterpret_cast<Tcache *>(kvCacheBuffer.getKBlockPtr(seqIdx, logic_time_now));

                half *k_cache_stats_max_ptr = reinterpret_cast<half *>(k_cache_batch + kvCacheBuffer.mBytesPerSeq) + kvCacheBuffer.mTokensPerBlock * num_head_kv_buffer * (ENABLE_ZEROS? 2 : 1);
                half *k_cache_stats_min_ptr = k_cache_stats_max_ptr + kvCacheBuffer.SubChunkGroupSize * kvCacheBuffer.mElesPerIndicator;

                // int sub_chunk_idx = (tlength % kvCacheBuffer.mTokensPerBlock) / kvCacheBuffer.tokensPerSubChunk;
                const int sub_chunk_idx = logic_sub_chunk_now % kvCacheBuffer.SubChunkGroupSize;
                half *k_cache_stats_max_ptr_local = k_cache_stats_max_ptr + sub_chunk_idx * kvCacheBuffer.mElesPerIndicator + head_rank * Dh;
                half *k_cache_stats_min_ptr_local = k_cache_stats_min_ptr + sub_chunk_idx * kvCacheBuffer.mElesPerIndicator + head_rank * Dh;
                
#pragma unroll
                for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
                {
                    // Make sure we read data within the bound.
                    // Dh OOB values will be handled by zero_q.
                    // Seq OOB values will be masked out when storing back to smem.
                    auto const jj = min(k_idx.y + k_vec_i * K_ELTS_PER_CHUNK, Dh - K_VEC_SIZE);     // Channel-wise iteration

                    // k_vec_stats_max[k_loop][k_vec_i] = *reinterpret_cast<const Qk_vec_m *>(&k_cache_stats_max_ptr_local[jj]);
                    // k_vec_stats_min[k_loop][k_vec_i] = *reinterpret_cast<const Qk_vec_m *>(&k_cache_stats_min_ptr_local[jj]);
                    k_vec_stats_max[k_loop][k_vec_i] = *reinterpret_cast<const K_vec_k *>(&k_cache_stats_max_ptr_local[jj]);
                    k_vec_stats_min[k_loop][k_vec_i] = *reinterpret_cast<const K_vec_k *>(&k_cache_stats_min_ptr_local[jj]);
                }
            }

            __syncthreads();

#pragma unroll
            for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
            {

                const int _logic_sub_chunk_now = logic_sub_chunk_base + k_loop * K_PER_ITER;
                const int logic_sub_chunk_now = _logic_sub_chunk_now;
                const int logic_time_now = logic_sub_chunk_now * kvCacheBuffer.tokensPerSubChunk;

                const int physic_sub_chunk_now = physic_sub_chunk_base + k_loop * K_PER_ITER;

                assert (K_VECS_PER_THREAD == 1);
                // NOTE (Shang): Since K_VECS_PER_THREAD == 1, we omit the loop of k_vec_i here.

                // float qk_ = qk_hmma_dot_simple<THREADS_PER_KEY>(q_vec[0], k_vec[0]) * params.inv_sqrt_dh;
                // float qk_max = qk_hmma_dot_simple<THREADS_PER_KEY>(q_vec[0], k_vec_stats_max[k_loop][0]);   
                // float qk_min = qk_hmma_dot_simple<THREADS_PER_KEY>(q_vec[0], k_vec_stats_min[k_loop][0]);  // If only one representative, we can use qk_hmma_dot_simple for calculation
                float qk_min_max = qk_hmma_dot_min_max<THREADS_PER_KEY>(q_vec[0], k_vec_stats_min[k_loop][0], k_vec_stats_max[k_loop][0]);  
                
                // // NOTE (Shang): No need to do the reduction here. the hmma_dot above already does the reduction within the head.
                // float qk_max_tmp = qk_max;
                // if (qk_vec_idx < Dh_MAX)
                // {
                //     // qk = dot<Qk_vec_acum, Qk_vec_k>(q, k);
                //     if constexpr (QK_VECS_PER_Dh_MAX <= WARP_SIZE)
                //     {
                //         #pragma unroll
                //         for (int mask = QK_VECS_PER_Dh_MAX / 2; mask >= 1; mask /= 2)
                //         {
                //             qk_max += __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), qk_max, mask);
                //         }
                //     }
                // }

                // if (blockIdx.y == 0 && blockIdx.z == 0 && (threadIdx.x % 16 == 0))
                // {
                //     half *q_local = reinterpret_cast<half *>(&q_vec[0]);
                //     // blockIdx.x is q_head_dim, blockIdx.y is batch idx (every 16 threads output once: k_idx.y == 0?)
                //     // Should have a restriction on the validity of logic_sub_chunk_now when writing back
                //     printf("blockIdx.x (head_idx): %d, threadIdx.x:%d, logic_sub_chunk_now (sub_chunk_idx): %d, qk_min_max: %f, params.num_heads %d, n_sub_chunks:%d \n", blockIdx.x, threadIdx.x, logic_sub_chunk_now, qk_min_max, params.num_heads, n_sub_chunks);
                // }

                if (k_idx.y == 0){
                    if (logic_sub_chunk_now < n_sub_chunks){
                        half *out_stats_ptr_local =  out_stats_ptr + logic_sub_chunk_now;
                        *out_stats_ptr_local = __float2half(qk_min_max);
                        // printf("logic_sub_chunk_now (sub_chunk_idx): %d, qk_min_max: %f, out_stats_ptr_local[0]: %f\n", logic_sub_chunk_now, qk_min_max, __half2float(out_stats_ptr_local[0]));
                    }
                }
            }
        }
    }



    template <
        // The type of the inputs. Supported types: float, uint16_t, nv_bfloat16.
        typename T,
        // The type of the cache.
        typename Tcache,
        // Type of struct containing KV cache
        typename RetrievalKVCacheBuffer, typename StreamingKVCacheBuffer, 
        // The hidden dimension per head.
        unsigned Dh,
        // The number of threads in a threadblock.
        unsigned THREADS_PER_BLOCK,
        // Whether enable multi-block mode for long-sequence-length.
        bool DO_MULTI_BLOCK = false,
        // Whether use INT4KV
        bool INT4KV = false,
        bool KV_WITH_ZEROS = false,
        bool SMEM_PRELOAD = false,
        // The number of threads per key.
        unsigned THREADS_PER_KEY = mmha::threads_per_key<T, dh_max(Dh)>(),
        // The number of threads per value.
        unsigned THREADS_PER_VALUE = mmha::threads_per_value<T>(dh_max(Dh)),
        // The unroll factor for loading from K cache.
        // unsigned K_LOOP_UNROLL = 8, // 8,
        // The unroll factor for loading from V cache.
        // Set it default to 4 for higher occupancy (by reducing registers usage).
        unsigned V_LOOP_UNROLL = 4>
    __global__ void masked_multihead_attention_page_selector_compute(
        Multihead_attention_page_selector_params<T> params, RetrievalKVCacheBuffer retrieval_kv_buffer, StreamingKVCacheBuffer streaming_kv_buffer){
            const int qheads_per_kv_head = params.num_heads / params.num_kv_heads;
            const int kv_head_idx = blockIdx.x / qheads_per_kv_head;
            
            const int is_retrieval_head = params.retrieval_head_flags_ptr[kv_head_idx]!=0;
            const int head_rank = params.head_rank_table_ptr[kv_head_idx];

            const bool do_dynamic_sparse = (params.dynamic_sparse_page_idxes_ptr != nullptr);
            // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x ==0 ){
            //     printf("Here in masked_multihead_attention_page_selector_compute do dynamic_sparse is %d\n", do_dynamic_sparse);
            // }

            if (is_retrieval_head){
                // NOTE: We cannot set two branches for do_dynamic_sparse and !do_dynamic_sparse, because the smem will overflow.
                // NOTE: We can probably move the do_dynamic_sparse branch to the outer wrapper in the future.
                // if (do_dynamic_sparse){
                    masked_multihead_attention_page_selector_kernel<T, Tcache, RetrievalKVCacheBuffer, Dh, THREADS_PER_BLOCK, true /*IS_RETRIEVAL_HEAD*/, DO_MULTI_BLOCK, true /*DO_DYNAMIC_SPARSE*/, INT4KV, KV_WITH_ZEROS, SMEM_PRELOAD, THREADS_PER_KEY, THREADS_PER_VALUE, V_LOOP_UNROLL>(params, retrieval_kv_buffer, head_rank);
                // }
                // else{
                //     masked_multihead_attention_page_selector_kernel<T, Tcache, RetrievalKVCacheBuffer, Dh, THREADS_PER_BLOCK, true /*IS_RETRIEVAL_HEAD*/, DO_MULTI_BLOCK, false /*DO_DYNAMIC_SPARSE*/, INT4KV, KV_WITH_ZEROS, SMEM_PRELOAD, THREADS_PER_KEY, THREADS_PER_VALUE, V_LOOP_UNROLL>(params, retrieval_kv_buffer, head_rank);
                // }
            }else{
                masked_multihead_attention_page_selector_kernel<T, Tcache, StreamingKVCacheBuffer, Dh, THREADS_PER_BLOCK, false, false /*DO_MULTI_BLOCK*/, false /*DO_DYNAMIC_SPARSE*/, INT4KV, KV_WITH_ZEROS, SMEM_PRELOAD, THREADS_PER_KEY, THREADS_PER_VALUE, V_LOOP_UNROLL>(params, streaming_kv_buffer, head_rank);
            }
        }
    

    template <typename T, int Dh, bool DO_MULTI_BLOCK>
    inline size_t smem_size_in_bytes(const Multihead_attention_page_selector_params<T> &params, int threads_per_block)
    {
        using Tk = typename kernel_type_t<T>::Type;
        size_t transpose_rotary_size = 0;
        if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX)
        {
            transpose_rotary_size =
                2ULL * static_cast<size_t>(params.rotary_embedding_dim) * sizeof(Tk);
        }
        return transpose_rotary_size;
    }

} // namespace mmha