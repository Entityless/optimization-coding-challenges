#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif  // defined(__GNUC__)

using Element_type = float;

double get_rand_double_0_to_1() {
    return rand() * 1.0 / RAND_MAX;
}

size_t get_rand_int_in_range(size_t range) {
    return ((size_t)rand()) % range;
}

struct Buckets {
    std::vector<Element_type*> m_arrays;

    Buckets(size_t count, size_t size, bool random = false) {
        m_arrays.resize(count);
        for (size_t i = 0; i < count; i++) {
            m_arrays[i] = (Element_type*) _mm_malloc(size * sizeof(Element_type), 4096);

            if (!random) {
                for (size_t j = 0; j < size; j++) {
                    m_arrays[i][j] = 0;
                }
            } else {
                for (size_t j = 0; j < size; j++) {
                    m_arrays[i][j] = get_rand_double_0_to_1();
                }
            }
        }

        printf("buckets generation finished, count = %zu, size = %zu\n", count, size);
    }

    ~Buckets() {
    }
};

struct Accumulation_sequence {
    std::vector<size_t> m_pool_offsets, m_bucket_offsets;
    std::vector<uint8_t> m_bucket_ids;

    size_t m_bucket_count, m_bucket_size;
    size_t m_pool_count, m_pool_size;
    double m_fill_ratio;
    size_t m_fill_count;
    size_t m_sequence_size;

    Accumulation_sequence(size_t bucket_count, size_t bucket_size,
                          size_t pool_count, size_t pool_size,
                          double fill_ratio, size_t fill_count) :
                          m_bucket_count(bucket_count), m_bucket_size(bucket_size),
                          m_pool_count(pool_count), m_pool_size(pool_size),
                          m_fill_ratio(fill_ratio), m_fill_count(fill_count) {
        for (size_t pool_offset = 0; pool_offset < pool_size; pool_offset++) {
            for (size_t bucket_id = 0; bucket_id < bucket_count; bucket_id++) {
                if (get_rand_double_0_to_1() < fill_ratio) {
                    for (size_t i = 0; i < fill_count; i++) {
                        m_pool_offsets.push_back(pool_offset);
                        m_bucket_offsets.push_back(get_rand_int_in_range(bucket_size));
                        m_bucket_ids.push_back(bucket_id);
                    }
                }
            }
        }

        m_sequence_size = m_pool_offsets.size();

        printf("seq preparation finished:\nbucket count = %zu, bucket size = %zu\n", bucket_count, bucket_size);
        printf("pool count = %zu, pool size = %zu\n", pool_count, pool_size);
        printf("fill ratio = %f, fill count = %zu\n", fill_ratio, fill_count);
    }
};

void std_accumulation(Buckets& dst_buckets, const Buckets& src_pools, const Accumulation_sequence& seq) {
    for (size_t pool_id = 0; pool_id < seq.m_pool_count; pool_id++) {
        for (size_t i = 0; i < seq.m_sequence_size; i++) {
            int bucket_id = seq.m_bucket_ids[i];
            int pool_offset = seq.m_pool_offsets[i];
            int bucket_offset = seq.m_bucket_offsets[i];

            dst_buckets.m_arrays[bucket_id][bucket_offset] += src_pools.m_arrays[pool_id][pool_offset];
        }
    }
}

void optimized_accumulation(Buckets& dst_buckets, const Buckets& src_pools, const Accumulation_sequence& seq) {
    for (size_t pool_id = 0; pool_id < seq.m_pool_count; pool_id++) {
        Element_type* pool = src_pools.m_arrays[pool_id];
        for (size_t i = 0; i < seq.m_sequence_size; i++) {
            int bucket_id = seq.m_bucket_ids[i];
            int pool_offset = seq.m_pool_offsets[i];
            int bucket_offset = seq.m_bucket_offsets[i];

            dst_buckets.m_arrays[bucket_id][bucket_offset] += pool[pool_offset];
        }
    }
}

int main(int argc, char** argv) {
    srand(0);

    size_t bucket_count = omp_get_max_threads() / 2;
    size_t bucket_size = 2 * 1024 * 1024 / sizeof(Element_type);

    size_t pool_count = 20;
    size_t pool_size = 64 * 1024 * 1024 / sizeof(Element_type);

    double fill_ratio = 0.2;
    size_t fill_count = 4;

    Buckets std_buckets(bucket_count, bucket_size);
    Buckets optimized_output_buckets(bucket_count, bucket_size);
    Buckets pools(pool_count, pool_size, true);

    Accumulation_sequence sequence(bucket_count, bucket_size, pool_count, pool_size, fill_ratio, fill_count);

    {
        double stt = omp_get_wtime();
        std_accumulation(std_buckets, pools, sequence);
        double edt = omp_get_wtime();
        printf("STD Elapsed = %f\n", edt - stt);
    }

    {
        double stt = omp_get_wtime();
        optimized_accumulation(optimized_output_buckets, pools, sequence);
        double edt = omp_get_wtime();
        printf("OPT Elapsed = %f\n", edt - stt);
    }

    double diff_sum = 0;

    for (int bucket_id = 0; bucket_id < bucket_count; bucket_id++) {
        for (int bucket_offset = 0; bucket_offset < bucket_size; bucket_offset++) {
            diff_sum += std::abs(std_buckets.m_arrays[bucket_id][bucket_offset] - optimized_output_buckets.m_arrays[bucket_id][bucket_offset]);
        }
    }

    printf("diff sum = %f\n", diff_sum);

    return 0;
}
