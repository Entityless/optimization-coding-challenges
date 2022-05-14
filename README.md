
# Optimization Coding Challenges

Given `buckets[bucket_count][bucket_size]` and `pools[pool_count][pool_size]`;

For each element in `pool = pools[pool_id]`, for each `bucket_id` in range `bucket_count`, if `rand(0, 1) < fill_ratio`, we add it `fill_count` times to a random position in `buckets[bucket_id]`.

Since we do the same operation for each pool, we can abstract above actions into an array of `tuple<bucket_id, pool_offset, bucket_offset>`, and perform:

``` c++

for (int pool_id = 0; pool_id < pool_count; pool_id++) {
    for (auto t : accumulation_array) {
        bucket_id = t.bucket_id;
        pool_offset = t.pool_offset;
        bucket_offset = t.bucket_offset;
        pools[pool_id][pool_offset] = buckets[bucket_id][bucket_offset];
    }
}

```

Your are given `buckets` memset to 0, `pools` with random floating-point numbers, and theee arrays of `vector` format (SoA) representing the array of `tuple<bucket_id, pool_offset, bucket_offset>` (AoS). Since the operation of each pool is identical, you can do optimization on the reduction sequence using this feature. Also, the multi-thread hardware feature is also encouraged to be utilized.
