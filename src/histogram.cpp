#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <string>

#ifndef SIZE
#define SIZE ((size_t)1024*1024*1024)
#endif

#ifndef BINS
#define BINS (size_t)256
#endif

#include <chrono>
#include <omp.h>
#include <immintrin.h>

static double seconds() noexcept {
	auto now     = std::chrono::system_clock::now();
	auto epoch   = now.time_since_epoch();
	auto seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);

	// return the number of seconds
	return seconds.count() / 1'000'000'000.0;
}

void compute_histogram_simple(uint8_t* page, size_t page_size, size_t* histogram) noexcept {
	for (size_t i = 0; i < page_size; ++i)
		histogram[page[i]]++;
}


void compute_histogram_omp(uint8_t* page, size_t page_size, size_t* histogram) noexcept {
	constexpr size_t max_threads = 32;
	constexpr size_t stride = BINS + 8;
	size_t hs[stride * max_threads] = {};

	#pragma omp parallel
	{
		
		size_t t_id = omp_get_thread_num();
		size_t n_t = omp_get_num_threads();

		size_t n = (page_size + n_t - 1) / n_t;
		size_t i = n * t_id;
		n = i + n;

		n = n > page_size ? page_size : n;

		for (; i < n; i++) {
			hs[t_id * stride + page[i]]++;
		}
	}

	for (size_t i = 0; i < max_threads ; ++i)
		for (size_t j = 0; j < BINS; ++j)
			histogram[j] += hs[i * stride + j];
}

#ifdef __GNUC__
	#define ALIGN(x) x __attribute__((aligned(32)))
#elif defined(_MSC_VER)
	#define ALIGN(x) __declspec(align(32))
#endif

// __m256i _mm256_i32gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale)
// __m256i _mm256_load_si256 (__m256i const * mem_addr)
void compute_histogram_simd(uint8_t* page, size_t page_size, size_t* histogram) noexcept {
	size_t i = 0;


	__m128i idx_mask1 = _mm_set1_epi32(0x000000ff);
	__m128i idx_mask2 = _mm_set1_epi32(0x0000ff00);
	__m128i idx_mask3 = _mm_set1_epi32(0x00ff0000);
	__m128i idx_mask4 = _mm_set1_epi32(0xff000000);
	__m128i raw_idx;
	__m128i idx;

	__m256i cur       = _mm256_set1_epi64x(0);
	__m256i inc_const = _mm256_set1_epi64x(1);

	for (i = 0; i < page_size; i += 16) {

		// First iteration
		raw_idx = _mm_load_si128((__m128i const *)(page + i));

		idx = _mm_and_si128(raw_idx, idx_mask1);
		idx = _mm_srli_epi32(idx, 0);

		cur = _mm256_i32gather_epi64(histogram, idx, 8);
		cur = _mm256_add_epi64(cur, inc_const);

		histogram[reinterpret_cast<uint32_t*>(&idx)[0]] = reinterpret_cast<uint64_t*>(&cur)[0];
		histogram[reinterpret_cast<uint32_t*>(&idx)[1]] = reinterpret_cast<uint64_t*>(&cur)[1];
		histogram[reinterpret_cast<uint32_t*>(&idx)[2]] = reinterpret_cast<uint64_t*>(&cur)[2];
		histogram[reinterpret_cast<uint32_t*>(&idx)[3]] = reinterpret_cast<uint64_t*>(&cur)[3];

		// Second iteration
		idx = _mm_and_si128(raw_idx, idx_mask2);
		idx = _mm_srli_epi32(idx, 8);

		cur = _mm256_i32gather_epi64(histogram, idx, 8);
		cur = _mm256_add_epi64(cur, inc_const);

		histogram[reinterpret_cast<uint32_t*>(&idx)[0]] = reinterpret_cast<uint64_t*>(&cur)[0];
		histogram[reinterpret_cast<uint32_t*>(&idx)[1]] = reinterpret_cast<uint64_t*>(&cur)[1];
		histogram[reinterpret_cast<uint32_t*>(&idx)[2]] = reinterpret_cast<uint64_t*>(&cur)[2];
		histogram[reinterpret_cast<uint32_t*>(&idx)[3]] = reinterpret_cast<uint64_t*>(&cur)[3];

		// Third iteration
		idx = _mm_and_si128(raw_idx, idx_mask3);
		idx = _mm_srli_epi32(idx, 16);

		cur = _mm256_i32gather_epi64(histogram, idx, 8);
		cur = _mm256_add_epi64(cur, inc_const);

		histogram[reinterpret_cast<uint32_t*>(&idx)[0]] = reinterpret_cast<uint64_t*>(&cur)[0];
		histogram[reinterpret_cast<uint32_t*>(&idx)[1]] = reinterpret_cast<uint64_t*>(&cur)[1];
		histogram[reinterpret_cast<uint32_t*>(&idx)[2]] = reinterpret_cast<uint64_t*>(&cur)[2];
		histogram[reinterpret_cast<uint32_t*>(&idx)[3]] = reinterpret_cast<uint64_t*>(&cur)[3];

		// Forth iteration
		idx = _mm_and_si128(raw_idx, idx_mask4);
		idx = _mm_srli_epi32(idx, 24);

		cur = _mm256_i32gather_epi64(histogram, idx, 8);
		cur = _mm256_add_epi64(cur, inc_const);

		histogram[reinterpret_cast<uint32_t*>(&idx)[0]] = reinterpret_cast<uint64_t*>(&cur)[0];
		histogram[reinterpret_cast<uint32_t*>(&idx)[1]] = reinterpret_cast<uint64_t*>(&cur)[1];
		histogram[reinterpret_cast<uint32_t*>(&idx)[2]] = reinterpret_cast<uint64_t*>(&cur)[2];
		histogram[reinterpret_cast<uint32_t*>(&idx)[3]] = reinterpret_cast<uint64_t*>(&cur)[3];
	}

	for (; i < page_size; ++i) histogram[page[i]]++;
}


auto get_function = [](char* str) {
	if (strcmp(str, "simple") == 0) return compute_histogram_simple;
	if (strcmp(str, "simd"  ) == 0) return compute_histogram_simd;
	if (strcmp(str, "omp"   ) == 0) return compute_histogram_omp;
	return compute_histogram_simple;
};

void test_function(char* f_str) {
	auto f = get_function(f_str);
	uint8_t* p = (uint8_t*)malloc(sizeof(uint8_t) * SIZE);
	for (size_t i = 0; i < SIZE; i++) p[i] = i % BINS;

	size_t h[BINS] = {};
	size_t h_truth[BINS] = {};
	compute_histogram_simple(p, SIZE, h_truth);
	f                       (p, SIZE, h);

	printf("Histograms ... ");
	auto res = memcmp(h_truth, h, BINS);
	if (res == 0) printf("Matches :^)!\n");
	else {
		printf("does not match :-(\n");

		for (size_t i = 0; i < BINS; ++i)
			printf(
				"[% 4d ] % 25ld - % 25ld = % 25ld\n",
				(int)i,
				(int64_t)h_truth[i],
				(int64_t)h[i],
				(int64_t)(h_truth[i] - h[i])
			);

	}
}

int main(int argc, char** argv) {
	size_t h[BINS] = {};
	size_t i;
	size_t t;
	uint8_t* p;

	size_t runs = 1;

	auto f = compute_histogram_simple;
	if (argc > 1) {
		if (strcmp(argv[1], "test") == 0) return test_function(argv[2]), 0;
		else if (strcmp(argv[1], "avg") == 0) {

			runs = std::stoi(argv[2]);

			if (argc > 3) f = get_function(argv[3]);
		}
		else f = get_function(argv[1]);
	}

	p = (uint8_t*)malloc(sizeof(uint8_t) * SIZE);
	for (i = 0; i < SIZE; i++) p[i] = i % BINS;

	auto s = seconds();

	for (i = 0; i < runs; ++i) f(p, SIZE, h);
	
	auto e = seconds();
	printf("% 17ld   elements in %10.8lf seconds.\n", (int64_t)SIZE * runs, e - s);
	printf("% 17.10lf M elements per seconds.\n", runs * SIZE / (e - s) / (1'000'000.0));

	volatile int x = h[rand() % BINS];
	x = 0;

	return x;
}
