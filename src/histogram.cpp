#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

#ifndef SIZE
#define SIZE ((size_t)1024*1024*1024)
#endif

#ifndef BINS
#define BINS (size_t)256
#endif

size_t Size = SIZE;
size_t Bins = BINS;

size_t* temp_buffer = NULL;

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

typedef void(*compute_f)(uint8_t*, size_t, size_t*);
typedef void(*init_f)(uint8_t*, size_t);


// std rand is annoyingly slow when generating a **billion** random numbers
unsigned long xorshf96() {
	static unsigned long x=123456789, y=362436069, z=521288629;
	unsigned long t;
	x ^= x << 16;
	x ^= x >> 5;
	x ^= x << 1;

	t = x;
	x = y;
	y = z;
	z = t ^ x ^ y;

	return z;
}

void compute_histogram_simple(uint8_t* page, size_t page_size, size_t* histogram) {
	for (size_t i = 0; i < page_size; ++i)
		histogram[page[i]]++;
}

void compute_histogram_omp(uint8_t* page, size_t page_size, size_t* histogram) {
	size_t max_threads = omp_get_max_threads();
	size_t stride = Bins + 8;
	size_t* hs = (size_t*)calloc(stride * max_threads, sizeof(size_t));

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
		for (size_t j = 0; j < Bins; ++j)
			histogram[j] += hs[i * stride + j];

	free(hs);
}

void compute_histogram_omp_simd(uint8_t* page, size_t page_size, size_t* histogram) {
	size_t max_threads = omp_get_max_threads();
	size_t stride = Bins + 8;
	size_t simd_size = 4;
	size_t align = 128;

	// so we will say that we have simd_size histogram for each threads
	size_t* hs = (size_t*)calloc(
		max_threads * simd_size * stride, sizeof(size_t)
	);

	#pragma omp parallel
	{
		
		size_t t_id = omp_get_thread_num();
		size_t n_t = omp_get_num_threads();

		size_t n = (size_t)(ceilf(page_size / (1.f * align * n_t)) * align);
		size_t i = n * t_id;
		n = i + n;

		n = n > page_size ? page_size : n;

		__m128i idx_mask1 = _mm_set1_epi32(0x000000ff);
		__m128i idx_mask2 = _mm_set1_epi32(0x0000ff00);
		__m128i idx_mask3 = _mm_set1_epi32(0x00ff0000);
		__m128i idx_mask4 = _mm_set1_epi32(0xff000000);
		__m128i raw_idx;
		__m128i idx;

		__m256i cur       = _mm256_set1_epi64x(0);
		__m256i inc_const = _mm256_set1_epi64x(1);


		#define H_(simd_id, id) hs[\
			t_id * simd_size * stride +\
			simd_id * stride +\
			reinterpret_cast<uint32_t*>(&idx)[id]\
		]

		#define H(id) H_(id, id)

		for (i = 0; i < n; i += 16) {
			// First iteration
			raw_idx = _mm_load_si128((__m128i const *)(page + i));

			idx = _mm_and_si128(raw_idx, idx_mask1);
			idx = _mm_srli_epi32(idx, 0);

			H(0)++; H(1)++; H(2)++; H(3)++;

			// Second iteration
			idx = _mm_and_si128(raw_idx, idx_mask2);
			idx = _mm_srli_epi32(idx, 8);

			H(0)++; H(1)++; H(2)++; H(3)++;

			// Third iteration
			idx = _mm_and_si128(raw_idx, idx_mask3);
			idx = _mm_srli_epi32(idx, 16);

			H(0)++; H(1)++; H(2)++; H(3)++;

			// Forth iteration
			idx = _mm_and_si128(raw_idx, idx_mask4);
			idx = _mm_srli_epi32(idx, 24);

			H(0)++; H(1)++; H(2)++; H(3)++;
		}

		for (size_t j = 1; j < simd_size; ++j)
			for (size_t k = 0; k < Bins; ++k)
				hs[t_id * simd_size * stride + k] += hs[t_id * simd_size * stride + j * stride + k];

		#undef H
		#undef H_
	}

	for (size_t i = 0; i < max_threads; ++i)
		for (size_t j = 0; j < Bins; ++j)
			histogram[j] += hs[i * stride * simd_size + j];

	free(hs);
}


void compute_histogram_simd(uint8_t* page, size_t page_size, size_t* histogram) {
	size_t i = 0;

	size_t* h = (size_t*)calloc(Bins * 4, sizeof(size_t));

	__m128i idx_mask1 = _mm_set1_epi32(0x000000ff);
	__m128i idx_mask2 = _mm_set1_epi32(0x0000ff00);
	__m128i idx_mask3 = _mm_set1_epi32(0x00ff0000);
	__m128i idx_mask4 = _mm_set1_epi32(0xff000000);
	__m128i raw_idx;
	__m128i idx;

	__m256i cur       = _mm256_set1_epi64x(0);
	__m256i inc_const = _mm256_set1_epi64x(1);

	#define H(id) h[id * Bins + reinterpret_cast<uint32_t*>(&idx)[id]]

	for (i = 0; i < page_size; i += 16) {

		// First iteration
		raw_idx = _mm_load_si128((__m128i const *)(page + i));

		idx = _mm_and_si128(raw_idx, idx_mask1);
		idx = _mm_srli_epi32(idx, 0);

		H(0)++;
		H(1)++;
		H(2)++;
		H(3)++;

		// Second iteration
		idx = _mm_and_si128(raw_idx, idx_mask2);
		idx = _mm_srli_epi32(idx, 8);

		H(0)++;
		H(1)++;
		H(2)++;
		H(3)++;

		// Third iteration
		idx = _mm_and_si128(raw_idx, idx_mask3);
		idx = _mm_srli_epi32(idx, 16);

		H(0)++;
		H(1)++;
		H(2)++;
		H(3)++;

		// Forth iteration
		idx = _mm_and_si128(raw_idx, idx_mask4);
		idx = _mm_srli_epi32(idx, 24);

		H(0)++;
		H(1)++;
		H(2)++;
		H(3)++;
	}

	#undef H

	for (size_t j = 0; j < 4; ++j) for (size_t k = 0; k < Bins; ++k) {
		histogram[k] += h[j * Bins + k];
	}

	for (; i < page_size; ++i) histogram[page[i]]++;

	free(h);
}

void init_rand(uint8_t* page, size_t page_size) {
	for (size_t i = 0; i < page_size; ++i) page[i] = xorshf96() % Bins;
}

void init_const(uint8_t* page, size_t page_size) {
	uint8_t c = 0;
	for (size_t i = 0; i < page_size; ++i) page[i] = c;
}

void init_inc(uint8_t* page, size_t page_size) {
	for (size_t i = 0; i < page_size; ++i) page[i] = i % Bins;
}

init_f get_init(const char* str) {
	if (strcmp(str, "const") == 0) return init_const;
	if (strcmp(str, "rand" ) == 0) return init_rand;
	if (strcmp(str, "inc"  ) == 0) return init_inc;
	printf("Warining init %s not recognized.\n", str);
	return init_inc;
}
const char* get_init_name(init_f f) {
	if (f == init_const)   return "const";
	if (f == init_rand)    return "rand";
	if (f == init_inc)     return "inc";
	return "???";
}

compute_f get_function(char* str) {
	if (strcmp(str, "omp_simd") == 0) return compute_histogram_omp_simd;
	if (strcmp(str, "simple"  ) == 0) return compute_histogram_simple;
	if (strcmp(str, "simd"    ) == 0) return compute_histogram_simd;
	if (strcmp(str, "omp"     ) == 0) return compute_histogram_omp;
	printf("Warning function %s not recognized.\n", str);
	return compute_histogram_simple;
}

const char* get_function_name(compute_f f) {
	if (f == compute_histogram_omp_simd) return "omp_simd";
	if (f == compute_histogram_simple)   return "simple";
	if (f == compute_histogram_simd)     return "simd";
	if (f == compute_histogram_omp)      return "omp";
	return "???";
}

const char* get_openmp_schedule() {
	return "Static";
}

int test_function(compute_f f, uint8_t* p) {
	size_t* h       = (size_t*)calloc(Bins, sizeof(size_t));
	size_t* h_truth = (size_t*)calloc(Bins, sizeof(size_t));
	compute_histogram_simple(p, Size, h_truth);
	f                       (p, Size, h);

	printf("Histograms ... ");
	int res = memcmp(h_truth, h, Bins * sizeof(size_t));
	int ret = 0;

	if (res == 0) {
		printf("Matches :^)!\n");
		ret = 0;
	}
	else {
		printf("does not match :-(\n");

		for (size_t i = 0; i < Bins; ++i)
			printf(
				"[% 4d ] % 25lld - % 25lld = % 25lld\n",
				(int)i,
				(int64_t)h_truth[i],
				(int64_t)h[i],
				(int64_t)(h_truth[i] - h[i])
			);

		ret = -1;
	}

	free(h);
	free(h_truth);
	return ret;
}

void shuffle(compute_f *array, size_t n) {
	if (n > 1) for (size_t i = 0; i < n - 1; i++) {
		size_t j = i + rand() / (RAND_MAX / (n - i) + 1);

		compute_f t = array[j];
		array[j] = array[i];
		array[i] = t;
	}
}

void benchmark_function(
	init_f i_f,
	compute_f* fs,
	size_t n_f,
	size_t n_runs,
	uint8_t* page,
	size_t page_size,
	size_t* histogram
) noexcept {
	struct Report {
		const char* schedule;
		size_t      threads;
		const char* kernel;
		const char* init;
		size_t      size;
		size_t      bins;
		double      time;
	};

	compute_f* f_array = (compute_f*)malloc(n_f * n_runs * sizeof(compute_f));

	for (size_t i = 0; i < n_f; ++i) for (size_t j = 0; j < n_runs; ++j)
		f_array[i * n_runs + j] = fs[i];
	shuffle(f_array, n_f * n_runs);


	FILE* file = fopen("report.csv", "a");
	if (!file) {
		printf("Can't save to file %s\n", "report.csv");
		printf("Error: %d\n", errno);
	}

	fseek(file, 0L, SEEK_END);
	size_t file_size = ftell(file);
	rewind(file);

	if (file_size == 0) fprintf(file, "init;kernel;size;bins;time;threads;schedule\n");
	for (size_t i = 0; i < n_f * n_runs; ++i) {
		memset(histogram, 0, sizeof(size_t) * Bins);
		double s = seconds();
		f_array[i](page, page_size, histogram);
		double e = seconds();

		Report r;

		r.init = get_init_name(i_f);
		r.kernel = get_function_name(f_array[i]);
		r.size = Size;
		r.bins = Bins;
		r.time = e - s;
		r.threads = omp_get_max_threads();
		r.schedule = get_openmp_schedule();

		fprintf(
			file,
			"%-20s;%-20s;% 20lld;% 20lld;% 20lf;% 20lld;%-20s\n",
			r.init,
			r.kernel,
			r.size,
			r.bins,
			r.time,
			r.threads,
			r.schedule
		);
		fflush(file);
		printf(
			"[% 5lld] %-20s;%-20s;% 20lld;% 20lld;% 20lf;% 20lld;%-20s\n",
			(int64_t)i,
			r.init,
			r.kernel,
			r.size,
			r.bins,
			r.time,
			r.threads,
			r.schedule
		);
	}

	free(f_array);
}

int main(int argc, char** argv) {
	size_t i;
	size_t t;
	size_t* h;
	uint8_t* p;
	size_t runs = 1;
	compute_f f = compute_histogram_simple;
	init_f init = init_inc;

	if (argc > 1) {
		size_t cursor = 1;

		if (strcmp(argv[cursor], "size") == 0) {
			Size = strtoll(argv[++cursor], NULL, 10);
			cursor++;
		}
		if (strcmp(argv[cursor], "bins") == 0) {
			Bins = strtoll(argv[++cursor], NULL, 10);
			cursor++;
		}
		if (strcmp(argv[cursor], "init") == 0) {
			init = get_init(argv[++cursor]);
			cursor++;
		}
		
		int ret = 0;
		p = (uint8_t*)calloc(Size, sizeof(uint8_t));
		h = (size_t*)calloc(Size, sizeof(size_t));

		init(p, Size);

		if (strcmp(argv[cursor], "test") == 0) ret = test_function(get_function(argv[++cursor]), p);
		else if (strcmp(argv[cursor], "avg") == 0) {
			runs = strtol(argv[++cursor], NULL, 10);

			if (argc > 3) f = get_function(argv[++cursor]);

			auto s = seconds();
			for (i = 0; i < runs; ++i) f(p, Size, h);
			auto e = seconds();

			printf("% 17lld   elements in %10.8lf seconds.\n", (int64_t)(Size * runs), e - s);
			printf("% 17.10lf M elements per seconds.\n", runs * Size / (e - s) / (1'000'000.0));

			ret = 0;
		}
		else if (strcmp(argv[cursor], "benchmark") == 0) {
			size_t n_runs = strtol(argv[++cursor], NULL, 10);

			size_t n_f = argc - cursor - 1;
			compute_f* fs = (compute_f*)malloc(sizeof(compute_f) * n_f);

			for (size_t i = 0; i < n_f; ++i) fs[i] = get_function(argv[++cursor]);

			benchmark_function(init, fs, n_f, n_runs, p, Size, h);

			free(fs);
			ret = 0;
		}

		free(p);
		free(h);
		return ret;
	} else {
		printf(
			"Use with the following options\n"
			"  size       <size>\n"
			"  bins       <bins>\n"
			"  init       <init_name>\n"
			"  test       <kernel_name>\n"
			"  avg        <n_runs> <kernel_name = simple>\n"
			"  benchmark  <n_runs> <kernel1> <kernel2> ... <kernel_n>\n"
		);
	}

	return 0;
}
