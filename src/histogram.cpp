#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <string>

#define SIZE ((size_t)1024*1024*1024)
#define BINS (size_t)256
//#define SIZE ((size_t)256*1)

#include <chrono>
#include <omp.h>

static double seconds() noexcept {
	auto now     = std::chrono::system_clock::now();
	auto epoch   = now.time_since_epoch();
	auto seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);

	// return the number of seconds
	return seconds.count() / 1'000'000'000.0;
}

void compute_histogram_simple(unsigned char* page, size_t page_size, size_t* histogram) noexcept {
	for (size_t i = 0; i < page_size; ++i) histogram[page[i]]++;
}


void compute_histogram_omp(unsigned char* page, size_t page_size, size_t* histogram) noexcept {
	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < page_size; ++i) histogram[page[i]]++;
}


void compute_histogram_reduce(unsigned char* page, size_t page_size, size_t* histogram) noexcept {
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

auto get_function = [](char* str) {
	if (strcmp(str, "simple") == 0) return compute_histogram_simple;
	if (strcmp(str, "reduce") == 0) return compute_histogram_reduce;
	if (strcmp(str, "omp"   ) == 0) return compute_histogram_omp;
	return compute_histogram_simple;
};

void test_function(char* f_str) {
	auto f = get_function(f_str);
	uint8_t* p = (unsigned char *)malloc(sizeof(char) * SIZE);
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
				"[% 4d ] % 25lld - % 25lld = % 25lld\n",
				(int)i,
				(int64_t)h_truth[i],
				(int64_t)h[i],
				(int64_t)(h_truth[i] - h[i])
			);

	}
}

int main(int argc, char** argv) {
	unsigned char *p;
	size_t h[BINS] = {};
	size_t i;
	size_t t;

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

	p = (unsigned char *)malloc(sizeof(char) * SIZE);
	for (i = 0; i < SIZE; i++) p[i] = i % BINS;

	auto s = seconds();

	for (i = 0; i < runs; ++i) f(p, SIZE, h);
	
	auto e = seconds();
	printf("% 17lld   elements in %10.8lf seconds.\n", (int64_t)SIZE * runs, e - s);
	printf("% 17.10lf M elements per seconds.\n", runs * SIZE / (e - s) / (1'000'000.0));

	volatile int x = h[rand() % BINS];
	x = 0;

	return x;
}
