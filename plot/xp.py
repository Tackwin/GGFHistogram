import subprocess

for s in range(10, 31):
	size = 2 ** s
	for b in range(5, 9):
		bins = 2 ** b
		iters = 100
		for init in ["rand", "const", "inc"]:
			print(
				"Histogram.exe",
				"size", str(size),
				"bins", str(bins),
				"init", str(init),
				"benchmark", str(iters), "simd", "omp"
			)
			subprocess.call([
				"Histogram.exe",
				"size", str(size),
				"bins", str(bins),
				"init", str(init),
				"benchmark", str(iters), "simd", "omp"
			])
