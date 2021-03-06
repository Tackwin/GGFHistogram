#include "Ease.hpp"

/*
clang++ Build.cpp -o Build.exe -std=c++17
*/

Build build(Flags flags) noexcept {
	Build b = Build::get_default(flags);

	b.name = "Histogram";

	b.add_header("src/");
	b.add_source_recursively("src/");
	b.add_define("_CRT_SECURE_NO_WARNINGS");

	return b;
}