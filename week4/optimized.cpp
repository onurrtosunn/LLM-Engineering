
#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(long long iterations, int param1, int param2) {
    double result = 1.0;
    for (long long i = 1; i <= iterations; ++i) {
        double j = static_cast<double>(i) * param1 - param2;
        result -= (1.0 / j);
        j = static_cast<double>(i) * param1 + param2;
        result += (1.0 / j);
    }
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    double result = calculate(100'000'000, 4, 1) * 4;
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> execution_time = end_time - start_time;

    std::cout << std::fixed << std::setprecision(12) << "Result: " << result << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(6) << execution_time.count() << " seconds" << std::endl;

    return 0;
}
