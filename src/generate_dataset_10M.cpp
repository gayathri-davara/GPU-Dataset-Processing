#include <iostream>
#include <fstream>
#include <random>

int main()
{
    const size_t N = 10'000'000;

    std::ofstream file(
        "C:\\Users\\PowerX\\GPU-Dataset-Processing\\data\\dataset_10M.bin",
        std::ios::binary
    );

    if (!file)
    {
        std::cerr << "Error: Could not create dataset file.\n";
        return 1;
    }

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(0.0f, 1000.0f);

    for (size_t i = 0; i < N; i++)
    {
        float value = distribution(generator);
        file.write(
            reinterpret_cast<const char*>(&value),
            sizeof(float)
        );
    }

    file.close();

    std::cout << "Dataset generated successfully.\n";
    std::cout << "Elements: " << N << "\n";
    std::cout << "File: data\\dataset_10M.bin\n";
    std::cout << "Size: approximately 40 MB\n";

    return 0;
}