#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

struct Result {
    double size;
    double cpu;
    double kernel;
    double total;
    double kernelSpeedup;
    double totalSpeedup;
};

std::string num(double value)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << value;
    return ss.str();
}

void generateGraph(
    const std::string& filename,
    const std::string& title,
    const std::string& yLabel,
    const std::vector<double>& values,
    const std::string& valueSuffix)
{
    const int width = 1000;
    const int height = 650;

    const int left = 100;
    const int right = 60;
    const int top = 80;
    const int bottom = 100;

    const int graphWidth = width - left - right;
    const int graphHeight = height - top - bottom;

    double maxValue = *std::max_element(values.begin(), values.end());
    maxValue *= 1.15;

    std::ofstream out(filename);

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"" << width << "\" height=\"" << height << "\" "
        << "viewBox=\"0 0 " << width << " " << height << "\">\n";

    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";

    out << "<text x=\"" << width / 2
        << "\" y=\"40\" text-anchor=\"middle\" "
        << "font-family=\"Arial\" font-size=\"26\" font-weight=\"bold\">"
        << title << "</text>\n";

    // Axes
    out << "<line x1=\"" << left << "\" y1=\"" << top
        << "\" x2=\"" << left << "\" y2=\"" << top + graphHeight
        << "\" stroke=\"black\" stroke-width=\"2\"/>\n";

    out << "<line x1=\"" << left << "\" y1=\"" << top + graphHeight
        << "\" x2=\"" << left + graphWidth << "\" y2=\""
        << top + graphHeight
        << "\" stroke=\"black\" stroke-width=\"2\"/>\n";

    // Y-axis label
    out << "<text x=\"25\" y=\"" << height / 2
        << "\" text-anchor=\"middle\" "
        << "font-family=\"Arial\" font-size=\"18\" "
        << "transform=\"rotate(-90 25 " << height / 2 << ")\">"
        << yLabel << "</text>\n";

    // Grid + Y labels
    for (int i = 0; i <= 5; i++)
    {
        double value = maxValue * i / 5.0;
        double y = top + graphHeight - graphHeight * i / 5.0;

        out << "<line x1=\"" << left << "\" y1=\"" << y
            << "\" x2=\"" << left + graphWidth << "\" y2=\"" << y
            << "\" stroke=\"#dddddd\"/>\n";

        out << "<text x=\"" << left - 15 << "\" y=\"" << y + 6
            << "\" text-anchor=\"end\" font-family=\"Arial\" font-size=\"14\">"
            << num(value) << valueSuffix << "</text>\n";
    }

    // Data line
    out << "<polyline fill=\"none\" stroke=\"#2563eb\" stroke-width=\"4\" points=\"";

    for (size_t i = 0; i < values.size(); i++)
    {
        double x;

        if (values.size() == 1)
            x = left + graphWidth / 2.0;
        else
            x = left + graphWidth * i / (values.size() - 1.0);

        double y = top + graphHeight - graphHeight * values[i] / maxValue;

        out << x << "," << y << " ";
    }

    out << "\"/>\n";

    // Points + labels
    for (size_t i = 0; i < values.size(); i++)
    {
        double x;

        if (values.size() == 1)
            x = left + graphWidth / 2.0;
        else
            x = left + graphWidth * i / (values.size() - 1.0);

        double y = top + graphHeight - graphHeight * values[i] / maxValue;

        out << "<circle cx=\"" << x << "\" cy=\"" << y
            << "\" r=\"7\" fill=\"#2563eb\"/>\n";

        out << "<text x=\"" << x << "\" y=\"" << top + graphHeight + 30
            << "\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"15\">"
            << static_cast<int>(i == 0 ? 1 :
                i == 1 ? 5 :
                i == 2 ? 10 :
                i == 3 ? 50 : 100)
            << "M</text>\n";

        out << "<text x=\"" << x << "\" y=\"" << y - 15
            << "\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"14\">"
            << num(values[i]) << valueSuffix << "</text>\n";
    }

    out << "<text x=\"" << width / 2
        << "\" y=\"" << height - 25
        << "\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"18\">"
        << "Dataset Size (Million Elements)</text>\n";

    out << "</svg>\n";
}

int main()
{
    std::vector<Result> data = {
        {1,   1.9734,   0.187328,  1.081184,  10.53,  1.83},
        {5,   9.9669,   0.176640,  3.189056,  56.42,  3.13},
        {10, 19.6427,   0.218080,  6.083776,  90.07,  3.23},
        {50, 95.6955,   0.821696, 32.505569, 116.46,  2.94},
        {100,190.1299,  1.410784, 57.622017, 134.77,  3.30}
    };

    std::vector<double> cpu;
    std::vector<double> kernel;
    std::vector<double> total;
    std::vector<double> kernelSpeedup;
    std::vector<double> totalSpeedup;

    for (const auto& r : data)
    {
        cpu.push_back(r.cpu);
        kernel.push_back(r.kernel);
        total.push_back(r.total);
        kernelSpeedup.push_back(r.kernelSpeedup);
        totalSpeedup.push_back(r.totalSpeedup);
    }

    generateGraph(
        "graphs/cpu_execution_time.svg",
        "CPU Execution Time vs Dataset Size",
        "Time (ms)",
        cpu,
        " ms"
    );

    generateGraph(
        "graphs/gpu_kernel_time.svg",
        "GPU Kernel Execution Time vs Dataset Size",
        "Time (ms)",
        kernel,
        " ms"
    );

    generateGraph(
        "graphs/gpu_total_time.svg",
        "GPU Total Execution Time vs Dataset Size",
        "Time (ms)",
        total,
        " ms"
    );

    generateGraph(
        "graphs/total_speedup.svg",
        "GPU Total Speedup vs Dataset Size",
        "Speedup",
        totalSpeedup,
        "x"
    );

    generateGraph(
        "graphs/kernel_speedup.svg",
        "GPU Kernel Speedup vs Dataset Size",
        "Speedup",
        kernelSpeedup,
        "x"
    );

    std::cout << "All graphs generated successfully.\n";

    return 0;
}