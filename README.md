\# 🚀 LET'S GOOOO BRO



Now we're generating the \*\*README.md and .gitignore directly inside your real Windows project\*\*.



\## 1️⃣ Generate README



\*\*Run on: PowerX PC — x64 Native Tools Command Prompt for Visual Studio\*\*



```bat

cd /d C:\\Users\\PowerX\\GPU-Dataset-Processing



notepad README.md

```



Paste this entire README:



```markdown

\# GPU Dataset Processing Using CUDA



\## Parallel Computing Lab Evaluation — Team 8



\*\*Topic:\*\* GPU Dataset Processing  

\*\*Parallel Model:\*\* CUDA  

\*\*GPU:\*\* NVIDIA GeForce RTX 5060 Ti  

\*\*CUDA Toolkit:\*\* 13.4



\---



\## 1. Objective



The objective of this project is to process a large numerical dataset using CUDA and analyze the execution-time difference between sequential CPU processing and GPU-based parallel processing.



The program performs:



\- SUM

\- MINIMUM

\- MAXIMUM

\- AVERAGE



The experiment compares:



1\. Sequential CPU processing

2\. Parallel CUDA GPU processing



The experiment measures CPU execution time, GPU kernel execution time, GPU total processing time, validation error, and speedup.



\---



\## 2. Hardware and Software



\### Hardware



\- NVIDIA GeForce RTX 5060 Ti

\- Approximately 16 GB VRAM

\- Compute Capability 12.0



\### Software



\- Windows

\- NVIDIA Driver 610.88

\- CUDA Toolkit 13.4

\- Visual Studio 2026 Build Tools

\- MSVC x64 compiler

\- C++

\- CUDA C++



The CUDA programs were compiled for the native GPU architecture using:



```text

nvcc -arch=compute\_120 -code=sm\_120

```



\---



\## 3. Dataset



Synthetic floating-point datasets were generated using a reproducible Mersenne Twister random-number generator.



Random values are generated between:



```text

0.0 and 1000.0

```



A fixed random seed of `42` is used so that the generated datasets are reproducible.



\### Dataset sizes



| Dataset | Elements | Approx. Size |

|---:|---:|---:|

| 1M | 1,000,000 | 4 MB |

| 5M | 5,000,000 | 20 MB |

| 10M | 10,000,000 | 40 MB |

| 50M | 50,000,000 | 200 MB |

| 100M | 100,000,000 | 400 MB |



The generated binary datasets are excluded from GitHub because of their large size.



\---



\## 4. CUDA Implementation



The GPU implementation uses a parallel reduction strategy.



Each CUDA block processes a portion of the input dataset and calculates partial:



\- Sum

\- Minimum

\- Maximum



Shared memory is used for block-level reduction.



The CPU combines the resulting block partials.



\### CUDA configuration



```text

Threads per block: 256

Number of blocks : 256

```



The implementation avoids global floating-point atomic operations for the main reduction.



\---



\## 5. Timing Methodology



\### CPU Time



The time required by the sequential CPU implementation to process all dataset elements.



\### GPU Kernel Time



The CUDA kernel execution time measured using CUDA events.



\### GPU Total Time



The end-to-end GPU processing time including:



\- Host-to-device transfer

\- CUDA kernel execution

\- Device-to-host transfer



File loading is excluded from these processing measurements.



This distinction is important because kernel-only acceleration can be much larger than complete GPU application speedup when memory-transfer overhead is significant.



\---



\# 6. Benchmark Results



| Dataset | CPU Time (ms) | GPU Kernel (ms) | GPU Total (ms) | Kernel Speedup | Total Speedup |

|---:|---:|---:|---:|---:|---:|

| 1M | 1.9734 | 0.187328 | 1.081184 | 10.53x | 1.83x |

| 5M | 9.9669 | 0.176640 | 3.189056 | 56.42x | 3.13x |

| 10M | 19.6427 | 0.218080 | 6.083776 | 90.07x | 3.23x |

| 50M | 95.6955 | 0.821696 | 32.505569 | 116.46x | 2.94x |

| 100M | 190.1299 | 1.410784 | 57.622017 | 134.77x | 3.30x |



\---



\## 7. Validation



GPU results were compared against CPU results.



Minimum and maximum values matched across all tested datasets.



Average-value differences remained approximately `0.000001` or smaller.



The SUM values show small numerical differences because CPU and GPU floating-point reductions use different accumulation orders.



For example, for 100M elements:



```text

CPU SUM       = 49997551604.514511

GPU SUM       = 49997551456.000000

SUM difference = 148.514511

```



This difference is small relative to the approximately 50-billion total.



Therefore, validation uses numerical tolerance rather than exact floating-point equality.



\---



\## 8. Performance Analysis



\### CPU Scaling



CPU processing time increases substantially as the dataset size increases.



```text

1M   = 1.9734 ms

10M  = 19.6427 ms

100M = 190.1299 ms

```



\### GPU Kernel Performance



The CUDA kernel remains very fast even for the largest dataset.



```text

1M   = 0.187328 ms

100M = 1.410784 ms

```



Kernel-only speedup increased from:



```text

10.53x at 1M

```



to:



```text

134.77x at 100M

```



\### End-to-End Performance



When memory transfers are included, the measured total speedup is lower:



```text

1.83x at 1M

3.30x at 100M

```



This demonstrates the effect of CPU-GPU memory-transfer overhead on complete application performance.



\---



\## 9. Graphs



The `graphs` directory contains:



\- CPU execution time

\- GPU kernel execution time

\- GPU total execution time

\- GPU kernel speedup

\- GPU total speedup



Graph files are stored in SVG format for high-quality scaling in reports and presentations.



\---



\## 10. Project Structure



```text

GPU-Dataset-Processing/

│

├── data/

│   ├── dataset\_1M.bin

│   ├── dataset\_5M.bin

│   ├── dataset\_10M.bin

│   ├── dataset\_50M.bin

│   └── dataset\_100M.bin

│

├── graphs/

│   ├── cpu\_execution\_time.svg

│   ├── gpu\_kernel\_time.svg

│   ├── gpu\_total\_time.svg

│   ├── kernel\_speedup.svg

│   └── total\_speedup.svg

│

├── results/

│   └── benchmark\_results.csv

│

├── src/

│   ├── cuda\_test.cu

│   ├── generate\_dataset.cpp

│   ├── generate\_dataset\_5M.cpp

│   ├── generate\_dataset\_10M.cpp

│   ├── generate\_dataset\_50M.cpp

│   ├── generate\_dataset\_100M.cpp

│   ├── gpu\_dataset.cu

│   ├── gpu\_dataset\_v1\_working.cu

│   ├── gpu\_benchmark.cu

│   ├── gpu\_benchmark\_5M.cu

│   ├── gpu\_benchmark\_10M.cu

│   ├── gpu\_benchmark\_50M.cu

│   └── gpu\_benchmark\_100M.cu

│

├── README.md

└── .gitignore

```



\---



\## 11. Build Instructions



Open an \*\*x64 Native Tools Command Prompt for Visual Studio\*\*.



Navigate to the project:



```bat

cd /d C:\\Users\\PowerX\\GPU-Dataset-Processing

```



Compile the main CUDA benchmark:



```bat

nvcc -arch=compute\_120 -code=sm\_120 .\\src\\gpu\_benchmark.cu -o .\\gpu\_benchmark.exe

```



Run:



```bat

.\\gpu\_benchmark.exe

```



\---



\## 12. Reproducing the Experiment



Example: generate the 100M dataset.



Compile the generator:



```bat

cl .\\src\\generate\_dataset\_100M.cpp /EHsc /Fe:.\\generate\_dataset\_100M.exe

```



Run:



```bat

.\\generate\_dataset\_100M.exe

```



Compile the CUDA benchmark:



```bat

nvcc -arch=compute\_120 -code=sm\_120 .\\src\\gpu\_benchmark\_100M.cu -o .\\gpu\_benchmark\_100M.exe

```



Run:



```bat

.\\gpu\_benchmark\_100M.exe

```



\---



\## 13. Conclusion



The experiment demonstrates the benefits of GPU parallel processing for large numerical datasets.



The maximum measured kernel-only speedup was:



```text

134.77x

```



at 100M elements.



The maximum measured end-to-end speedup was:



```text

3.30x

```



at 100M elements.



The results demonstrate that CUDA can provide substantial computational acceleration while also showing that CPU-GPU memory-transfer overhead affects complete application performance.



\---



\## 14. Team



\*\*Parallel Computing Lab Evaluation — Team 8\*\*



\*\*Topic:\*\* GPU Dataset Processing  

\*\*Technology:\*\* NVIDIA CUDA

```



Save and close Notepad.



\---



\# 2️⃣ Generate `.gitignore`



\*\*Run on: PowerX PC — same command prompt\*\*



```bat

notepad .gitignore

```



Paste:



```gitignore

\# Executables

\*.exe



\# Visual Studio object files

\*.obj



\# Debug/build files

\*.pdb

\*.ilk

\*.lib

\*.exp



\# CUDA build artifacts

\*.ptx

\*.cubin

\*.fatbin



\# Large generated datasets

data/\*.bin

\*.bin



\# Temporary files

\*.tmp

\*.log



\# IDE files

.vs/

.vscode/

.idea/



\# OS files

Thumbs.db

.DS\_Store

```



