nvcc new_main.cu `pkg-config --cflags --libs opencv4` -diag-suppress=611
./a.out img.jpg 1.5