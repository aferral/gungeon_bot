g++ capture_and_write.c -o capture_and_write -lX11 -lXext `pkg-config opencv --cflags --libs` `$cv`-Ofast -mfpmath=both -march=native -m64 -funroll-loops -mavx2

