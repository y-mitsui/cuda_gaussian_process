install_dir=/usr/local/lib/python2.7/dist-packages/

gaussian_process2:
	nvcc -I/usr/local/magma/include -O2 -g -arch=sm_20 -c -Xcompiler -fPIC gaussian_process.cu  -o gaussian_process.o
	cython --cplus cu_gaussian_process.pyx
	x86_64-linux-gnu-gcc -pthread -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fno-strict-aliasing -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security -fPIC -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -I/usr/include/python2.7 -c cu_gaussian_process.cpp -o cu_gaussian_process.o
	c++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wl,-Bsymbolic-functions -Wl,-z,relro -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security cu_gaussian_process.o ./gaussian_process.o -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -L/opt/OpenBLAS/lib -L/usr/local/cuda-7.5/lib64 -L/usr/local/magma/lib/  -lm -lgsl -lopenblas -lcudart -lcusolver -lcublas -lmagma -o cu_gaussian_process.so

install:
	install cu_gaussian_process.so $(install_dir)
	
clean:
	rm cu_gaussian_process.cpp
	rm gaussian_process.o
	rm cu_gaussian_process.o
	rm cu_gaussian_process.so
