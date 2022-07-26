GXX=/home/gtx1080ti/riscv/Xuantie-900-gcc-elf-newlib-x86_64-V2.2.4/bin/riscv64-unknown-elf-g++
USE_IM2COL=-DUSE_IM2COL
USE_WINOGRAD=-DUSE_WINOGRAD

wg:
	$(GXX) -O3  -DUSE_RISCV_VECTOR $(USE_WINOGRAD) -march=rv64imafdcv0p7_zfh_xtheadc src/conv2d_wg.cpp -o bin/conv2d_vector

wg_serial:
	$(GXX) -O3  -DUSE_WINOGRAD -march=rv64imafdcv0p7_zfh_xtheadc src/conv2d_wg.cpp -o bin/conv2d_serial


vector:
	#$(GXX) -O2  -DUSE_RISCV_VECTOR $(USE_IM2COL) -march=rv64imafdcv0p7_zfh_xtheadc -S src/conv2d.cpp 
	#$(GXX) -O2  -DUSE_RISCV_VECTOR $(USE_IM2COL) -march=rv64imafdcv0p7_zfh_xtheadc -g -c src/conv2d.cpp
	$(GXX) -O3  -DUSE_RISCV_VECTOR $(USE_IM2COL) -march=rv64imafdcv0p7_zfh_xtheadc src/conv2d.cpp -o bin/conv2d_vector

serial:
	$(GXX) -O3  -DUSE_IM2COL -march=rv64imafdcv0p7_zfh_xtheadc src/conv2d.cpp -o bin/conv2d_serial


cpvector: bin/conv2d_vector
	scp bin/conv2d_vector root@192.168.2.75:/root/rvv-test/
cpserial: bin/conv2d_serial
	scp bin/conv2d_serial root@192.168.2.75:/root/rvv-test/
