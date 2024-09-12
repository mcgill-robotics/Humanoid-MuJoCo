
### Build and Run MJPC GUI application

```
cd mujoco_mpc

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja -DCMAKE_C_COMPILER:STRING=clang-12 -DCMAKE_CXX_COMPILER:STRING=clang++-12 -DMJPC_BUILD_GRPC_SERVICE:BOOL=ON

cmake --build . --config=Release

cd bin
./mjpc
```
