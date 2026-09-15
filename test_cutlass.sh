cat >/tmp/cutlass_probe.cu <<'EOF'
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>

int main() {
  auto shape = cute::make_shape(cute::Int<16>{}, cute::Int<16>{});
  (void)shape;
  return 0;
}
EOF

nvcc -std=c++17 \
  -I"$CUTLASS_PATH/include" \
  -c /tmp/cutlass_probe.cu \
  -o /tmp/cutlass_probe.o

echo $?