#include "adaptive_gemm.h"
#include "gemm_problem.h"

torch::Tensor adaptive_gemm(torch::Tensor a, 
  torch::Tensor b, 
  const c10::optional<at::Tensor>& c,
  const c10::optional<at::Tensor>& bias, 
  double alpha,
  double beta,
  const std::string& epilogue)
{
  //check input tensor
  check_input(a, b);

  //parse the epilogue kind
  

  //allocate output tensor 
  int m = a.size(0);
  int n = b.size(1);
  int k = b.size(0);
  auto d = torch::zeros({m, n}, a.option());

  //set gemm problem
  at::Tensor bias_tensor = bias.value_or(at::Tensor{});
  at::Tensor c_tensor = c.value_or(at::Tensor{});
  bool has_bias = bias.defined();
  bool has_c = c.defined();
  cudaStream_t cur_stream =
        c10::cuda::getCurrentCUDAStream(a.get_device()).stream();

  GemmProblem gemmProblem(a, b, d, cur_stream);
  gemmProblem.has_c = has_c;
  gemmProblem.c = (has_c) ? c_tensor.data_ptr<c.dtype>() : nullptr;
  gemmProblem.dtype_c = (has_c) ? c_tensor.dtype : float;
  gemmProblem.has_bias = has_bias;
  gemmProblem.bias = (has_bias) ? c_tensor.data_ptr<bias.dtype>() : nullptr;
  gemmProblem.epilogue = parse_epilogue(epilogue);

}