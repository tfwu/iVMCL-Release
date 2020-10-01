#include "pytorch_cuda_helper.hpp"
#include "extract_hog_cuda_kernel.cuh"

void ExtractHOGForwardCUDAKernelLaucher(const Tensor x, const int num,
                                        const int channels, const int height, const int width,
                                        const int hist_channels, const int blocks_height,
                                        const int blocks_width,
                                        const int visible_height, const int  visible_width,
                                        const int out_channels,
                                        const int out_height, const int out_width,
                                        const int sbin,
                                        Tensor grad_v, Tensor grad_i,
                                        Tensor hist, Tensor norm,
                                        Tensor feat ) {
  at::cuda::CUDAGuard device_guard(x.device());

  const int count_grad = num * (visible_height - 2) * (visible_width - 2);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x.scalar_type(), "hog_grad_forward_cuda_kernel", ([&] {
        const scalar_t *input_data = x.data_ptr<scalar_t>();
        scalar_t *grad_v_data = grad_v.data_ptr<scalar_t>();
        scalar_t *grad_i_data = grad_i.data_ptr<scalar_t>();
        scalar_t *hist_data = hist.data_ptr<scalar_t>();
        scalar_t *norm_data = norm.data_ptr<scalar_t>();
        scalar_t *output_data = feat.data_ptr<scalar_t>();

        hog_grad_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(count_grad), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
              count_grad, input_data, num,
              channels, height, width,
              hist_channels, blocks_height, blocks_width,
              visible_height, visible_width,
              out_channels, out_height, out_width,
              sbin,
              grad_v_data, grad_i_data, hist_data, norm_data, output_data);
      }));

  AT_CUDA_CHECK(cudaGetLastError());

  const int count_hist = num * hist_channels * blocks_height * blocks_width;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    x.scalar_type(), "hog_bin_forward_cuda_kernel", ([&] {
      const scalar_t *input_data = x.data_ptr<scalar_t>();
      scalar_t *grad_v_data = grad_v.data_ptr<scalar_t>();
      scalar_t *grad_i_data = grad_i.data_ptr<scalar_t>();
      scalar_t *hist_data = hist.data_ptr<scalar_t>();
      scalar_t *norm_data = norm.data_ptr<scalar_t>();
      scalar_t *output_data = feat.data_ptr<scalar_t>();

      hog_bin_forward_cuda_kernel<scalar_t>
          <<<GET_BLOCKS(count_hist), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            count_hist, input_data, num,
            channels, height, width,
            hist_channels, blocks_height, blocks_width,
            visible_height, visible_width,
            out_channels, out_height, out_width,
            sbin,
            grad_v_data, grad_i_data, hist_data, norm_data, output_data);
    }));

  AT_CUDA_CHECK(cudaGetLastError());

  const int count_norm = num * blocks_height * blocks_width;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    x.scalar_type(), "hog_norm_forward_cuda_kernel", ([&] {
      const scalar_t *input_data = x.data_ptr<scalar_t>();
      scalar_t *grad_v_data = grad_v.data_ptr<scalar_t>();
      scalar_t *grad_i_data = grad_i.data_ptr<scalar_t>();
      scalar_t *hist_data = hist.data_ptr<scalar_t>();
      scalar_t *norm_data = norm.data_ptr<scalar_t>();
      scalar_t *output_data = feat.data_ptr<scalar_t>();

      hog_norm_forward_cuda_kernel<scalar_t>
          <<<GET_BLOCKS(count_norm), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            count_norm, input_data, num,
            channels, height, width,
            hist_channels, blocks_height, blocks_width,
            visible_height, visible_width,
            out_channels, out_height, out_width,
            sbin,
            grad_v_data, grad_i_data, hist_data, norm_data, output_data);
    }));

  AT_CUDA_CHECK(cudaGetLastError());

  const int count_comp = num * out_height * out_width;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    x.scalar_type(), "hog_comp_forward_cuda_kernel", ([&] {
      const scalar_t *input_data = x.data_ptr<scalar_t>();
      scalar_t *grad_v_data = grad_v.data_ptr<scalar_t>();
      scalar_t *grad_i_data = grad_i.data_ptr<scalar_t>();
      scalar_t *hist_data = hist.data_ptr<scalar_t>();
      scalar_t *norm_data = norm.data_ptr<scalar_t>();
      scalar_t *output_data = feat.data_ptr<scalar_t>();

      hog_comp_forward_cuda_kernel<scalar_t>
          <<<GET_BLOCKS(count_comp), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            count_comp, input_data, num,
            channels, height, width,
            hist_channels, blocks_height, blocks_width,
            visible_height, visible_width,
            out_channels, out_height, out_width,
            sbin,
            grad_v_data, grad_i_data, hist_data, norm_data, output_data);
    }));

  AT_CUDA_CHECK(cudaGetLastError());
}
