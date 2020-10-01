#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void ExtractHOGForwardCUDAKernelLaucher(const Tensor input, const int num,
                                        const int channels, const int height, const int width,
                                        const int hist_channels, const int blocks_height,
                                        const int blocks_width,
                                        const int visible_height, const int visible_width,
                                        const int out_channels,
                                        const int out_height, const int out_width,
                                        const int sbin,
                                        Tensor grad_v, Tensor grad_i,
                                        Tensor hist, Tensor norm,
                                        Tensor feat);

void extract_hog_forward_cuda(const Tensor input, int sbin, Tensor output,
                              Tensor grad_v, Tensor grad_i,
                              Tensor hist, Tensor norm)
{
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    if (in_channels != 3)
    {
        AT_ERROR("wrong input channels, expected to be 3\n");
        return;
    }
    int in_height = input.size(2);
    int in_width = input.size(3);

    int out_channels = output.size(1);
    if (out_channels != 31)
    {
        AT_ERROR("wrong output channels, expected to be 31\n");
        return;
    }
    int out_height = output.size(2);
    int out_width = output.size(3);

    int visible_height = grad_v.size(2) + 2;
    int visible_width = grad_v.size(3) + 2;

    int hist_channels = hist.size(1);
    int blocks_height = hist.size(2);
    int blocks_width = hist.size(3);

    ExtractHOGForwardCUDAKernelLaucher(input, batch_size, in_channels, in_height, in_width,
                                       hist_channels, blocks_height, blocks_width,
                                       visible_height, visible_width,
                                       out_channels, out_height, out_width,
                                       sbin,
                                       grad_v, grad_i, hist, norm, output);
}
#endif

void extract_hog_forward(const Tensor input, int sbin, Tensor output,
                         Tensor grad_v, Tensor grad_i,
                         Tensor hist, Tensor norm) {
      if (input.device().is_cuda()) {
    #ifdef MMCV_WITH_CUDA
        CHECK_CUDA_INPUT(input);
        CHECK_CUDA_INPUT(output);
        CHECK_CUDA_INPUT(grad_v);
        CHECK_CUDA_INPUT(grad_i);
        CHECK_CUDA_INPUT(hist);
        CHECK_CUDA_INPUT(norm);

        extract_hog_forward_cuda(input, sbin, output, grad_v, grad_i, hist, norm);
    #else
        AT_ERROR("ExtractHOG is not compiled with GPU support");
    #endif
      }
      else {
        AT_ERROR("ExtractHOG is not implemented on CPU");
      }
}

