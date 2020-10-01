#ifndef EXTRACT_HOG_CUDA_KERNEL_CUH
#define EXTRACT_HOG_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif


// small value, used to avoid division by zero
#define eps 0.0001

template <typename T>
__global__ void hog_grad_forward_cuda_kernel(const int nthreads, const T* im, const int num,
        const int channels, const int height, const int width,
        const int hist_channels, const int blocks_height, const int
        blocks_width, const int visible_height, const int  visible_width,
        const int out_channels, const int out_height, const int out_width,
        const int sbin,
        T* grad_v, T* grad_i, T* hist, T* norm, T* feat){
    // unit vectors used to compute gradient orientation
    T uu[9] = {1.0000,
        0.9397,
        0.7660,
        0.500,
        0.1736,
        -0.1736,
        -0.5000,
        -0.7660,
        -0.9397};
    T vv[9] = {0.0000,
        0.3420,
        0.6428,
        0.8660,
        0.9848,
        0.9848,
        0.8660,
        0.6428,
        0.3420};

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int w = index % (visible_width-2) + 1;
        int h = (index / (visible_width-2)) % (visible_height-2) + 1;
        int n = (index / (visible_width-2)) / (visible_height-2);
        int pw = min(w, width-2);
        int ph = min(h, height-2);

        // first color channel
        const T* im_off = im + (n*channels*height + ph) * width + pw;
        T dy = im_off[width] - im_off[-width];
        T dx = im_off[1] - im_off[-1];
        T v = dx*dx + dy*dy;

        // second color channel
        im_off += height * width;
        T dy2 = im_off[width] - im_off[-width];
        T dx2 = im_off[1] - im_off[-1];
        T v2 = dx2*dx2 + dy2*dy2;

        // third color channel
        im_off += height * width;
        T dy3 = im_off[width] - im_off[-width];
        T dx3 = im_off[1] - im_off[-1];
        T v3 = dx3*dx3 + dy3*dy3;

        // pick channel with strongest gradient
        if (v2 > v) {
            v = v2;
            dx = dx2;
            dy = dy2;
        }
        if (v3 > v) {
            v = v3;
            dx = dx3;
            dy = dy3;
        }

        // snap to one of 18 orientations
        T best_dot = 0;
        int best_o = 0;
        for (int o = 0; o < 9; o++) {
            T dot = uu[o]*dx + vv[o]*dy;
            if (dot > best_dot) {
                best_dot = dot;
                best_o = o;
            } else if (-dot > best_dot) {
                best_dot = -dot;
                best_o = o+9;
            }
        }
        v = sqrt(v);

        grad_v[index] = v;
        grad_i[index] = best_o;
    }
}

template <typename T>
__global__ void hog_bin_forward_cuda_kernel(const int nthreads, const T* im, const int num,
        const int channels, const int height, const int width,
        const int hist_channels, const int blocks_height, const int
        blocks_width, const int visible_height, const int  visible_width,
        const int out_channels, const int out_height, const int out_width,
        const int sbin,
        T* grad_v, T* grad_i, T* hist, T*norm, T*feat){
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int wb = index % blocks_width;
        int hb = (index / blocks_width) % blocks_height;
        int ob = ((index / blocks_width) / blocks_height) % hist_channels;
        int n = ((index / blocks_width) / blocks_height) / hist_channels;

        // add to 4 histograms around pixel using linear interpolation
        int w0 = (int)floor(((T)wb-1+0.5)*(T)sbin -0.5);
        int w1 = (int)ceil(((T)wb+1+0.5)*(T)sbin -0.5);
        int h0 = (int)floor(((T)hb-1+0.5)*(T)sbin -0.5);
        int h1 = (int)ceil(((T)hb+1+0.5)*(T)sbin -0.5);
        T sum = 0.0;
        // for (int w = 1; w < visible_width-1; w++) {
        //     for(int h = 1; h < visible_height-1; h++) {
        for (int w = max(1, w0); w < min(visible_width-1, w1); w++){
            for (int h = max(1, h0); h < min(visible_height-1, h1); h++){
                T wp = ((T)w+0.5)/(T)sbin - 0.5;
                T hp = ((T)h+0.5)/(T)sbin - 0.5;
                int iwp = (int)floor(wp);
                int ihp = (int)floor(hp);
                T vw0 = wp-iwp;
                T vh0 = hp-ihp;
                T vw1 = 1.0-vw0;
                T vh1 = 1.0-vh0;
                int iw = w-1;
                int ih = h-1;
                int o = grad_i[(n*(visible_height-2)+ih)*(visible_width-2)+iw];
                T v = grad_v[(n*(visible_height-2)+ih)*(visible_width-2)+iw];
                if (iwp == wb && ihp == hb && o == ob) {
                    sum += vw1*vh1*v;
                }
                if (iwp+1 == wb && ihp == hb && o == ob) {
                    sum += vw0*vh1*v;
                }
                if (iwp == wb && ihp+1 == hb && o == ob) {
                    sum += vw1*vh0*v;
                }
                if (iwp+1 == wb && ihp+1 == hb && o == ob) {
                    sum += vw0*vh0*v;
                }
            }
        }
        hist[index] = sum;
    }
}

template <typename T>
__global__ void hog_norm_forward_cuda_kernel(const int nthreads, const T* im, const int num,
        const int channels, const int height, const int width,
        const int hist_channels, const int blocks_height, const int
        blocks_width, const int visible_height, const int  visible_width,
        const int out_channels, const int out_height, const int out_width,
        const int sbin,
        T* grad_v, T* grad_i, T* hist, T*norm, T*feat){

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int w = index % blocks_width;
        int h = (index / blocks_width) % blocks_height;
        int n = (index / blocks_width) / blocks_height;
        T sum = 0.0;
        for (int o = 0; o < 9; o++) {
            int off1 = ((n*hist_channels + o)*blocks_height + h)*blocks_width + w;
            int off2 = ((n*hist_channels + (o+9))*blocks_height + h)*blocks_width + w;
            sum += (hist[off1]+hist[off2])*(hist[off1]+hist[off2]);
        }
        norm[index] = sum;
    }
}

template <typename T>
__global__ void hog_comp_forward_cuda_kernel(const int nthreads, const T* im, const int num,
        const int channels, const int height, const int width,
        const int hist_channels, const int blocks_height, const int
        blocks_width, const int visible_height, const int  visible_width,
        const int out_channels, const int out_height, const int out_width,
        const int sbin,
        T* grad_v, T* grad_i, T* hist, T*norm, T*feat){

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int w = index % out_width;
        int h = (index / out_width) % out_height;
        int n = (index / out_width) / out_height;
        int off;
        T n1, n2, n3, n4;
        off = (n*blocks_height + (h+1))*blocks_width + (w+1);
        n1 = 1.0 / sqrt(norm[off] + norm[off+1] + norm[off+blocks_width]
                + norm[off+blocks_width+1] + eps);
        off = (n*blocks_height + h)*blocks_width + (w+1);
        n2 = 1.0 / sqrt(norm[off] + norm[off+1] + norm[off+blocks_width]
                + norm[off+blocks_width+1] + eps);
        off = (n*blocks_height + (h+1))*blocks_width + w;
        n3 = 1.0 / sqrt(norm[off] + norm[off+1] + norm[off+blocks_width]
                + norm[off+blocks_width+1] + eps);
        off = (n*blocks_height + h)*blocks_width + w;
        n4 = 1.0 / sqrt(norm[off] + norm[off+1] + norm[off+blocks_width]
                + norm[off+blocks_width+1] + eps);

        T t1 = 0;
        T t2 = 0;
        T t3 = 0;
        T t4 = 0;

        // contrast-sensitive features
        int hoff = (n*hist_channels*blocks_height + (h+1))*blocks_width + (w+1);
        int foff = (n*out_channels*out_height + h)*out_width + w;
        for (int o = 0; o < 18; o++) {
            T h1 = min(hist[hoff] * n1, 0.2);
            T h2 = min(hist[hoff] * n2, 0.2);
            T h3 = min(hist[hoff] * n3, 0.2);
            T h4 = min(hist[hoff] * n4, 0.2);
            feat[foff] = 0.5 * (h1 + h2 + h3 + h4);
            t1 += h1;
            t2 += h2;
            t3 += h3;
            t4 += h4;
            foff += out_height*out_width;
            hoff += blocks_height*blocks_width;
        }

        // contrast-insensitive features
        hoff = (n*hist_channels*blocks_height + (h+1))*blocks_width + (w+1);
        for (int o = 0; o < 9; o++) {
            T sum = hist[hoff] + hist[hoff+9*blocks_width*blocks_height];
            T h1 = min(sum * n1, 0.2);
            T h2 = min(sum * n2, 0.2);
            T h3 = min(sum * n3, 0.2);
            T h4 = min(sum * n4, 0.2);
            feat[foff]= 0.5 * (h1 + h2 + h3 + h4);
            foff += out_height*out_width;
            hoff += blocks_height*blocks_width;
        }

        // texture features
        feat[foff] = 0.2357 * t1;
        foff += out_height*out_width;
        feat[foff] = 0.2357 * t2;
        foff += out_height*out_width;
        feat[foff] = 0.2357 * t3;
        foff += out_height*out_width;
        feat[foff] = 0.2357 * t4;
    }
}

#endif  // EXTRACT_HOG_CUDA_KERNEL_CUH
