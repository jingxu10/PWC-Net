import torch
import corr_gpu
import corr1d_gpu

class correlation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=1, corr_multiply=1):
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply
        output = corr_gpu.corr_cuda_forward(input1, input2,
                               ctx.pad_size,
                               ctx.kernel_size,
                               ctx.max_displacement,
                               ctx.stride1,
                               ctx.stride2,
                               ctx.corr_multiply)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1 = torch.zeros(input1.size()).cuda()
        grad_input2 = torch.zeros(input2.size()).cuda()

        corr_gpu.corr_cuda_backward(input1, input2,
                                grad_output,
                                grad_input1,
                                grad_input2,
                                ctx.pad_size,
                                ctx.kernel_size,
                                ctx.max_displacement,
                                ctx.stride1,
                                ctx.stride2,
                                ctx.corr_multiply)

        return grad_input1, grad_input2, None, None, None, None, None, None


#----- 1D correlation (for disparity) Jinwei Gu -----

class correlation1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=1, corr_multiply=1):
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply
        output = corr1d_gpu.corr1d_cuda_forward(input1, input2,
                               ctx.pad_size,
                               ctx.kernel_size,
                               ctx.max_displacement,
                               ctx.stride1,
                               ctx.stride2,
                               ctx.corr_multiply)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        grad_input1 = torch.zeros(input1.size()).cuda()
        grad_input2 = torch.zeros(input2.size()).cuda()

        #grad_input1 = grad_output.new()
        #grad_input2 = grad_output.new()

        corr1d_gpu.corr1d_cuda_backward(input1, input2,
                                grad_output,
                                grad_input1,
                                grad_input2,
                                ctx.pad_size,
                                ctx.kernel_size,
                                ctx.max_displacement,
                                ctx.stride1,
                                ctx.stride2,
                                ctx.corr_multiply)
        return grad_input1, grad_input2, None, None, None, None, None, None

class Correlation(torch.nn.Module):
    def __init__(self, pad_size=None, kernel_size=None, max_displacement=None, stride1=None, stride2=None, corr_multiply=None):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def reset_params(self):
        return

    def forward(self, input1, input2):
        return correlation.apply(input1, input2, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)

    def __repr__(self):
        return self.__class__.__name__

#----- correlation in 1D (for disparity) Jinwei Gu -----

class Correlation1d(torch.nn.Module):
    def __init__(self, pad_size=None, kernel_size=None, max_displacement=None, stride1=None, stride2=None, corr_multiply=None):
        super(Correlation1d, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def reset_params(self):
        return

    def forward(self, input1, input2):
        return correlation1d.apply(input1, input2)

    def __repr__(self):
        return self.__class__.__name__
