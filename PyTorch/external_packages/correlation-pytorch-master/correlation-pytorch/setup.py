#!/usr/bin/env python

import os
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
	name='correlation_package',
	ext_modules=[
		CppExtension(
			name='corr_cpu',
			sources=['correlation_package/src/corr.cpp']
		),
		CppExtension(
			name='corr1d_cpu',
			sources=['correlation_package/src/corr1d.cpp']
		),
		CUDAExtension(
			name='corr_gpu',
			sources=[
				'correlation_package/src/corr_cuda.cpp',
				'correlation_package/src/corr_cuda_kernel.cu'
			],
		),
		CUDAExtension(
			name='corr1d_gpu',
			sources=[
				'correlation_package/src/corr1d_cuda.cpp',
				'correlation_package/src/corr1d_cuda_kernel.cu'
			],
		)
	],
	cmdclass={
		'build_ext': BuildExtension
	})


# import os
#
# from setuptools import setup, find_packages
#
# this_file = os.path.dirname(__file__)
#
# setup(
#     name="correlation_package",
#     version="0.1",
#     description="Correlation layer from FlowNetC",
#     url="https://github.com/jbarker-nvidia/pytorch-correlation",
#     author="Jon Barker",
#     author_email="jbarker@nvidia.com",
#     # Require cffi
#     install_requires=["cffi>=1.0.0"],
#     setup_requires=["cffi>=1.0.0"],
#     # Exclude the build files.
#     packages=find_packages(exclude=["build"]),
#     # Package where to put the extensions. Has to be a prefix of build.py
#     ext_package="",
#     # Extensions to compile
#     cffi_modules=[
#         os.path.join(this_file, "build.py:ffi")
#     ],
# )
