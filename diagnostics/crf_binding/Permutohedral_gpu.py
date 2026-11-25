import ctypes as c
from pathlib import Path
import torch
_filter=c.CDLL(str(Path(__file__).with_name('lattice.so'))).insid3_filter
_filter.argtypes=[c.c_void_p]*4+[c.c_int]*3+[c.c_float]*3
_filter.restype=None

def forward(x,image,bilateral,alpha,beta,gamma):
    assert x.ndim==4 and x.size(0)==1 and x.size(-1)==2 and image.shape==(*x.shape[:3],3)
    assert x.dtype==image.dtype==torch.float32 and x.is_cuda and image.device==x.device
    assert x.is_contiguous() and image.is_contiguous()
    h,w=x.shape[1:3]
    positions=x.new_zeros(h*w*(5 if bilateral else 2))
    out=torch.zeros_like(x)
    _filter(out.data_ptr(),x.data_ptr(),image.data_ptr(),positions.data_ptr(),h,w,bilateral,alpha,beta,gamma)
    return out
