require 'nnbhwdtexfuncs'
require 'image'

foo=torch.CudaTensor(1,512,512,3)

lena=image.lena()
lena=lena:transpose(1,2):transpose(3,2):contiguous():cuda()
foo:select(1,1):copy(lena)

texfun=nn.TexFunCustom():cuda()
texfun:forward(foo)

out=texfun.output:float():select(1,1):transpose(3,2):transpose(1,2):contiguous()
image.display(out)


