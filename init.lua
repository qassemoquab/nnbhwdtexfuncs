require "torch"
require "cunn"
require "libnnbhwdtexfuncs"

local nnbhwdtexfuncs = {}

nnbhwdtexfuncs.ExtractInterpolate = require 'nnbhwdtexfuncs.ExtractInterpolate'
nnbhwdtexfuncs.TexFunRandResize = require 'nnbhwdtexfuncs.TexFunRandResize'
nnbhwdtexfuncs.TexFunRandFlip = require 'nnbhwdtexfuncs.TexFunRandFlip'
nnbhwdtexfuncs.TexFunFixedResize = require 'nnbhwdtexfuncs.TexFunFixedResize'
nnbhwdtexfuncs.TexFunCustom = require 'nnbhwdtexfuncs.TexFunCustom'
nnbhwdtexfuncs.TexFunCropJitter = require 'nnbhwdtexfuncs.TexFunCropJitter'

return nnbhwdtexfuncs
