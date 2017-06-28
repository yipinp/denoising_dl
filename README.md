# denoising_dl
image or video denoising based on deep learning and other machine learning method.
The program will explore many neural network to denoise still image and will extend to video denoising in the future.
Currently, the MLP/CNN/CNN with batch normalization is implemented. 
The compared golden program is BM3D, the python version comes from : https://github.com/liuhuang31/BM3D-Denoise. 
After test, the final result is not the same to the BM3D matlab. Let's using the matlab as the reference code.
The origin python code git : https://github.com/MarkPrecursor/BM3D_Denosing.git
The original BM3D is matlab version here : http://www.cs.tut.fi/~foi/GCF-BM3D/

There are some BM3D source codes:
VapourSynth-BM3D :  https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D/releases
This is a python wrapper around Marc Lebrun's implementation of BM3d:
https://github.com/ericmjonas/pybm3d
http://www.ipol.im/pub/art/2012/l-bm3d/
