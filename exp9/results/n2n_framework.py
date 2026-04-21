import sys
sys.path.append('d:/02_Projects/my_antigravity/PlotNeuralNet')
from pycore.tikzeng import *

arch = [
    to_head('.'),
    to_cor(),
    to_begin(),
    
    # 1. Original Noisy Image
    to_Conv("noisy", 1, 1, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=1, caption="Noisy Image $y$" ),
    
    # 2. Downsampling into Sub1 and Sub2
    # Sub1 (top)
    to_Conv("sub1", 1, 1, offset="(2.5, 4, 0)", to="(noisy-east)", height=20, depth=20, width=1, caption="Sub-image $y_1$" ),
    to_connection("noisy", "sub1"),
    
    # Sub2 (bottom)
    to_Conv("sub2", 1, 1, offset="(2.5, -4, 0)", to="(noisy-east)", height=20, depth=20, width=1, caption="Sub-image $y_2$" ),
    to_connection("noisy", "sub2"),
    
    # 3. Generator Structure (U-Net for sub1)
    to_Conv("enc1", 32, 32, offset="(2.0, 0, 0)", to="(sub1-east)", height=20, depth=20, width=2, caption="Encoder" ),
    to_connection("sub1", "enc1"),
    
    to_Conv("enc2", 64, 64, offset="(1.0, 0, 0)", to="(enc1-east)", height=10, depth=10, width=4, caption="" ),
    to_connection("enc1", "enc2"),
    
    to_Conv("bottleneck", 128, 128, offset="(1.0, 0, 0)", to="(enc2-east)", height=5, depth=5, width=6, caption="U-Net" ),
    to_connection("enc2", "bottleneck"),
    
    to_Conv("dec2", 64, 64, offset="(1.0, 0, 0)", to="(bottleneck-east)", height=10, depth=10, width=4, caption="" ),
    to_connection("bottleneck", "dec2"),
    to_skip("enc2", "dec2"),
    
    to_Conv("dec1", 32, 32, offset="(1.0, 0, 0)", to="(dec2-east)", height=20, depth=20, width=2, caption="Decoder" ),
    to_connection("dec2", "dec1"),
    to_skip("enc1", "dec1"),
    
    # 4. Output Prediction
    to_Conv("pred", 1, 1, offset="(1.5, 0, 0)", to="(dec1-east)", height=20, depth=20, width=1, caption="Prediction $f(y_1)$" ),
    to_connection("dec1", "pred"),
    
    # 5. Reconstruction Loss Connection (from pred to sub2)
    # We use raw TikZ to draw a dashed red arrow between prediction and sub2
    r"""
    \draw [dashed, <->, draw=red, thick] (pred-south) -- node[midway, right, text=red, font=\bfseries] {Loss} (sub2-east);
    """,
    r"""
    \node[anchor=north, text width=4cm, align=center] at (noisy-south) {\vspace{1cm}};
    """,
    to_end()
]

def main():
    to_generate(arch, 'n2n_framework.tex')

if __name__ == '__main__':
    main()
