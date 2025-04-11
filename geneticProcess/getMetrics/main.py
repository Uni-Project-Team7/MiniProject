import sys
import os
sys.path.append(os.path.abspath("../../"))
from customOperations.archBuilderDir.encodingToArch import decode_and_build_unet

def get_stats(gene, device):
    
