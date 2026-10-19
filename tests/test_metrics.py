import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from dehaze.evaluation.metrics import psnr, ssim
class MetricTests(unittest.TestCase):
 def test_identical_images(self):
  x=torch.rand(3,32,32); self.assertEqual(psnr(x,x),float('inf')); self.assertAlmostEqual(ssim(x,x),1.0,places=6)
if __name__=='__main__': unittest.main()
