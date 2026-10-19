import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from dehaze.models.v0_baseline import V0ResidualCNN, parameter_count
class ModelTests(unittest.TestCase):
 def test_forward_shape_and_range(self):
  model=V0ResidualCNN(); x=torch.rand(2,3,64,64); y=model(x); self.assertEqual(y.shape,x.shape); self.assertTrue(torch.all((y>=0)&(y<=1))); self.assertGreater(parameter_count(model),0)
if __name__=='__main__': unittest.main()
