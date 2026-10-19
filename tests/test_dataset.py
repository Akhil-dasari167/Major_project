import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dehaze.data.dataset import discover_nh_haze_pairs, PairedDehazeDataset, EpochShuffleSampler
from dehaze.data.splits import SplitManifest, make_split, validate_split
class DatasetTests(unittest.TestCase):
 def test_pairs_and_split(self):
  pairs=discover_nh_haze_pairs(Path(__file__).resolve().parents[1]/'data'/'NH-HAZE'); self.assertEqual(len(pairs),55); split=make_split([x.image_id for x in pairs]); validate_split(split,[x.image_id for x in pairs]); self.assertEqual((len(split.train),len(split.validation),len(split.test)),(39,8,8)); self.assertFalse(set(split.train)&set(split.test))
 def test_overlapping_split_is_rejected(self):
  with self.assertRaises(ValueError): validate_split(SplitManifest(seed=1,train=['01'],validation=['01'],test=['02']))
 def test_patch_shape(self):
  pairs=discover_nh_haze_pairs(Path(__file__).resolve().parents[1]/'data'/'NH-HAZE'); sample=PairedDehazeDataset(pairs[:1],patch_size=64,seed=7)[0]; self.assertEqual(tuple(sample['input'].shape),(3,64,64)); self.assertEqual(tuple(sample['target'].shape),(3,64,64))
 def test_epoch_sampler_is_restart_safe(self):
  pairs=discover_nh_haze_pairs(Path(__file__).resolve().parents[1]/'data'/'NH-HAZE'); dataset=PairedDehazeDataset(pairs[:5]); sampler=EpochShuffleSampler(dataset,seed=42); sampler.set_epoch(3); first=list(sampler); sampler.set_epoch(3); self.assertEqual(first,list(sampler))
if __name__=='__main__': unittest.main()
