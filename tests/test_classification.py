import unittest

from src.pointnet_classification import *

class TestDataSetCreation(unittest.TestCase):
    def test_set_dataset_directories_exists(self):
        data_dir = set_dataset_directories()
        self.assertTrue(os.path.isdir(data_dir), "The returned directory does not exist.")

if __name__ == '__main__':
    unittest.main()