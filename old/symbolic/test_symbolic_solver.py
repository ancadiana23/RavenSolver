"""
Test functions used for reading and solving the problems
"""
import unittest

from temporal_agent import create_SDR
from read_problems import unique_attributes


class TestSymbolicSolver(unittest.TestCase):
    """
    Tester class
    """
    def test_create_sdr(self):
        """
        Test the create_sdr function
        """
        window = {'A': {'1': 2, '2': 1}, 'C': {'2': 3}, 'B': {'2': 2}}
        attributes = {'A': {'1': set([1, 2, 3]), '2': set([1, 2])},
                      'B': {'1': set([1, 2, 3, 4]), '2': set([1, 2, 3])},
                      'C': {'1': set([1, 2, 3]), '2': set([1, 2, 3])}}
        sdr = []
        create_SDR(window, attributes, sdr)
        self.assertEqual(sdr, [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])


    def test_unique_attributes(self):
        """
        Test the unique_attributes function
        """
        problem1 = {'A': {'1': 'x'},
                    'B': {'1': 'y', '2': 'z'},
                    'C': {'2': 'x', '3': 'y'}}
        problem2 = {'A': {'2': 'x'},
                    'B': {'1': 'x', '2': 'z'},
                    'C': {'1': 'z', '2': 'xx', '3': 'y'},
                    'D': {'4': 't'}}
        correct_attributes = {'A': {'1': set(['x']), '2': set(['x'])},
                              'B': {'1': set(['x', 'y']), '2': set(['z'])},
                              'C': {'1': set(['z']), '2': set(['x', 'xx']), '3': set(['y'])},
                              'D': {'4': set(['t'])}}
        attributes = {}
        unique_attributes(problem1, attributes)
        unique_attributes(problem2, attributes)

        self.assertEqual(attributes, correct_attributes)


if __name__ == '__main__':
    unittest.main()
