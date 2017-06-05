import filecmp
import os
import unittest

from parse_input import get_problems, write_problem


class TestViasualSolver(unittest.TestCase):

    def test_get_problems(self):
        problems = []
        folder_name = '../Problems'
        out_file = 'verify_problem.txt'

        get_problems(folder_name, problems)
        
        for i in range(len(problems)):
            in_file = os.path.join(folder_name, '%d.txt' % i)
            
            write_problem(problems[i], out_file)

            self.assertTrue(filecmp.cmp(in_file, out_file))
        
        os.remove(out_file)
            


if __name__ == '__main__':
    unittest.main()