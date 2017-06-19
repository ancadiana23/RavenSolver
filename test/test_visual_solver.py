import filecmp
import os
import unittest

from main.parse_images import get_problems, write_problem


class TestViasualSolver(unittest.TestCase):

    def test_get_problems(self):
        folder_name = './Data/Problems'
        out_file = 'verify_problem.txt'

        problems = get_problems(folder_name)

        for i in range(len(problems)):
            in_file = os.path.join(folder_name, '%d.txt' % i)

            write_problem(problems[i], out_file)

            self.assertTrue(filecmp.cmp(in_file, out_file))

        os.remove(out_file)

    def test_get_problems_sdr(self):
        problems = []
        folder_name = './Data/Problems_sdr'
        out_file = 'verify_problem.txt'

        problems = get_problems(folder_name)

        for i in range(len(problems)):
            in_file = os.path.join(folder_name, '%d.txt' % i)

            write_problem(problems[i], out_file)

            self.assertTrue(filecmp.cmp(in_file, out_file))

        os.remove(out_file)


if __name__ == '__main__':
    unittest.main()
