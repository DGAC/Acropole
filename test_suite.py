import unittest
from pathlib import Path


def load_tests_file(src_path: str, test_file_pattern: str) -> []:
    tests_file = []
    for file_to_test in Path(src_path).rglob(test_file_pattern):
        tests_file.append(str(file_to_test.as_posix()).replace("/", ".").replace(".py", ""))
    return tests_file


def get_suite_tests(lib_to_tests: []) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    for lib in lib_to_tests:
        try:
            # If the module defines a suite() function, call it to get the suite.
            mod = __import__(lib, globals(), locals(), ['suite'])
            suitefn = getattr(mod, 'suite')
            suite.addTest(suitefn())
        except (ImportError, AttributeError):
            # else, just load all the test cases from the module.
            suite.addTest(unittest.defaultTestLoader.loadTestsFromName(lib))
    return suite


# Init lib to test
library_to_tests = load_tests_file('tests', '*_tests.py')

# Run test lib
unittest.TextTestRunner(verbosity=3).run(get_suite_tests(library_to_tests))


