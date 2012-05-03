import re


def update_docstring(app, what, name, obj, options, lines):

    section_re = re.compile(r'(~+)')

    previous_line = lines[0].strip()
    for i, line in enumerate(lines[1:], 1):
        stripped_line = line.strip()
        match = re_search(section_re, line)
        if len(stripped_line) == len(match) and \
                            len(stripped_line) == len(previous_line):
            indent_level = find_indent_level(line[i-1])
            if previous_line.lower() == 'parameters':
                make_field_list(lines, i-1, indent_level, previous_line)
            else:
                make_rubric()


def find_indent_level(line):
    """Return the number of leading spaces in line.
    """
    stripped_line = line.lstrip()
    if len(stripped_line) == 0:
        return 0
    else:
        return len(line) - len(stripped_line)


def re_search(regex, string):
    """Runs regex.search(string) and returns an empty string if there is no
    match.  Otherwise, returns match.group(1).
    """
    match = regex.search(string)
    if match is None:
        return ''
    else:
        return match.group(1)


def make_field_list(lines, line_number, indent_level, field_title):

    lines[line_number] = ':' + field_title + ':'
    lines[line_number+1] = ''
    line_number += 2
    for line_number, line in enumerate(lines[line_number:], line_number):
        if line.strip() != '' and find_indent_level(line) == indent_level:
            break
    for line_number, line in enumerate(lines[line_number:], line_number):
        if line.strip() == '':
            return
        lines[line_number] = ' ' * 4 + line



def make_rubric():
    pass








def setup(app):

    app.connect('autodoc-process-docstring', update_docstring)



if __name__ == '__main__':
    import unittest

    class FindIndentLevelTests(unittest.TestCase):

        def test_empty_string(self):
            string = ''
            indent = find_indent_level(string)
            self.assertEqual(indent, 0)

        def test_all_spaces(self):
            string = '    '
            indent = find_indent_level(string)
            self.assertEqual(indent, 0)

        def test_left_justified(self):
            string = 'abcd'
            indent = find_indent_level(string)
            self.assertEqual(indent, 0)

        def test_with_indent(self):
            string = '    abcd   '
            indent = find_indent_level(string)
            self.assertEqual(indent, 4)


    class ReSearchTests(unittest.TestCase):

        def setUp(self):
            self.regex = re.compile(r'([0-9]+)')

        def test_no_match(self):
            string = 'abc'
            match = re_search(self.regex, string)
            self.assertEqual(match, '')

        def test_match(self):
            string = 'abc123abc'
            match = re_search(self.regex, string)
            self.assertEqual(match, '123')


    class MakeFieldListTests(unittest.TestCase):

        def setUp(self):
            self.lines = ['',
                          'Section',
                          '~~~~~~~',
                          '',
                          'parameter 1',
                          '    parameter 1 details',
                          '',
                          'parameter 2',
                          '    parameter 2 details',
                          ' ' * 5,
                          'Other text...']


        def test_case1(self):
            field_title = 'Parameters'
            make_field_list(self.lines, 1, 0, field_title)
            self.assertEqual(self.lines[1], ':Parameters:')
            self.assertEqual(self.lines[2], '')
            self.assertEqual(self.lines[3], '')
            self.assertEqual(self.lines[4], '    parameter 1')
            self.assertEqual(self.lines[5], '        parameter 1 details')
            self.assertEqual(self.lines[6], '')
            self.assertEqual(self.lines[7], 'parameter 2')

        def test_case2(self):
            field_title = 'Parameters'
            del self.lines[6]
            make_field_list(self.lines, 1, 0, field_title)
            self.assertEqual(self.lines[1], ':Parameters:')
            self.assertEqual(self.lines[2], '')
            self.assertEqual(self.lines[3], '')
            self.assertEqual(self.lines[4], '    parameter 1')
            self.assertEqual(self.lines[5], '        parameter 1 details')
            self.assertEqual(self.lines[6], '    parameter 2')
            self.assertEqual(self.lines[7], '        parameter 2 details')
            self.assertEqual(self.lines[8], ' ' * 5)
            self.assertEqual(self.lines[9], 'Other text...')

        def test_no_fields_blank_line(self):
            field_title = 'Parameters'
            self.lines = self.lines[1:4]
            make_field_list(self.lines, 0, 0, field_title)
            self.assertEqual(self.lines[0], ':Parameters:')
            self.assertEqual(self.lines[1], '')
            self.assertEqual(self.lines[2], '')
            self.assertEqual(len(self.lines), 3)

        def test_no_fields_no_blank_line(self):
            field_title = 'Parameters'
            self.lines = self.lines[1:3]
            make_field_list(self.lines, 0, 0, field_title)
            self.assertEqual(self.lines[0], ':Parameters:')
            self.assertEqual(self.lines[1], '')
            self.assertEqual(len(self.lines), 2)


    unittest.main()