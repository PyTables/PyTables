import re


def update_docstring(app, what, name, obj, options, lines):

    section_re = re.compile(r'(~+)')

    previous_line = lines[0].strip()
    for i, line in enumerate(lines[1:], 1):
        stripped_line = line.strip()
        match = re_search(section_re, line)
        if len(stripped_line) == len(match) and \
                            len(stripped_line) == len(previous_line):
            indent_level = find_indent_level(lines[i])
            if previous_line.lower() == 'parameters':
                make_field_list(lines, i-1, indent_level, previous_line)
            else:
                make_rubric(lines, i-1, indent_level, previous_line)
        previous_line = stripped_line


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
    """Converts a section of lines to a field list.
    The row lines[line_number] becomes the header.  The following lines
    are indented, until a blank or un-indented line is found.
    """
    lines[line_number] = ' ' * indent_level + ':' + field_title + ':'
    lines[line_number+1] = ''
    line_number += 2
    for line_number, line in enumerate(lines[line_number:], line_number):
        if line.strip() != '' and find_indent_level(line) == indent_level:
            break
    for line_number, line in enumerate(lines[line_number:], line_number):
        line_indent_level = find_indent_level(line)
        if line.strip() == '' or line_indent_level < indent_level:
            return
        if line_indent_level == indent_level:
            line = make_bold_var_name(line)
        lines[line_number] = ' ' * 4 + line


def make_bold_var_name(line):
    """Convert a string like 'varname : int' into '**varname** : int',
    or 'varname' into '**varname**'.
    """
    return re.sub(r'(^\s*)([a-zA-Z_0-9]+)((\s*:.*)|(.*))',
                  r'\1**\2**\3', line)


def make_rubric(lines, line_number, indent_level, field_title):
    """Converts lines[line_number] into a rubric directive.
    """
    lines[line_number] = ' ' * indent_level + '.. rubric:: ' + field_title
    lines[line_number+1] = ''


def setup(app):
    app.connect('autodoc-process-docstring', update_docstring)


if __name__ == '__main__':
    import unittest
    import copy


    class UpdateDocstringTests(unittest.TestCase):

        def test_case1(self):
            lines = ['Description of this class...',
                     '',
                     'Parameters',
                     '~~~~~~~~~~',
                     'parm1 : float',
                     '    parm1 description',
                     'parm2',
                     'parm3:{int or something:else}',
                     '',
                     '    New Section',
                     '    ~~~~~~~~~~~',
                     '    Text in the new section...']
            orig_lines = copy.deepcopy(lines)
            update_docstring(None, None, None, None, None, lines)
            for row in [0, 1, 8, 11]:
                self.assertEqual(lines[row], orig_lines[row])
            self.assertEqual(lines[2],  ':Parameters:')
            self.assertEqual(lines[3],  '')
            self.assertEqual(lines[4],  '    **parm1** : float')
            self.assertEqual(lines[5],  '        parm1 description')
            self.assertEqual(lines[6],  '    **parm2**')
            self.assertEqual(lines[7],  '    **parm3**:{int or something:else}')
            self.assertEqual(lines[8],  '')
            self.assertEqual(lines[9],  '    .. rubric:: New Section')
            self.assertEqual(lines[10], '')
            self.assertEqual(lines[11], '    Text in the new section...')

        def test_case2(self):
            lines = ['',
                     'Section title',
                     '~~~~~~~~~~~~~',
                     'Section content',
                     '    Parameters',
                     '    ~~~~~~~~~~',
                     '    parm1',
                     '        description 1',
                     'Not part of parameters']
            update_docstring(None, None, None, None, None, lines)
            self.assertEqual(lines[0], '')
            self.assertEqual(lines[1], '.. rubric:: Section title')
            self.assertEqual(lines[2], '')
            self.assertEqual(lines[3], 'Section content')
            self.assertEqual(lines[4], '    :Parameters:')
            self.assertEqual(lines[5], '')
            self.assertEqual(lines[6], '        **parm1**')
            self.assertEqual(lines[7], '            description 1')
            self.assertEqual(lines[8], 'Not part of parameters')


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

        def test_multiple_matches(self):
            string = 'abc21abc3456'
            match = re_search(self.regex, string)
            self.assertEqual(match, '21')


    class MakeFieldListTests(unittest.TestCase):

        def setUp(self):
            self.lines = ['',
                          'Section',
                          '~~~~~~~',
                          '',
                          'parameter1:int',
                          '    parameter1 details',
                          '',
                          'parameter2',
                          '    parameter2 details',
                          ' ' * 5,
                          'Other text...']


        def test_case1(self):
            field_title = 'Parameters'
            make_field_list(self.lines, 1, 0, field_title)
            self.assertEqual(self.lines[1], ':Parameters:')
            self.assertEqual(self.lines[2], '')
            self.assertEqual(self.lines[3], '')
            self.assertEqual(self.lines[4], '    **parameter1**:int')
            self.assertEqual(self.lines[5], '        parameter1 details')
            self.assertEqual(self.lines[6], '')
            self.assertEqual(self.lines[7], 'parameter2')

        # heading and parameter 1 are indented an extra 4 spaces
        # parameter 2 is un-indented, so it should not be part of the
        # field list
        def test_extra_indent(self):
            field_title = 'Parameters'
            self.lines[1] = ' ' * 4 + self.lines[1]
            self.lines[2] = ' ' * 4 + self.lines[2]
            self.lines[3] = ' ' * 4
            self.lines[4] = ' ' * 4 + self.lines[4]
            self.lines[5] = ' ' * 4 + self.lines[5]
            del self.lines[6]
            make_field_list(self.lines, 1, 4, field_title)
            self.assertEqual(self.lines[1], '    :Parameters:')
            self.assertEqual(self.lines[2], '')
            self.assertEqual(self.lines[3], ' ' * 4)
            self.assertEqual(self.lines[4], '        **parameter1**:int')
            self.assertEqual(self.lines[5], '            parameter1 details')
            self.assertEqual(self.lines[6], 'parameter2')
            self.assertEqual(self.lines[7], '    parameter2 details')
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


    class MakeBoldVarNameTests(unittest.TestCase):

        def test_case1(self):
            string = '    varname1 : int'
            output = make_bold_var_name(string)
            self.assertEqual(output, '    **varname1** : int')

        # verify additional colons are retained
        def test_case2(self):
            string = ' varname:{int, :float: or file}'
            output = make_bold_var_name(string)
            self.assertEqual(output, ' **varname**:{int, :float: or file}')

        # bold all the text if there is no colon
        def test_case3(self):
            string = '  varname2 '
            output = make_bold_var_name(string)
            self.assertEqual(output, '  **varname2** ')


    class MakeRubricTests(unittest.TestCase):

        def setUp(self):
            self.lines = ['',
                          'Section',
                          '~~~~~~~',
                          '',
                          'Some text after the rubric...',
                          'Other text...']

        def test_case1(self):
            make_rubric(self.lines, 1, 0, 'the rubric title')
            self.assertEqual(self.lines[0], '')
            self.assertEqual(self.lines[1], '.. rubric:: the rubric title')
            self.assertEqual(self.lines[2], '')
            self.assertEqual(self.lines[3], '')
            self.assertEqual(self.lines[4], 'Some text after the rubric...')

        def test_case2(self):
            del self.lines[3]
            make_rubric(self.lines, 1, 4, 'the rubric title')
            self.assertEqual(self.lines[0], '')
            self.assertEqual(self.lines[1], '    .. rubric:: the rubric title')
            self.assertEqual(self.lines[2], '')
            self.assertEqual(self.lines[3], 'Some text after the rubric...')


    unittest.main()