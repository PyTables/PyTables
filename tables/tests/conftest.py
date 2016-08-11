import os
import pytest
import tables as tb
from tempfile import TemporaryDirectory


@pytest.yield_fixture
def pytables_file():
    with TemporaryDirectory(prefix='tables') as dir:
        yield tb.open_file(os.path.join(dir, 'test.h5'), 'w', title='A test file')


@pytest.fixture
def array(pytables_file):
    return pytables_file.create_array(pytables_file.root, 'array', [1, 2], title="Array example")


@pytest.fixture
def table(pytables_file):
    return pytables_file.create_table(pytables_file.root, 'table', {'var1': tb.IntCol()}, "Table example")


@pytest.fixture
def file_with_attribute(pytables_file):
    root._v_attrs.testattr = 41
    return root


@pytest.fixture
def group(pytables_file):
    return pytables_file.create_group(root, 'agroup', "Group title")


@pytest.fixture
def array_in_group(group):
    a = group.create_array(group, 'anarray1', [1, 2, 3, 4, 5, 6, 7], "Array title 1")
    a.attrs.testattr = 42
    return a

