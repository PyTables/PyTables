import numarray.records

def normalize_format(fmt):
    """Normalize format to follow numarray conventions."""
    # Remove shape '()' at the forefront which is equivalent to an scalar
    if fmt[:2] == '()':
        fmt = fmt[2:]
    # Accept 'S' as a synonym of 'a'
    if fmt.find('S') >= 0:
        fmt = fmt.replace('S', 'a')
    return fmt


def getIter(object):
    """Return an iterator (if any) for object
    """
    iterator = None
    try:
        iterator = iter(object)
    except TypeError:
        pass
    return iterator

#
# Methods for flatten the buffer structure descriptions
#
def flattenDescr(descr, check=False):
    """Flatten a descr description of a buffer.

    Names of nested fields are returned as a level1/level2/.../levelN path.
    If ``check`` is True the function returns None when it finds some element
    with an incorrect format. This is not strictely necessary, but it is
    useful for testing purposes.
    """
    i = getIter(descr)
    if not i:
        return

    try:
        item = i.next()
        while item:
            if isinstance(item, tuple) and len(item) == 2 and \
            (isinstance(item[1], str) or isinstance(item[1], list)) and \
            isinstance(item[0], str):
                if isinstance(item[1], str):
                    yield item
                else:
                    for c in flattenDescr(item[1], check):
                        if c == None:
                            yield c
                        else:
                            name = '%s/%s' % (item[0], c[0])
                            yield (name, c[1])
            else:
                if check:
                    yield None
            item = i.next()
    except StopIteration:
        pass

def flattenFormats(formats, check=False):
    """Flatten a formats description of a buffer.

    If ``check`` is True the function returns None when it finds some
    element with an incorrect format, i.e. an element that is neither a
    string nor a sequence. This is not strictely necessary, but it is
    useful for testing purposes.
    """
    i = getIter(formats)
    if not i:
        return

    try:
        item = i.next()
        while item:
            if isinstance(item, str):
                yield normalize_format(item)
            elif isinstance(item, list) or isinstance(item, tuple):
                for c in flattenFormats(item, check):
                    yield c
            else:
                if check:
                    yield None
            item = i.next()
    except StopIteration:
        pass

def flattenNames(names, check=False):
    """Flatten a names description of a buffer.

    Names of nested fields are returned with its full path, i.e.
    level1/level2/.../levelN.
    If ``check`` is True the function returns None when it finds
    some element with an incorrect format, i.e. an element that is
    neither a string nor a 2-tuple. This is not strictely necessary, but
    it is useful for testing purposes.
    """
    i = getIter(names)
    if not i:
        return

    try:
        item = i.next()
        while item:
            if isinstance(item, str):
                yield item
            elif isinstance(item, tuple) and len(item) == 2\
            and isinstance(item[0], str) and isinstance(item[1], list):
                for c in flattenNames(item[1], check):
                    if c == None:
                        yield c
                    else:
                        yield '%s/%s' % (item[0], c)
            else:
                if check:
                    yield None
            item = i.next()
    except StopIteration:
        pass


#
# Methods to get a given description list from another one
#
def getDescr(names, formats):
    """Create a descr description by mixing formats and names lists.

    This method assumes that names and formats descriptions structure
    are good (i.e. that checkNames and checkFormats nesterecords methods
    raised no errors).
    """
    if not names:
        names = [item for item in makeNamesFromFormats(formats)]

    if type(formats) == str and type(names) == str:
        yield (names, formats)
        raise StopIteration

    if len(formats) != len(names):
        raise ValueError("""The formats and names structure don't match!""")

    mix = zip(names, formats)
    i = getIter(mix)
    if not i:
        return

    try:
        (name, fmt) = i.next()
        while (name, fmt):
            if isinstance(name, str) and isinstance(fmt, str):
                yield (name, fmt)
            else:
                l = []
                for (a, b) in getDescr(name[1], fmt):
                    l.append((a,b))
                yield (name[0], l)
            (name, fmt) = i.next()
    except StopIteration:
        pass

def makeNamesFromFormats(formats):
    """Create a names description from a formats one.

    Field names are generated automatically as c1, c2 and so on.
    This method assumes that formats description structure is good (i.e.
    that nestedrecords.checkFormats method raised no errors).
    """
    i = getIter(formats)
    if not i:
        return

    try:
        c = 0
        item = i.next()
        while item:
            c = c +1
            name = 'c%s' % c
            if isinstance(item, str):
                yield name
            else:
                l = []
                for a in makeNamesFromFormats(item):
                    l.append(a)
                yield (name, l)
            item = i.next()
    except StopIteration:
        pass

def getNamesFromDescr(descr):
    """Extract field names from a description sequence.

    Given a descr sequence, this method retrieves the embeded names list.
    This method assumes that descr description structure is good (i.e.
    that nestedrecords.checkDescr method raised no errors).
    """
    i = getIter(descr)
    if not i:
        return

    try:
        item = i.next()
        while item:
            if isinstance(item[1], str):
                yield item[0]
            else:
                l = []
                for j in getNamesFromDescr(item[1]):
                    l.append(j)
                r = (item[0], l)
                yield r
            item = i.next()
    except StopIteration:
        pass

def getFormatsFromDescr(descr):
    """Extract field formats from a description sequence.

    Given a descr sequence, this method retrieves the embeded formats list.
    This method assumes that descr description structure is good (i.e.
    that nestedrecords.checkDescr method raised no errors).
    """
    i = getIter(descr)
    if not i:
        return

    try:
        item = i.next()
        while item:
            item1 = item[1]
            if isinstance(item1, str):
                yield normalize_format(item1)
            else:
                l = []
                for j in getFormatsFromDescr(item1):
                    l.append(j)
                yield l
            item = i.next()
    except StopIteration:
        pass

# Methods to deal with descr description
def getFieldDescr(fieldName, descr):
    """Retrieve the descr list corresponding to a given field.

    For nested fields the fieldName is passed as x/y...
    This method assumes that descr description structure is good (i.e.
    that nestedrecords.checkDescr method raised no errors).
    """
    i = getIter(descr)
    if not i:
        return

    try:
        sw = ''
        item = i.next()
        while item:
            if fieldName == item[0]:
                yield item
                break
            if isinstance(item[1], list):
                if fieldName.startswith('%s/' %item[0]):
                    sw = item[0]
                else:
                    item = i.next()
                    continue
                [trash, newField] = fieldName.split(sw + '/')
                for c in getFieldDescr(newField, item[1]):
                    sw = '%s/%s' % (sw, c[0])
                    yield (sw, c[1])
            item = i.next()
    except StopIteration:
        pass

#
# Methods to deal with the names description
#
def getSubNames(names):
    """Retrieve the list of all names and sub-names in a names list.

    For nested fields all sub-field names are returned. For instance,
    a field x/y/z will be returned as x, y, z.
    This method is used to check that field names don't contain
    '/' characters.
    This method assumes that names description structure is good (i.e.
    that nestedrecords.checkNames method raised no errors).
    """
    i = getIter(names)
    if not i:
        return

    try:
        item = i.next()
        while item:
            if isinstance(item, str):
                yield item
            else:
##            elif isinstance(item, tuple) and len(item) == 2:
                yield item[0]
                for c in getSubNames(item[1]):
                    yield c
            item = i.next()
    except StopIteration:
        pass

def checkNamesUniqueness(names):
    """At every level of the names description check that names are unique.

    This method assumes that names description structure is good (i.e.
    that nestedrecords.checkNames method raised no errors.).
    This is a recursive function but it doesn't use generators.
    """
    topNames, deeperNames = getLevelNames(names)
##    print topNames
##    print deeperNames
    for name in topNames[:-1]:
        if topNames.count(name) > 1:
            raise ValueError("""\Names at every level must be unique!""")
    if deeperNames:
        checkNamesUniqueness(deeperNames)

def getLevelNames(names):
    """Retrieve the list of names in a given level.

    This method assumes that names description structure is good (i.e.
    that nestedrecords.checkNames method raised no errors.).
    """
    topNames = []
    deeperNames = []
    for item in names:
        if isinstance(item, str):
            topNames.append(item)
        else:
            topNames.append(item[0])
            # Names immediately under the current level must be
            # qualified with the current level full name
            for j in item[1]:
                if isinstance(j, str):
                    subname = '%s/%s' % (item[0], j)
                else:  # j is a 2-tuple
                    jlist = list(j)
                    jlist[0] = '%s/%s' % (item[0], jlist[0])
                    subname = tuple(jlist)
                deeperNames.append( subname)
    return topNames, deeperNames


#
# Methods to check the correctness of the buffer structure
#
def zipBufferDescr(row, structure):
    """Zip a buffer row with its `descr` description.

    This function is used to check if buffers have a consistent format.
    This is done by applying the function on every row of the buffer.
    The function zips the buffer row with the buffer descr description
    in a recursive way, till a flat list of 2-tuples is obtained.
    Each 2-tuple contains the value of a field and its description.
    Recursion is needed to deal with nested fields of the buffer.
    This method assumes that descr description structure is good (i.e.
    that nestedrecords.checkDescr method raised no errors.).
    """

##    print 'row **', row.__class__, row
##    print 'structure **', structure
    if len(row) != len(structure):
        raise ValueError("""The row structure doesn't match that provided"""\
                    """ by the format specification""")
    mix = zip(row, structure)
    i = getIter(mix)
    if not i:
        return

    try:
        (value, descr) = i.next()
        while (value, descr):
            fmt = descr[1]
            if isinstance(fmt, str):
                yield (value, fmt)
            else:
                for (a, b) in zipBufferDescr(value, descr[1]):
                    yield (a, b)
            (value, descr) = i.next()
    except StopIteration:
        pass

def flattenArraysList(array_list, descr, flat_array_list):
    """Flatten a buffer made of arrays list.
    """
    if isinstance(array_list, numarray.records.RecArray):
        raise TypeError("""``arrayList`` cannot be a recarray""")

    if isinstance(descr, tuple) and len(descr) == 2 and \
        isinstance(descr[1], str) and isinstance(descr[0], str):
        flat_array_list.append(array_list)
    elif isinstance(descr, tuple) and len(descr) == 2 and \
        isinstance(descr[1], list) and isinstance(descr[0], str):
        for sdpos, sdescr in enumerate(descr[1]):
            flattenArraysList(array_list[sdpos], sdescr, flat_array_list)
    elif isinstance(descr, list):
        for sdpos, sdescr in enumerate(descr):
            flattenArraysList(array_list[sdpos], sdescr, flat_array_list)
