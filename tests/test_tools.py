import pytest
import agentpy as ap

from agentpy.tools import *


def test_InfoStr():
    assert InfoStr('yay').__repr__() == 'yay'


def test_make_list():

    make_list = ap.tools.make_list
    assert make_list('123') == ['123']
    assert make_list(['123']) == ['123']
    assert make_list(None) == []
    assert make_list(None, keep_none=True) == [None]


def test_make_matrix():

    class MyList(list):
        def __repr__(self):
            return f"mylist {super().__repr__()}"

    m = make_matrix([2, 2], list_type=MyList)
    assert m.__repr__() == "mylist [mylist [None, None], mylist [None, None]]"


def test_attr_dict():

    ad = ap.AttrDict({'a': 1})
    ad.b = 2

    assert ad.a == 1
    assert ad.b == 2
    assert ad.a == ad['a']
    assert ad.b == ad['b']
    assert ad._short_repr() == "AttrDict (2 entries)"
    assert AttrDict(None) == {}  # Initialize with None


def test_ListDict():
    x = ListDict([1, 2, 3, 4, 5])
    assert list(x) == [1, 2, 3, 4, 5]
    x.replace(3, 8)
    assert list(x) == [1, 2, 8, 4, 5]
    x.append(9)
    assert list(x) == [1, 2, 8, 4, 5, 9]
    x.append(9)
    assert list(x) == [1, 2, 8, 4, 5, 9]
    x.remove(2)
    assert list(x) == [1, 9, 8, 4, 5]
    x.pop(0)
    assert list(x) == [5, 9, 8, 4]
