import pytest
import agentpy as ap

from agentpy.tools import *


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

    m = make_matrix([2, 2], loc_type=ap.Location, list_type=MyList)
    assert m.__repr__() == "mylist [mylist [Location (0, 0), Location (0, 1)]"\
                           ", mylist [Location (1, 0), Location (1, 1)]]"


def test_attr_dict():

    ad = ap.AttrDict({'a': 1})
    ad.b = 2

    assert ad.a == 1
    assert ad.b == 2
    assert ad.a == ad['a']
    assert ad.b == ad['b']
    assert ad.__repr__() == "AttrDict {'a': 1, 'b': 2}"
    assert ad._short_repr() == "AttrDict {2 entries}"
