"""Tests for LRU cache implementations in lrucacheextension."""

import threading
import unittest

from tables.lrucacheextension import ObjectCache
from tables.tests import common


class TestObjectCacheUpdateSlot(unittest.TestCase):
    """Regression tests for ObjectCache.updateslot_ (GitHub issue #1254).

    The bug: when fewer than 10 slots are occupied and a new item would push
    cachesize over maxcachesize, the eviction while-loop selects empty slots
    because their atime (0) is always less than any occupied slot's atime (>=1
    from incseqn).  removeslot_ on an empty slot is a no-op, so cachesize
    never decreases and the loop spins forever.
    """

    TIMEOUT = 5.0  # seconds; generous for slow CI machines

    def _setitem_in_thread(self, cache, key, value, size):
        """Call cache.setitem in a daemon thread, returning (thread, result)."""
        result = []

        def _run():
            cache.setitem(key, value, size)
            result.append("done")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t, result

    def test_eviction_with_single_occupied_slot(self):
        """Adding an item that exceeds maxcachesize must not loop forever.

        Scenario: 1 occupied slot (size=90), nslots=2, maxcachesize=100.
        Adding a second item of size=20 requires evicting the first item
        (90+20=110 > 100).  With the bug, the loop always picks the empty
        slot (atime=0) instead of the occupied one (atime>=1) and hangs.
        """
        cache = ObjectCache(2, 100, "test_single_occupied")
        cache.setitem("key1", "v1", 90)

        t, result = self._setitem_in_thread(cache, "key2", "v2", 20)
        t.join(self.TIMEOUT)

        self.assertTrue(
            result,
            "ObjectCache.setitem hung (infinite loop in updateslot_ "
            "when fewer than 10 slots are occupied — see issue #1254)",
        )

    def test_eviction_with_nine_occupied_slots(self):
        """Same scenario with 9 occupied slots (still < 10, still triggers bug)."""
        max_size = 1000
        cache = ObjectCache(10, max_size, "test_nine_occupied")

        # Fill 9 slots, each with size 100 → cachesize = 900
        for i in range(9):
            cache.setitem(f"key{i}", f"v{i}", 100)

        # Adding size=200 requires eviction: 900+200 > 1000
        t, result = self._setitem_in_thread(cache, "key_new", "v_new", 200)
        t.join(self.TIMEOUT)

        self.assertTrue(
            result,
            "ObjectCache.setitem hung with 9 occupied slots (issue #1254)",
        )

    def test_eviction_with_ten_or_more_occupied_slots(self):
        """With >= 10 occupied slots the original code works; verify no regression."""
        max_size = 2000
        cache = ObjectCache(20, max_size, "test_ten_occupied")

        # Fill 10 slots, each with size 180 → cachesize = 1800
        for i in range(10):
            cache.setitem(f"key{i}", f"v{i}", 180)

        # Adding size=500 requires eviction: 1800+500 > 2000
        t, result = self._setitem_in_thread(cache, "key_new", "v_new", 500)
        t.join(self.TIMEOUT)

        self.assertTrue(result, "ObjectCache.setitem hung with 10 occupied slots")

    def test_normal_insertion_no_eviction_needed(self):
        """Basic sanity: insertion that does not exceed maxcachesize must work."""
        cache = ObjectCache(4, 100, "test_no_eviction")
        cache.setitem("k1", "v1", 10)
        cache.setitem("k2", "v2", 10)
        # Total = 20, well under 100 — no eviction, no risk of loop
        nslot = cache.getslot("k1")
        self.assertGreaterEqual(nslot, 0)


def suite():
    theSuite = unittest.TestSuite()
    theSuite.addTest(common.make_suite(TestObjectCacheUpdateSlot))
    return theSuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
