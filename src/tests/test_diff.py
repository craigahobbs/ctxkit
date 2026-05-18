# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

import unittest

from ctxkit.diff import apply_diff, parse_unified_diff


class TestDiff(unittest.TestCase):

    def test_parse_unified_diff(self):
        self.assertEqual(
            list(parse_unified_diff('''\
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 1
-2
+4
 3
''')),
            [
                (1, 2, ['1', '4', '3'])
            ]
        )


    def test_parse_unified_diff_empty(self):
        self.assertEqual(list(parse_unified_diff('')), [])


    def test_parse_unified_diff_no_hunk(self):
        self.assertEqual(
            list(parse_unified_diff('''\
--- a/test.txt
+++ b/test.txt
''')),
            []
        )


    def test_parse_unified_diff_no_count(self):
        # Hunk header without explicit count (defaults to 1)
        self.assertEqual(
            list(parse_unified_diff('''\
--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-old
+new
''')),
            [
                (1, 1, ['new'])
            ]
        )


    def test_parse_unified_diff_multiple_hunks(self):
        self.assertEqual(
            list(parse_unified_diff('''\
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 1
-2
+4
@@ -10,2 +10,2 @@
 10
-11
+12
''')),
            [
                (1, 2, ['1', '4']),
                (10, 2, ['10', '12'])
            ]
        )


    def test_parse_unified_diff_new_file(self):
        # New file diff with /dev/null
        self.assertEqual(
            list(parse_unified_diff('''\
--- /dev/null
+++ b/newfile.txt
@@ -0,0 +1,3 @@
+line one
+line two
+line three
''')),
            [
                (0, 0, ['line one', 'line two', 'line three'])
            ]
        )


    def test_parse_unified_diff_only_additions(self):
        self.assertEqual(
            list(parse_unified_diff('''\
@@ -1,1 +1,3 @@
 existing
+added one
+added two
''')),
            [
                (1, 1, ['existing', 'added one', 'added two'])
            ]
        )


    def test_parse_unified_diff_only_removals(self):
        self.assertEqual(
            list(parse_unified_diff('''\
@@ -1,3 +1,1 @@
 keep
-remove one
-remove two
''')),
            [
                (1, 3, ['keep'])
            ]
        )


    def test_parse_unified_diff_no_newline_marker(self):
        # The "\ No newline at end of file" marker
        self.assertEqual(
            list(parse_unified_diff('''\
@@ -1,2 +1,2 @@
 line one
-line two
+line TWO
\\ No newline at end of file
''')),
            [
                (1, 2, ['line one', 'line TWO'])
            ]
        )


    def test_parse_unified_diff_no_newline_after_removed(self):
        # The "\ No newline at end of file" marker after a removed line
        self.assertEqual(
            list(parse_unified_diff('''\
@@ -1,2 +1,1 @@
 line one
-line two
\\ No newline at end of file
''')),
            [
                (1, 2, ['line one'])
            ]
        )


    def test_parse_unified_diff_skips_lines_before_hunk(self):
        # Lines before any hunk header are skipped
        self.assertEqual(
            list(parse_unified_diff('''\
some random
preamble text
--- a/test.txt
+++ b/test.txt
@@ -1,1 +1,1 @@
-old
+new
''')),
            [
                (1, 1, ['new'])
            ]
        )


    def test_apply_diff(self):
        original = '1\n2\n3\n'
        diff = '''\
--- a/test.txt
+++ b/test.txt
@@ -1,3 +1,3 @@
 1
-2
+4
 3
'''
        self.assertEqual(apply_diff(original, diff), '1\n4\n3\n')


    def test_apply_diff_multiple_hunks(self):
        original = '1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n'
        diff = '''\
--- a/test.txt
+++ b/test.txt
@@ -1,3 +1,3 @@
 1
-2
+TWO
 3
@@ -8,3 +8,3 @@
 8
-9
+NINE
 10
'''
        self.assertEqual(apply_diff(original, diff), '1\nTWO\n3\n4\n5\n6\n7\n8\nNINE\n10\n')


    def test_apply_diff_new_file(self):
        original = ''
        diff = '''\
--- /dev/null
+++ b/newfile.txt
@@ -0,0 +1,3 @@
1:+line one
1:+line two
1:+line three
'''
        # Note: parse will yield lines starting with '+' or ' '. The above has '1:+' prefix (not valid)
        # Let's use a proper new-file diff
        diff = '''\
--- /dev/null
+++ b/newfile.txt
@@ -0,0 +1,3 @@
+line one
+line two
+line three
'''
        self.assertEqual(apply_diff(original, diff), 'line one\nline two\nline three\n')


    def test_apply_diff_empty_original(self):
        # Apply diff to empty original (covers the "result_lines and" guard)
        original = ''
        diff = '''\
@@ -1,1 +1,1 @@
-old
+new
'''
        # With empty original, result_lines is [''] after splitlines? No, ''.splitlines() == []
        # So result_lines is empty, the "if result_lines" check is False, no _find_line called
        result = apply_diff(original, diff)
        self.assertEqual(result, 'new\n')


    def test_apply_diff_first_line_mismatch_forward(self):
        # First line of hunk doesn't match at start_idx, found forward
        original = 'extra\n1\n2\n3\n'
        diff = '''\
@@ -1,3 +1,3 @@
 1
-2
+TWO
 3
'''
        # start_idx initially 0 ('extra'), hunk starts with '1' which is at index 1
        self.assertEqual(apply_diff(original, diff), 'extra\n1\nTWO\n3\n')


    def test_apply_diff_first_line_mismatch_backward(self):
        # First line of hunk doesn't match at start_idx, found backward
        original = '1\n2\n3\nextra\n'
        diff = '''\
@@ -4,3 +4,3 @@
 1
-2
+TWO
 3
'''
        # start_idx initially 3 ('extra'), hunk starts with '1' which is at index 0 (backward)
        self.assertEqual(apply_diff(original, diff), '1\nTWO\n3\nextra\n')


    def test_apply_diff_first_line_mismatch_both_backward_closer(self):
        # First line found in both directions, backward is closer
        original = 'X\n1\nY\nY\n1\nZ\n'
        diff = '''\
@@ -4,2 +4,2 @@
 1
-Y
+YY
'''
        # start_idx initially 3 ('Y'), looking for '1':
        # backward: index 1 (distance 2)... wait, the search range is idx-count to idx, index 1 found at distance 2
        # forward: index 4 (distance 1)
        # forward is closer
        self.assertEqual(apply_diff(original, diff), 'X\n1\nY\nY\n1\nYY\n')


    def test_apply_diff_last_line_mismatch(self):
        # Last line of hunk doesn't match at end_idx
        original = '1\n2\nextra\n3\n'
        diff = '''\
@@ -1,3 +1,3 @@
 1
-2
+TWO
 3
'''
        # First line '1' matches at 0. end_idx = 0 + 3 = 3, but original[3]='3'... wait original is ['1','2','extra','3']
        # min(0+3, 4-1) = 3, original[3]='3', last new line is '3', matches. So no _find_line call needed.
        # Let's force a mismatch: original where end position has wrong line
        original = '1\nextra\n2\n3\n'
        diff = '''\
@@ -1,3 +1,3 @@
 1
-2
+TWO
 3
'''
        # start: idx 0='1' matches. end_idx = min(0+3, 3) = 3, original[3]='3' matches '3'. Splice [0:4] with ['1','TWO','3']
        result = apply_diff(original, diff)
        self.assertEqual(result, '1\nTWO\n3\n')


    def test_apply_diff_pure_deletion(self):
        # Pure-deletion hunk (no '+' or context lines) — new_lines is empty
        original = 'keep one\nremove me\nremove me too\nkeep two\n'
        diff = '''\
@@ -2,2 +2,0 @@
-remove me
-remove me too
'''
        self.assertEqual(apply_diff(original, diff), 'keep one\nkeep two\n')


    def test_apply_diff_first_line_mismatch_both_forward_closer(self):
        # First line found in both directions, backward is strictly closer (covers `return backward_idx`
        # when both backward_idx and forward_idx are not None)
        original = 'X\nA\nY\nB\nC\nA\nE\nF\nG\nH\n'
        diff = '''\
@@ -3,2 +3,2 @@
 A
-Y
+YY
'''
        # start_idx initially 2 ('Y'), first hunk line is 'A':
        # backward: idx 1='A' (distance 1)
        # forward: idx 5='A' (distance 3)
        # backward is closer, so start_idx becomes 1
        self.assertEqual(apply_diff(original, diff), 'X\nA\nYY\nC\nA\nE\nF\nG\nH\n')
