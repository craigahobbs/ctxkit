# Licensed under the MIT License
# https://github.com/craigahobbs/ctxkit/blob/main/LICENSE

"""
ctxkit unified diff utilities
"""

import re


# Apply a unified diff to a file
def apply_diff(original_text, diff_text):
    # Parse and apply hunks in reverse order to preserve line numbers
    original_lines = original_text.splitlines()
    result_lines = list(original_lines)
    for old_start, old_count, new_lines in reversed(list(parse_unified_diff(diff_text))):
        # Adjust start as necessary
        start_idx = max(old_start - 1, 0)
        if result_lines and result_lines[start_idx] != new_lines[0]:
            start_idx = _find_line(result_lines, new_lines[0], start_idx, 5)

        # Adjust end as necessary
        end_idx = min(start_idx + old_count, len(result_lines) - 1)
        if result_lines and result_lines[end_idx] != new_lines[-1]:
            end_idx = _find_line(result_lines, new_lines[-1], end_idx, 5)

        # Splice-in the new lines
        result_lines[start_idx:end_idx + 1] = new_lines

    return '\n'.join(result_lines) + '\n'


def _find_line(lines, line, idx, count):
    # Search for the line behind and forward
    backward_idx = None
    forward_idx = None
    for search_idx in range(idx - 1, max(idx - count, -1), -1):
        if lines[search_idx] == line:
            backward_idx = search_idx
            break
    for search_idx in range(idx + 1, min(idx + count, len(lines) - 1)):
        if lines[search_idx] == line:
            forward_idx = search_idx
            break

    # Return the closest found line
    if backward_idx is not None and forward_idx is not None:
        if idx - backward_idx < forward_idx - idx:
            return backward_idx
        else:
            return forward_idx
    elif backward_idx is not None:
        return backward_idx
    elif forward_idx is not None:
        return forward_idx

    # Line not found
    return idx


# Parse unified diff text, yielding hunks as (old_start, old_count, new_lines)
def parse_unified_diff(diff_text):
    old_start = None
    old_count = None
    new_lines = []
    for line in diff_text.splitlines():
        # New hunk header? Yield the previous hunk and start a new one
        hunk_match = _R_HUNK_HEADER.match(line)
        if hunk_match:
            if old_start is not None:
                yield (old_start, old_count, new_lines)
            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2)) if hunk_match.group(2) is not None else 1
            new_lines = []

        # Skip until we're inside a hunk
        elif old_start is None:
            continue

        # Added or context line
        elif line.startswith('+') or line.startswith(' '):
            new_lines.append(line[1:])

    # Yield the final hunk
    if old_start is not None:
        yield (old_start, old_count, new_lines)


# Unified diff format regex
_R_HUNK_HEADER = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@')
