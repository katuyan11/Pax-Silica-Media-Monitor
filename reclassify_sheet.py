"""Re-label every existing row in the Clean_Data tab with the fixed classifier.

Rows already in the sheet were labeled with the old substring matching, so
without this step the dashboard would mix old and new labels.

Usage (same environment variables as fetch_and_append.py):
    python reclassify_sheet.py            # dry run: shows what would change
    python reclassify_sheet.py --apply    # writes the new labels to the sheet

Only the `themes` and `stance` columns are touched.
"""

import sys
from collections import Counter

from fetch_and_append import (
    EXPECTED_COLUMNS,
    classify_stance,
    classify_themes,
    get_google_sheet,
)


def col_letter(index: int) -> str:
    """0-based column index -> A1 letter (supports A to Z, enough for 9 columns)."""
    return chr(ord("A") + index)


def main():
    apply_changes = "--apply" in sys.argv

    sheet = get_google_sheet()
    records = sheet.get_all_records()
    print(f"Rows read from sheet: {len(records)}")

    themes_idx = EXPECTED_COLUMNS.index("themes")
    stance_idx = EXPECTED_COLUMNS.index("stance")
    assert stance_idx == themes_idx + 1, "themes and stance columns must be adjacent"

    new_values = []
    changed_themes = 0
    changed_stance = 0
    stance_shift = Counter()

    for record in records:
        text = f"{record.get('title', '')} {record.get('description', '')}"
        new_themes = classify_themes(text)
        new_stance = classify_stance(text)

        old_themes = str(record.get("themes", ""))
        old_stance = str(record.get("stance", ""))

        if new_themes != old_themes:
            changed_themes += 1
        if new_stance != old_stance:
            changed_stance += 1
            stance_shift[(old_stance, new_stance)] += 1

        new_values.append([new_themes, new_stance])

    print(f"Rows whose themes would change: {changed_themes}")
    print(f"Rows whose stance would change: {changed_stance}")
    for (old, new), count in stance_shift.most_common():
        print(f"  stance {old or '(blank)'} -> {new}: {count}")

    if not apply_changes:
        print("\nDry run only. Re-run with --apply to write these labels.")
        return

    if not new_values:
        print("Nothing to write.")
        return

    first_row = 2  # row 1 holds the headers
    last_row = first_row + len(new_values) - 1
    cell_range = (
        f"{col_letter(themes_idx)}{first_row}:{col_letter(stance_idx)}{last_row}"
    )
    sheet.update(values=new_values, range_name=cell_range)
    print(f"Updated {len(new_values)} rows in {cell_range}.")


if __name__ == "__main__":
    main()
