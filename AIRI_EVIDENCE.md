# Airi-PC Evidence API

The browser text action returns the current page URL, exact title, body text, all paragraph texts, and the first `<p>` as `first_paragraph`.

`act-verify` returns real PNG evidence for both `before_screenshot` and `after_screenshot`, including dimensions and Base64 bytes, together with `changed_ratio` and `changed_bbox`.

These fields are intended for benchmark evidence and do not change normal computer-control semantics.
