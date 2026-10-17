"""Write fairpass_v3_submission.tex: fairpass_v3.tex without LaTeX comments.

  python make_submission.py        (run in WileyDesign/fairpass_v3)

Comment-only lines are dropped. On other lines, the text after an unescaped % is removed but the % itself is
kept, so that line-end spacing (and thus the typeset output) does not change. Escaped \\% is left alone.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def first_comment(line: str) -> int:
    i = 0
    while True:
        j = line.find("%", i)
        if j < 0:
            return -1
        k, n = j - 1, 0
        while k >= 0 and line[k] == "\\":
            n += 1
            k -= 1
        if n % 2 == 0:
            return j
        i = j + 1


def strip(text: str) -> str:
    out = []
    for line in text.split("\n"):
        j = first_comment(line)
        if j < 0:
            out.append(line)
        elif line[:j].strip():
            out.append(line[:j + 1])
    return "\n".join(out)


if __name__ == "__main__":
    src = (HERE / "fairpass_v3.tex").read_text(encoding="utf-8")
    (HERE / "fairpass_v3_submission.tex").write_text(strip(src), encoding="utf-8")
    print("written fairpass_v3_submission.tex")
