"""Shared SVG drawing primitives for MiniMax architecture diagrams."""
from html import escape

COLORS = {"purple": ("#7758a6", "#f3eef9"), "orange": ("#b9741d", "#fff5e5"),
          "green": ("#4b853e", "#f0f7ec"), "teal": ("#237e76", "#eaf7f4"),
          "red": ("#ad5a53", "#fff0ee"), "blue": ("#447da5", "#eef5fb")}
STYLE = """<style>
text{font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;fill:#22364c}
.title{font-size:44px;font-weight:700}.head{font-size:31px;font-weight:700}
.node{font-size:25px;font-weight:600}.body{font-size:24px}.small{font-size:22px}
.weight{font-size:22px;fill:#59677a}.note{font-size:22px;fill:#59677a}
</style><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0 0L10 5L0 10Z" fill="#617186"/></marker></defs>"""


class Plate:
    def __init__(self, name, w, h, title, color):
        self.name, self.w, self.h = name, w, h
        self.stroke, self.fill = COLORS[color]
        self.items = []
        self.text_bounds = []
        self.rect(1, 1, w - 2, h - 2, "#fbfcfe", self.stroke, 18)
        self.text(28, 49, title, "head")
        self.items.append(f'<line x1="28" y1="75" x2="{w-28}" y2="75" stroke="{self.stroke}" opacity=".35"/>')

    def rect(self, x, y, w, h, fill, stroke="#c9d4e0", radius=10):
        assert 0 <= x and 0 <= y and x + w <= self.w and y + h <= self.h
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')

    def text(self, x, y, text, cls="body", anchor="start"):
        assert 0 <= x <= self.w and 0 <= y <= self.h
        self.items.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(text)}</text>')
        self.text_bounds.append((x, y, text, cls, anchor))

    def box(self, x, y, w, lines, h=None, fill=None):
        h = h or (35 * len(lines) + 28)
        self.rect(x, y, w, h, fill or self.fill)
        for i, line in enumerate(lines):
            if isinstance(line, tuple):
                label, cls = line
            else:
                label, cls = line, "node" if i == 0 else "body"
            self.text(x + w / 2, y + 35 + i * 35, label, cls, "middle")
            self.items[-1] = self.items[-1].replace('<text ', f'<text data-max-width="{w-28}" ')
        return y + h

    def arrow(self, points, dashed=False):
        for x, y in points:
            assert 0 <= x <= self.w and 0 <= y <= self.h
        coords = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        self.items.append(f'<polyline points="{coords}" fill="none" stroke="#617186" stroke-width="2.2"{dash} marker-end="url(#arrow)"/>')

    def down(self, x, y1, y2):
        self.arrow([(x, y1), (x, y2)])

    def add(self, x, y):
        self.items.append(f'<circle cx="{x}" cy="{y}" r="20" fill="white" stroke="#617186" stroke-width="2"/>')
        self.text(x, y + 8, "+", "node", "middle")

    def render(self):
        body = "\n".join(self.items)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img"><title>{escape(self.name)}</title>'
                f'<rect width="{self.w}" height="{self.h}" fill="white"/>{STYLE}{body}</svg>')
