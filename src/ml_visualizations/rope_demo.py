"""Связанные представления RoPE; общая логика для notebook и проверок."""

from io import BytesIO

import ipywidgets as widgets
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc

from .rope import frequencies, rotate, scores, synthetic_vectors


def make_figure(q, k, *, m, n, c, base, pair, delta_range):
    """Геометрия одной пары, частоты/углы всех пар, полный score."""
    theta = frequencies(len(q), base)
    qp, kp = q.reshape(-1, 2), k.reshape(-1, 2)
    qr = rotate(q, m + c, base).reshape(-1, 2)
    kr = rotate(k, n + c, base).reshape(-1, 2)
    fig = Figure(figsize=(11, 9), layout="constrained", facecolor="white")
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.25])
    ax = fig.add_subplot(grid[:2, 0])
    vectors = [(qp[pair], "#0072B2", "q исходный", "--"),
               (kp[pair], "#D55E00", "k исходный", "--"),
               (qr[pair], "#0072B2", "q после RoPE", "-"),
               (kr[pair], "#D55E00", "k после RoPE", "-")]
    radius = max(np.linalg.norm(v) for v, *_ in vectors)
    limit = max(1.0, radius * 1.35)
    for v, color, label, style in vectors:
        ax.plot([0, v[0]], [0, v[1]], color=color, linestyle=style,
                marker="o" if style == "-" else "s", markevery=[1],
                markerfacecolor=color if style == "-" else "white", label=label)
        ax.annotate("", xy=v, xytext=0.87 * v,
                    arrowprops={"arrowstyle": "->", "color": color})
    # Дуга сохраняет знак поворота; полные обороты опускаются.
    for j, (v, p, color) in enumerate([(qp[pair], m + c, "#0072B2"),
                                      (kp[pair], n + c, "#D55E00")]):
        start = np.degrees(np.arctan2(v[1], v[0]))
        angle = p * theta[pair]
        sweep = np.degrees(np.sign(angle) * (abs(angle) % (2 * np.pi)))
        diameter = limit * (0.5 + j * 0.22)
        ax.add_patch(Arc((0, 0), diameter, diameter,
                        theta1=min(start, start + sweep), theta2=max(start, start + sweep),
                        color=color, alpha=0.65))
        if abs(sweep) > 1e-8:
            end = np.radians(start + sweep)
            before = end - np.sign(sweep) * min(abs(np.radians(sweep)) / 3, 0.12)
            ax.annotate("", xy=diameter/2 * np.array([np.cos(end), np.sin(end)]),
                        xytext=diameter/2 * np.array([np.cos(before), np.sin(before)]),
                        arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.65})
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
           xlabel=f"Координата {2 * pair}", ylabel=f"Координата {2 * pair + 1}",
           title=f"Одна пара i={pair} • позиции m+c, n+c")
    ax.axhline(0, color="0.8", linewidth=0.7)
    ax.axvline(0, color="0.8", linewidth=0.7)
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", fontsize=8)
    indices = np.arange(len(theta))
    colors = ["#009E73" if i == pair else "#bcc5cb" for i in indices]
    freq = fig.add_subplot(grid[0, 1])
    freq.bar(indices, theta, color=colors)
    freq.set(yscale="log", xticks=indices, xlabel="Пара i (выбрана зелёная)",
             ylabel="θᵢ, рад / позицию", title="Частоты всех пар")
    freq.grid(axis="y", alpha=0.2)
    angles = fig.add_subplot(grid[1, 1])
    angles.plot(indices, (m + c) * theta, "o-", color="#0072B2", label="(m+c)θᵢ")
    angles.plot(indices, (n + c) * theta, "s--", color="#D55E00", label="(n+c)θᵢ")
    angles.axvspan(pair - 0.15, pair + 0.15, color="#009E73", alpha=0.15)
    angles.set(xticks=indices, xlabel="Пара i", ylabel="Угол, рад (без modulo 2π)",
               title="Углы всех пар")
    angles.legend(fontsize=8)
    angles.grid(alpha=0.2)
    rel = fig.add_subplot(grid[2, :])
    deltas = np.arange(delta_range[0], delta_range[1] + 1)
    values = np.array([q @ rotate(k, int(delta), base) for delta in deltas])
    rel.plot(deltas, values, color="#333b50", label="qᵀ R_Δ k — сумма по всем d координатам")
    delta = n - m
    current = q @ rotate(k, delta, base)
    if deltas[0] <= delta <= deltas[-1]:
        rel.scatter([delta], [current], color="#009E73", zorder=3, label=f"Δ=n−m={delta}")
        rel.axvline(delta, color="#009E73", alpha=0.3)
    else:
        rel.text(0.02, 0.96, f"Текущий Δ={delta} вне выбранного диапазона",
                 transform=rel.transAxes, va="top")
    rel.set(xlim=delta_range, xlabel="Относительный сдвиг Δ = n − m",
            ylabel="Полный dot product", title="Фиксированные q и k • score не обязан убывать с |Δ|")
    rel.grid(alpha=0.2)
    rel.legend(fontsize=8, loc="lower right")
    return fig


def _validate_config(config):
    required = {"d", "base", "seed", "m", "n", "c", "pair", "delta_range"}
    if set(config) != required:
        raise ValueError(f"CONFIG должен содержать ровно ключи: {sorted(required)}")
    frequencies(config["d"], config["base"])
    synthetic_vectors(config["d"], config["seed"])
    if config["d"] > 32 or not 10 <= config["base"] <= 1e6:
        raise ValueError("В демонстрации d: 2…32 (чётная), base: 10…1000000")
    for name, low, high in [("m", -128, 128), ("n", -128, 128), ("c", -256, 256),
                            ("pair", 0, config["d"] // 2 - 1)]:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} должен быть целым числом в диапазоне [{low}, {high}]")
    span = config["delta_range"]
    if (not isinstance(span, (tuple, list)) or len(span) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) for x in span)
            or not -256 <= span[0] < span[1] <= 256):
        raise ValueError("delta_range: два целых числа −256 ≤ начало < конец ≤ 256")


class RopeDemo:
    """Контролы обновляют один PNG и HTML без отдельного JS/plot backend."""

    def __init__(self, config):
        _validate_config(config)
        self.initial = dict(config, delta_range=tuple(config["delta_range"]))
        self._busy = False
        style = {"description_width": "110px"}
        layout = widgets.Layout(width="330px")

        def slider(name, low, high):
            return widgets.IntSlider(value=config[name], min=low, max=high,
                                     description=name, continuous_update=False,
                                     style=style, layout=layout)

        self.controls = {
            "d": widgets.Dropdown(options=list(range(2, 33, 2)), value=config["d"],
                                  description="d (координат)", style=style, layout=layout),
            "base": widgets.FloatLogSlider(value=config["base"], base=10, min=1, max=6,
                                            step=0.1, description="base", readout_format=".0f",
                                            continuous_update=False, style=style, layout=layout),
            "seed": widgets.BoundedIntText(value=config["seed"], min=0, max=2**32 - 1,
                                           description="seed", style=style, layout=layout),
            "m": slider("m", -128, 128), "n": slider("n", -128, 128),
            "c": slider("c", -256, 256), "pair": slider("pair", 0, config["d"] // 2 - 1),
            "delta_range": widgets.IntRangeSlider(value=config["delta_range"], min=-256,
                                                   max=256, description="Диапазон Δ",
                                                   continuous_update=False, style=style,
                                                   layout=layout),
        }
        self.reset_button = widgets.Button(description="Сброс к CONFIG", icon="undo")
        self.reset_button.on_click(self.reset)
        self.summary = widgets.HTML()
        self.details = widgets.HTML()
        self.image = widgets.Image(format="png", layout=widgets.Layout(width="100%", max_width="1100px"))
        self.ui = widgets.VBox([
            widgets.HTML("<b>RoPE · одна голова внимания</b><br>"
                         "Позиции — целые индексы; углы — радианы. Отпустите ползунок для обновления.<br>"
                         "Изменение d или seed создаёт воспроизводимые q/k; позиции, base и пара их сохраняют."),
            widgets.Box(list(self.controls.values()), layout=widgets.Layout(display="flex", flex_flow="row wrap")),
            self.reset_button, self.summary, self.image, self.details,
        ])
        self._vector_key = None
        for control in self.controls.values():
            control.observe(self._changed, names="value")
        self.refresh()

    @property
    def state(self):
        return {name: control.value for name, control in self.controls.items()}

    def _changed(self, change):
        if self._busy:
            return
        # При уменьшении d сначала сужаем i, не вызывая промежуточных рендеров.
        self._busy = True
        try:
            self.controls["pair"].max = self.controls["d"].value // 2 - 1
        finally:
            self._busy = False
        self.refresh()

    def reset(self, _button=None):
        self._busy = True
        try:
            self.controls["d"].value = self.initial["d"]
            self.controls["pair"].max = self.initial["d"] // 2 - 1
            for name, value in self.initial.items():
                self.controls[name].value = value
        finally:
            self._busy = False
        self.refresh()

    def refresh(self):
        s = self.state
        # RangeSlider допускает равные концы: показываем явную ошибку без traceback/fallback.
        if s["delta_range"][0] >= s["delta_range"][1]:
            self.summary.value = "<b>Ошибка: начало диапазона Δ должно быть меньше конца.</b>"
            self.image.value = b""
            self.details.value = ""
            return
        key = (s["d"], s["seed"])
        if key != self._vector_key:
            self.q, self.k = synthetic_vectors(*key)
            self._vector_key = key
        self.result = scores(self.q, self.k, s["m"], s["n"], s["c"], s["base"])
        r = self.result
        self.summary.value = (
            f"<b>Все {s['d']} координат:</b> Δ=n−m={s['n'] - s['m']}; "
            f"(m,n)=({s['m']},{s['n']}) → ({s['m']+s['c']},{s['n']+s['c']})<br>"
            f"Прямое qₘᵀkₙ = <b>{r['direct']:.9f}</b> · "
            f"qᵀR_Δk = <b>{r['relative']:.9f}</b> · "
            f"Общий сдвиг = <b>{r['shifted']:.9f}</b><br>"
            f"|прямое − относительное| = {abs(r['direct']-r['relative']):.2e}; "
            f"|прямое − сдвинутое| = {abs(r['direct']-r['shifted']):.2e}. "
            "Это скалярное произведение, не вероятность внимания."
        )
        theta = frequencies(s["d"], s["base"])
        i = s["pair"]
        qr = rotate(self.q, s["m"] + s["c"], s["base"]).reshape(-1, 2)
        kr = rotate(self.k, s["n"] + s["c"], s["base"]).reshape(-1, 2)
        qp, kp = self.q.reshape(-1, 2), self.k.reshape(-1, 2)
        rows = "".join(
            f"<tr style='{'background:#def2e9' if j == i else ''}'><td>{j}</td>"
            f"<td>({2*j}, {2*j+1})</td><td>{t:.6g}</td>"
            f"<td>{(s['m']+s['c'])*t:.6f}</td><td>{(s['n']+s['c'])*t:.6f}</td>"
            f"<td>{qr[j] @ kr[j]:.6f}</td></tr>" for j, t in enumerate(theta)
        )
        self.details.value = (
            f"<b>Только пара i={i}:</b> q={np.round(qp[i], 4)}, k={np.round(kp[i], 4)}<br>"
            f"После вращения: q={np.round(qr[i], 4)}, k={np.round(kr[i], 4)}; "
            f"вклад пары = <b>{qr[i] @ kr[i]:.9f}</b>.<br>"
            "Дуги сохраняют знак поворота: плюс — против часовой стрелки, минус — по часовой. "
            "Полные обороты опущены; полные знаковые углы — ниже.<br>"
            "<table style='border-collapse:collapse;text-align:right' cellpadding='5'>"
            "<tr><th>i</th><th>Координаты</th><th>θᵢ</th><th>(m+c)θᵢ, рад</th>"
            "<th>(n+c)θᵢ, рад</th><th>Вклад пары</th></tr>" + rows + "</table>"
        )
        fig = make_figure(self.q, self.k, m=s["m"], n=s["n"], c=s["c"], base=s["base"],
                          pair=i, delta_range=s["delta_range"])
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=110)
        self.image.value = buffer.getvalue()
        fig.clear()
