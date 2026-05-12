import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return [], []
    if not HAS_NUMPY:
        avg = [sum(values[i : i + window]) / window for i in range(len(values) - window + 1)]
        offset = (window - 1) // 2
        return list(range(offset, offset + len(avg))), avg
    values = np.asarray(values, dtype=float)
    avg = np.convolve(values, np.ones(window) / window, mode="valid")
    offset = (window - 1) // 2
    x_idx = np.arange(offset, offset + len(avg))
    return x_idx, avg


def read_training_csv(filename):
    path = Path(filename)
    with path.open("r", newline="") as f:
        sample = f.readline()
        f.seek(0)
        has_header = any(ch.isalpha() for ch in sample)
        if has_header:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                parsed = {}
                for key, value in row.items():
                    if key is None or value in (None, ""):
                        continue
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        pass
                if parsed:
                    rows.append(parsed)
            return rows

        reader = csv.reader(f)
        rows = []
        for row in reader:
            if len(row) < 2:
                continue
            try:
                rows.append(
                    {
                        "timesteps": float(row[0]),
                        "episode_reward_mean": float(row[1]),
                    }
                )
            except ValueError:
                continue
        return rows


def get_series(rows, key):
    if not rows or key not in rows[0]:
        return None
    values = [row.get(key, float("nan")) for row in rows]
    if HAS_NUMPY:
        return np.array(values, dtype=float)
    return values


def plot_line(ax, x, y, label, color, smooth=True):
    ax.plot(x, y, color=color, alpha=0.35, linewidth=1.2, label=f"{label} raw")
    if smooth and len(y) >= 5:
        window = max(2, len(y) // 20)
        idx, avg = moving_average(y, window)
        if len(avg):
            smooth_x = x[idx] if HAS_NUMPY else [x[i] for i in idx]
            ax.plot(smooth_x, avg, color=color, linewidth=2.2, label=f"{label} moving avg")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def save_reward_plot(rows, output_dir, base_name, timestamp):
    x = get_series(rows, "timesteps")
    reward = get_series(rows, "episode_reward_mean")
    if x is None or reward is None:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_line(ax, x, reward, "Reward mean", "#1f77b4")
    reward_min = get_series(rows, "episode_reward_min")
    reward_max = get_series(rows, "episode_reward_max")
    if reward_min is not None and reward_max is not None:
        ax.fill_between(x, reward_min, reward_max, color="#1f77b4", alpha=0.10, label="min/max range")
    ax.set_title("Episode reward")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.legend(loc="best")
    fig.tight_layout()

    output = output_dir / f"{base_name}_reward_{timestamp}.jpg"
    fig.savefig(output, dpi=300, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return output


def svg_polyline(x, y, width, height, pad, color):
    pairs = [
        (float(xv), float(yv))
        for xv, yv in zip(x, y)
        if xv == xv and yv == yv
    ]
    if not pairs:
        return ""
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    if len(x) < 2:
        return ""
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0
    pts = []
    for xv, yv in zip(x, y):
        px = pad + ((xv - x_min) / (x_max - x_min)) * (width - 2 * pad)
        py = height - pad - ((yv - y_min) / (y_max - y_min)) * (height - 2 * pad)
        pts.append(f"{px:.1f},{py:.1f}")
    return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>'


def svg_panel(title, x, series, width=760, height=260):
    pad = 42
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    items = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="#ddd"/>',
        f'<text x="{pad}" y="24" font-family="Arial" font-size="16" font-weight="bold">{title}</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#aaa"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#aaa"/>',
    ]

    all_y = []
    for _, y, _ in series:
        if y is None:
            continue
        all_y.extend(float(v) for v in y if v == v)
    x_vals = [float(v) for v in x if v == v]
    if x_vals and all_y:
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(all_y), max(all_y)
        if x_max == x_min:
            x_max += 1.0
        if y_max == y_min:
            y_max += 1.0
        y_pad = (y_max - y_min) * 0.05
        y_min -= y_pad
        y_max += y_pad

        for i in range(5):
            gx = pad + plot_w * i / 4
            val = x_min + (x_max - x_min) * i / 4
            items.append(f'<line x1="{gx:.1f}" y1="{pad}" x2="{gx:.1f}" y2="{height-pad}" stroke="#eee"/>')
            items.append(f'<text x="{gx:.1f}" y="{height-pad+16}" text-anchor="middle" font-family="Arial" font-size="10">{val:.0f}</text>')
        for i in range(5):
            gy = pad + plot_h * i / 4
            val = y_max - (y_max - y_min) * i / 4
            items.append(f'<line x1="{pad}" y1="{gy:.1f}" x2="{width-pad}" y2="{gy:.1f}" stroke="#eee"/>')
            items.append(f'<text x="{pad-6}" y="{gy+3:.1f}" text-anchor="end" font-family="Arial" font-size="10">{val:.1f}</text>')

    legend_x = pad
    for label, y, color in series:
        if y is None:
            continue
        items.append(svg_polyline(x, y, width, height, pad, color))
        items.append(f'<rect x="{legend_x}" y="{height-22}" width="10" height="10" fill="{color}"/>')
        items.append(f'<text x="{legend_x+14}" y="{height-13}" font-family="Arial" font-size="11">{label}</text>')
        legend_x += 120
    return "\n".join(items)


def save_svg_dashboard(rows, output_dir, base_name, timestamp):
    x = get_series(rows, "timesteps")
    if x is None:
        return None

    panels = [
        (
            "Reward",
            [("mean", get_series(rows, "episode_reward_mean"), "#1f77b4")],
        ),
        (
            "Route outcomes",
            [
                ("goal", get_series(rows, "success_rate"), "#2ca02c"),
                ("oob", get_series(rows, "out_of_bound_rate"), "#d62728"),
                ("trunc", get_series(rows, "truncated_rate"), "#ff7f0e"),
            ],
        ),
        (
            "Path quality",
            [
                ("length", get_series(rows, "episode_len_mean"), "#9467bd"),
                ("revisits", get_series(rows, "revisited_edges_mean"), "#8c564b"),
            ],
        ),
        (
            "Signal - SINR",
            [("sinr", get_series(rows, "avg_sinr"), "#17becf")],
        ),
        (
            "Signal - QoS",
            [("qos", get_series(rows, "avg_qos"), "#bcbd22")],
        ),
        (
            "Curriculum",
            [
                ("coverage", get_series(rows, "coverage_scale"), "#1f77b4"),
                ("qos", get_series(rows, "qos_scale"), "#2ca02c"),
                ("mincov", get_series(rows, "min_coverage_scale"), "#d62728"),
            ],
        ),
        (
            "Reward min/max",
            [
                ("min", get_series(rows, "episode_reward_min"), "#d62728"),
                ("max", get_series(rows, "episode_reward_max"), "#2ca02c"),
            ],
        ),
    ]

    panel_w, panel_h = 760, 260
    cols = 2
    rows_n = (len(panels) + cols - 1) // cols
    width = panel_w * cols
    height = panel_h * rows_n + 60
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f7f7"/>',
        f'<text x="28" y="34" font-family="Arial" font-size="22" font-weight="bold">Training dashboard: {base_name}</text>',
    ]
    for i, (title, series) in enumerate(panels):
        col = i % cols
        row = i // cols
        x0 = col * panel_w
        y0 = 50 + row * panel_h
        body.append(f'<g transform="translate({x0},{y0})">')
        body.append(svg_panel(title, x, series, width=panel_w, height=panel_h))
        body.append("</g>")
    body.append("</svg>")

    output = output_dir / f"{base_name}_dashboard_{timestamp}.svg"
    output.write_text("\n".join(body), encoding="utf-8")
    return output


def save_svg_reward_plot(rows, output_dir, base_name, timestamp):
    x = get_series(rows, "timesteps")
    reward = get_series(rows, "episode_reward_mean")
    if x is None or reward is None:
        return None

    width, height = 960, 420
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f7f7"/>',
        f'<g transform="translate(80,50)">',
        svg_panel("Episode reward mean", x, [("reward", reward, "#1f77b4")], width=820, height=320),
        "</g>",
        "</svg>",
    ]
    output = output_dir / f"{base_name}_reward_{timestamp}.svg"
    output.write_text("\n".join(body), encoding="utf-8")
    return output


def save_classic_matplotlib_plot(rows, output_dir, base_name, timestamp, title, use_index_steps):
    x = list(range(len(rows))) if use_index_steps else get_series(rows, "timesteps")
    reward = get_series(rows, "episode_reward_mean")
    if x is None or reward is None:
        return None

    x = list(x)
    reward = list(reward)
    window = max(2, len(reward) // 25)
    idx, avg = moving_average(reward, window)
    if len(avg):
        plot_x = [x[i] for i in idx]
        plot_y = avg
    else:
        plot_x = x
        plot_y = reward

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(plot_x, plot_y, color="red", linewidth=1.5, label="Moving Average")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(True)
    ax.legend(loc="upper left")
    fig.tight_layout()

    output = output_dir / f"{base_name}_classic_moving_average_{timestamp}.jpg"
    fig.savefig(output, dpi=300, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return output


def save_classic_svg_plot(rows, output_dir, base_name, timestamp, title, use_index_steps):
    x = list(range(len(rows))) if use_index_steps else get_series(rows, "timesteps")
    reward = get_series(rows, "episode_reward_mean")
    if x is None or reward is None:
        return None

    x = list(x)
    reward = list(reward)
    window = max(2, len(reward) // 25)
    idx, avg = moving_average(reward, window)
    if len(avg):
        x = [x[i] for i in idx]
        reward = avg

    width, height = 760, 560
    pad_left, pad_top, pad_right, pad_bottom = 82, 58, 54, 72
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    pairs = [(float(a), float(b)) for a, b in zip(x, reward) if a == a and b == b]
    if len(pairs) < 2:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_pad = max(50.0, (y_max - y_min) * 0.08)
    y_min -= y_pad
    y_max += y_pad
    if x_max == x_min:
        x_max += 1.0
    if y_max == y_min:
        y_max += 1.0

    def sx(v):
        return pad_left + ((v - x_min) / (x_max - x_min)) * plot_w

    def sy(v):
        return pad_top + plot_h - ((v - y_min) / (y_max - y_min)) * plot_h

    points = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in pairs)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" stroke="#222" stroke-width="2"/>',
        f'<text x="{width/2:.1f}" y="36" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',
        f'<rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#111"/>',
    ]

    for i in range(6):
        gx = pad_left + plot_w * i / 5
        val = x_min + (x_max - x_min) * i / 5
        parts.append(f'<line x1="{gx:.1f}" y1="{pad_top}" x2="{gx:.1f}" y2="{pad_top+plot_h}" stroke="#b0b0b0"/>')
        parts.append(f'<text x="{gx:.1f}" y="{pad_top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="13">{val:.0f}</text>')
    for i in range(7):
        gy = pad_top + plot_h * i / 6
        val = y_max - (y_max - y_min) * i / 6
        parts.append(f'<line x1="{pad_left}" y1="{gy:.1f}" x2="{pad_left+plot_w}" y2="{gy:.1f}" stroke="#b0b0b0"/>')
        parts.append(f'<text x="{pad_left-10}" y="{gy+4:.1f}" text-anchor="end" font-family="Arial" font-size="13">{val:.0f}</text>')

    legend_x, legend_y = pad_left + 18, pad_top + 18
    parts.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="160" height="34" fill="white" stroke="#d0d0d0"/>',
            f'<line x1="{legend_x+16}" y1="{legend_y+17}" x2="{legend_x+56}" y2="{legend_y+17}" stroke="red" stroke-width="2"/>',
            f'<text x="{legend_x+66}" y="{legend_y+22}" font-family="Arial" font-size="13">Moving Average</text>',
            f'<polyline points="{points}" fill="none" stroke="red" stroke-width="2"/>',
            f'<text x="{width/2:.1f}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="15">Step</text>',
            f'<text x="24" y="{height/2:.1f}" text-anchor="middle" transform="rotate(-90 24 {height/2:.1f})" font-family="Arial" font-size="15">Reward</text>',
            "</svg>",
        ]
    )

    output = output_dir / f"{base_name}_classic_moving_average_{timestamp}.svg"
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


def save_full_dashboard(rows, output_dir, base_name, timestamp):
    x = get_series(rows, "timesteps")
    if x is None:
        return None

    fig, axes = plt.subplots(4, 2, figsize=(16, 18), sharex=True)
    axes = axes.ravel()

    reward = get_series(rows, "episode_reward_mean")
    if reward is not None:
        plot_line(axes[0], x, reward, "Reward mean", "#1f77b4")
        axes[0].set_title("Reward convergence")
        axes[0].set_ylabel("Reward")

    for key, label, color in [
        ("success_rate", "Goal", "#2ca02c"),
        ("out_of_bound_rate", "OOB", "#d62728"),
        ("truncated_rate", "Truncated", "#ff7f0e"),
    ]:
        y = get_series(rows, key)
        if y is not None:
            plot_line(axes[1], x, y, label, color)
    axes[1].set_title("Route outcomes")
    axes[1].set_ylabel("Percent")
    axes[1].set_ylim(bottom=0)

    for key, label, color in [
        ("episode_len_mean", "Episode length", "#9467bd"),
        ("revisited_edges_mean", "Revisited edges", "#8c564b"),
    ]:
        y = get_series(rows, key)
        if y is not None:
            plot_line(axes[2], x, y, label, color)
    axes[2].set_title("Path quality")
    axes[2].set_ylabel("Count")
    axes[2].set_ylim(bottom=0)

    y = get_series(rows, "avg_sinr")
    if y is not None:
        plot_line(axes[3], x, y, "Avg SINR", "#17becf")
    axes[3].set_title("Signal - SINR")
    axes[3].set_ylabel("SINR")

    y = get_series(rows, "avg_qos")
    if y is not None:
        plot_line(axes[4], x, y, "Avg QoS", "#bcbd22")
    axes[4].set_title("Signal - QoS")
    axes[4].set_ylabel("QoS")

    for key, label, color in [
        ("coverage_scale", "Coverage scale", "#1f77b4"),
        ("qos_scale", "QoS scale", "#2ca02c"),
        ("min_coverage_scale", "Min coverage scale", "#d62728"),
    ]:
        y = get_series(rows, key)
        if y is not None:
            axes[5].plot(x, y, color=color, linewidth=2, label=label)
    axes[5].set_title("Curriculum schedule")
    axes[5].set_ylabel("Scale")
    axes[5].set_ylim(-0.05, 1.05)
    axes[5].grid(True, alpha=0.25)
    axes[5].legend(loc="best")

    reward_min = get_series(rows, "episode_reward_min")
    reward_max = get_series(rows, "episode_reward_max")
    if reward_min is not None and reward_max is not None:
        axes[6].plot(x, reward_min, color="#d62728", alpha=0.8, label="Reward min")
        axes[6].plot(x, reward_max, color="#2ca02c", alpha=0.8, label="Reward max")
        axes[6].set_title("Reward min/max")
        axes[6].set_ylabel("Reward")
        axes[6].grid(True, alpha=0.25)
        axes[6].legend(loc="best")

    axes[7].axis("off")

    for ax in axes:
        ax.set_xlabel("Timesteps")

    fig.suptitle(f"Training dashboard: {base_name}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output = output_dir / f"{base_name}_dashboard_{timestamp}.jpg"
    fig.savefig(output, dpi=300, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return output


def plot_csv(filename, output_dir=None, show=False):
    rows = read_training_csv(filename)
    if not rows:
        print("Errore: nessun dato valido nel file CSV.")
        return []

    csv_path = Path(filename)
    base_name = csv_path.stem
    if output_dir is None:
        output_dir = csv_path.parent / "plots"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = []
    classic_title = "Mean Values of Step vs Reward - Balanced Weights"
    use_index_steps = False

    if not HAS_MATPLOTLIB:
        reward_plot = save_svg_reward_plot(rows, output_dir, base_name, timestamp)
        if reward_plot:
            outputs.append(reward_plot)
        classic_plot = save_classic_svg_plot(rows, output_dir, base_name, timestamp, classic_title, use_index_steps)
        if classic_plot:
            outputs.append(classic_plot)
        dashboard = save_svg_dashboard(rows, output_dir, base_name, timestamp)
        if dashboard:
            outputs.append(dashboard)
        print("matplotlib non installato: impossibile generare JPG in questa shell; generato fallback SVG.")
        for output in outputs:
            print(f"Grafico salvato: {output}")
        return outputs

    reward_plot = save_reward_plot(rows, output_dir, base_name, timestamp)
    if reward_plot:
        outputs.append(reward_plot)

    classic_plot = save_classic_matplotlib_plot(rows, output_dir, base_name, timestamp, classic_title, use_index_steps)
    if classic_plot:
        outputs.append(classic_plot)

    if "success_rate" in rows[0]:
        dashboard = save_full_dashboard(rows, output_dir, base_name, timestamp)
        if dashboard:
            outputs.append(dashboard)

    for output in outputs:
        print(f"Grafico salvato: {output}")

    if show:
        plt.show()
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Plot training reward and metrics CSV files.")
    parser.add_argument("csv_file", nargs="?", default="reward_distance.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    plot_csv(args.csv_file, output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
