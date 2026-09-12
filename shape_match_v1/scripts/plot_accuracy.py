#!/usr/bin/env python3
"""Plot measured error only; no smoothing or fabricated samples."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root = Path(__file__).resolve().parent.parent
    with (root/'results/accuracy.csv').open() as stream:
        records = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for kind, color, label in [('translation', '#147bb8', 'Translation phase'),
                                ('pose', '#218858', 'Translation + rotation + scale')]:
        rows = [r for r in records if r['kind'] == kind and int(r['found'])]
        ax.plot([int(r['case']) for r in rows], [float(r['position_error']) for r in rows],
                'o-', color=color, label=label, markersize=4, linewidth=1)
    ax.axhline(1/30, color='#b33b3b', linestyle='--', label='1/30 pixel reference')
    ax.set(xlabel='Independent synthetic case', ylabel='2D position error (pixel)',
           title='Measured subpixel error: 30 noise-free synthetic cases', ylim=(0, .04))
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(root/'results/accuracy_plot.svg')
    plt.close(fig)

if __name__ == '__main__':
    main()
