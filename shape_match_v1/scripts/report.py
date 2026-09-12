#!/usr/bin/env python3
"""Validate independent outputs and produce measured reports. Does not run matching.

Dependencies: Python stdlib + Pillow for embedding the original image in SVG.
The old reference label error is corrected ONLY in a derived comparison table.
"""
import argparse
import base64
import csv
import html
import io
import json
import math
import platform
import statistics
from pathlib import Path
from PIL import Image

def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))

def summarize(values):
    if not values:
        return None
    data = sorted(values)
    return {'count': len(data), 'rmse': math.sqrt(statistics.mean(x*x for x in data)),
            'mean': statistics.mean(data), 'p95': data[math.ceil(.95*len(data))-1],
            'max': max(data)}

def report(root, results):
    predictions = rows(results/'matching_results.csv')
    references = rows(root/'data/reference/matching_results.csv')
    # User-confirmed historical model-label mapping correction, never used by detector.
    ring_indices = {1, 2, 5}
    for ref in references:
        ref['model'] = 'ring' if int(ref['index']) in ring_indices else 'nut'
    unused = set(range(len(predictions)))
    comparisons = []
    for ref in references:
        candidates = [i for i in unused if predictions[i]['model'] == ref['model']]
        if not candidates:
            comparisons.append({'reference_index': ref['index'], 'model': ref['model'], 'matched': False})
            continue
        def distance(i):
            return math.hypot(float(predictions[i]['row'])-float(ref['row']),
                              float(predictions[i]['column'])-float(ref['column']))
        i = min(candidates, key=distance)
        if distance(i) > 3:
            comparisons.append({'reference_index': ref['index'], 'model': ref['model'], 'matched': False})
            continue
        unused.remove(i)
        pred = predictions[i]
        # Ring has 8-fold nominal symmetry; regular hex nuts have 6-fold symmetry.
        period = 45 if ref['model'] == 'ring' else 60
        angular = (float(pred['angle_deg'])-float(ref['angle_deg'])+period/2)%period-period/2
        comparisons.append({'reference_index': ref['index'], 'prediction_index': pred['index'],
                            'model': ref['model'], 'matched': True, 'position_difference_px': distance(i),
                            'angle_difference_mod_symmetry_deg': angular,
                            'scale_difference': float(pred['scale_row'])-float(ref['scale_row']),
                            'score_difference': float(pred['score'])-float(ref['score'])})
    accuracy = rows(results/'accuracy.csv')
    errors = [float(x['position_error']) for x in accuracy if int(x['found'])]
    by_kind = {kind: summarize([float(x['position_error']) for x in accuracy
                               if x['kind'] == kind and int(x['found'])])
               for kind in sorted({x['kind'] for x in accuracy})}
    timing = json.loads((results/'timing.json').read_text())
    summary = {'platform': platform.platform(), 'sample': {'expected': 7, 'detected': len(predictions),
               'associated': sum(x['matched'] for x in comparisons), 'unassociated': len(unused),
               'comparisons': comparisons}, 'synthetic': {'total': len(accuracy), 'found': len(errors),
               'position': summarize(errors), 'groups': by_kind, 'target_rmse_px': 1/30,
               'passes_tested_rmse': bool(errors) and summarize(errors)['rmse'] <= 1/30},
               'timing': timing, 'halcon_runtime_comparison': 'not executed; uploaded CSV only'}
    if (results/'benchmark.json').exists():
        summary['repeated_benchmark'] = json.loads((results/'benchmark.json').read_text())
    (results/'validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    # Geometric visualization: no inferred contours; only actual matcher output is drawn.
    image = Image.open(root/'data/reference/original.png').convert('L')
    buffer = io.BytesIO(); image.save(buffer, format='PNG')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{image.width}" height="{image.height}" viewBox="0 0 {image.width} {image.height}">',
           '<image width="100%" height="100%" href="data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()+'"/>']
    for instance in json.loads((results/'contours.json').read_text()):
        color = '#ffcc00' if instance['model'] == 'ring' else '#00aa55'
        for contour in instance['contours']:
            points = ' '.join(f'{column:.4f},{row:.4f}' for row, column in contour)
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.2"/>')
    for pred in predictions:
        svg.append(f'<text x="{float(pred["column"])+6}" y="{float(pred["row"])-8}" font-size="12" fill="#e22">{html.escape(pred["model"])}</text>')
    svg.append('</svg>');(results/'recognition.svg').write_text('\n'.join(svg))
    pos=summary['synthetic']['position']
    lines=['# 实测验证报告','', '本报告由 scripts/report.py 根据可执行程序输出自动生成。', '',
           '## 样图检出', '', f'- 目标数：7；检出：{len(predictions)}；按类别与位置关联成功：{summary["sample"]["associated"]}。',
           '- 使用全图搜索，未传入目标位置或上下条带 ROI。',
           '- 原始 HALCON CSV 保留不变；历史标签错误只在比较时按已确认映射修正。',
           '- 对称角度按等价旋转比较；HALCON 输出不是绝对真值。', '',
           '| 类别 | 参考序号 | 位置差/px | 对称角度差/deg | 尺度差 |',
           '|---|---:|---:|---:|---:|']
    for x in comparisons:
        if x['matched']:
            lines.append(f'| {x["model"]} | {x["reference_index"]} | {x["position_difference_px"]:.6f} | {x["angle_difference_mod_symmetry_deg"]:.6f} | {x["scale_difference"]:.6f} |')
    lines += ['', '## 独立合成精度', '',
              f'- 检出 {len(errors)}/{len(accuracy)}；二维位置 RMSE **{pos["rmse"]:.6f} px**，P95 {pos["p95"]:.6f} px，最大 {pos["max"]:.6f} px。',
              '- 20 个纯平移相位案例，10 个平移/旋转/等比尺度组合案例。',
              '- 模板与搜索图分别由连续几何生成，8×8 像素积分近似；未使用整数图像 warp 生成真值。',
              '- 该小型无噪声、单形状合成集通过 1/30 px RMSE 门槛，不代表真实相机或所有目标达到该精度。',
              '- 尚未进行可溯源实拍位移验证，也未安装 HALCON 运行时执行逐参数差分测试。', '',
              '## 性能', '', f'- 本次完整样图搜索：**{timing["total_ms"]:.3f} ms**；10 ms 目标尚未达到。',
              f'- 金字塔 {timing["pyramid_ms"]:.3f} ms；粗搜 {timing["top_level_ms"]:.3f} ms；跟踪 {timing["tracking_ms"]:.3f} ms；精定位 {timing["refinement_ms"]:.3f} ms。',
              '- CPU 单线程核心；图像 640×480；2 个模型；建模和文件 I/O 不计入上述搜索时间。', '',
              '## 契约与兼容边界', '',
              '- 公开算子均有工作实现，无空函数或固定样例结果返回。',
              '- 接口对齐范围与已知行为差异见 docs/API.md；这不是完整 HALCON SDK/ABI 替代。',
              '- 测试日志见 results/test.log；ASan/UBSan 检查日志见 results/sanitizer.log。',
              '- LeakSanitizer 因执行环境使用 ptrace 无法运行；随后关闭泄漏检查执行 ASan/UBSan，未声称完成泄漏检测。', '']
    benchmark=summary.get('repeated_benchmark')
    if benchmark:
        lines += ['## 同进程重复性能基线', '',
                  f'- 预热 {benchmark["warmup"]} 次，测量 {benchmark["iterations"]} 次；单线程。',
                  f'- 中位数 **{benchmark["median_ms"]:.3f} ms**；最小 {benchmark["min_ms"]:.3f} ms；最大 {benchmark["max_ms"]:.3f} ms。',
                  '- 每次均验证返回 7 个实例；不包含建模及文件 I/O。',
                  '- 处理器及编译参数见 environment.json。', '']
    (results/'VALIDATION.md').write_text('\n'.join(lines))
    if len(predictions)!=7 or summary['sample']['associated']!=7 or unused:
        raise SystemExit('Sample detection validation failed')
    if len(errors)!=len(accuracy):
        raise SystemExit('Synthetic recall validation failed')
    if not summary['synthetic']['passes_tested_rmse']:
        raise SystemExit('Tested synthetic position RMSE exceeds 1/30 pixel')
    print(json.dumps({k:v for k,v in summary.items() if k!='sample'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--results', type=Path, default=Path('results'))
    args=parser.parse_args();report(args.root,args.results)
