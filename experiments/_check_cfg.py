# 一次性校验脚本：静态检查 ECA 模板 yaml 的层索引/通道衔接（不加载 torch）
import sys
import yaml

ok = True

def check(path):
    global ok
    print(f"\n=== {path} ===")
    d = yaml.safe_load(open(path, encoding="utf-8"))
    W, DP, MX = d["width_multiple"], d["depth_multiple"], d["max_channels"]
    layers = d["backbone"] + d["head"]
    ch = []
    n_prev_scale_layer = None
    for i, (f, rep, m, a) in enumerate(layers):
        # from 索引合法性
        fl = f if isinstance(f, list) else [f]
        for x in fl:
            if x != -1 and not (0 <= x < i):
                print(f"  ✗ 层{i} {m}: 非法 from 索引 {x}"); ok = False
        if m == "Conv":
            c2 = min(round(a[0] * W / 8) * 8, MX); ch.append(c2)
        elif m == "C2f":
            c2 = min(round(a[0] * W / 8) * 8, MX); ch.append(c2)
        elif m == "SPPF":
            c2 = min(round(a[0] * W / 8) * 8, MX); ch.append(c2)
        elif m == "ECABlock":
            ch.append(ch[-1])  # 通道不变
        elif m == "nn.Upsample":
            ch.append(ch[f] if f != -1 else ch[-1])
        elif m == "Concat":
            srcs = [ch[-1] if x == -1 else ch[x] for x in f]
            ch.append(sum(srcs))
        elif m == "Detect":
            print(f"  Detect from {f} → 通道 {[ch[x] for x in f]}")
            if len(f) != 3:
                print("  ✗ Detect 应接 3 个尺度"); ok = False
            else:
                p3, p4, p5 = ch[f[0]], ch[f[1]], ch[f[2]]
                if not (p3 < p4 < p5):
                    print(f"  ✗ 尺度应递增 P3<P4<P5，实际 {p3},{p4},{p5}"); ok = False
                else:
                    print(f"  ✓ 通道递增 {p3} → {p4} → {p5}")
            continue
        else:
            print(f"  ? 未知模块 {m}（跳过通道追踪）"); ch.append(ch[-1])
    print(f"  共 {len(layers)} 层")

for p in ["cfgs/v8n_eca_backbone.yaml", "cfgs/v8n_eca_sppf.yaml", "cfgs/v8n_eca_neck.yaml"]:
    check(p)
print("\n" + ("全部通过 ✓" if ok else "存在问题 ✗"))
sys.exit(0 if ok else 1)
