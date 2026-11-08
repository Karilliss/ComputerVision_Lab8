import argparse
import csv
import os
import sys
import numpy as np
import cv2
import matplotlib
if "--no_show" in sys.argv:      
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = []                           

FOCAL_FULL = 3740.0    
STEP_MM = 40.0       
BASE_FULL_MM = 160.0   
CENTER_VIEW = 3        

PLY_HEADER = ("ply\nformat %s 1.0\nelement vertex %d\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")


class Dataset:
    def __init__(self, root, illum, exp, scale, gt_scale=1.0):
        self.root, self.illum, self.exp, self.scale = root, illum, exp, scale
        self.f = FOCAL_FULL * scale
        p = os.path.join(root, "dmin.txt")
        self.dmin = float(open(p).read().split()[0]) * scale if os.path.exists(p) else 0.0
        self._views = {}
        p = os.path.join(root, "disp1.png")
        self.d1 = None
        if os.path.exists(p):
            g = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) * gt_scale
            if scale != 1.0:
                g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) * scale
            self.d1 = g

    def has(self, k):
        return k in self._views or os.path.exists(
            os.path.join(self.root, f"Illum{self.illum}", f"Exp{self.exp}", f"view{k}.png"))

    def view(self, k):
        if k not in self._views:
            p = os.path.join(self.root, f"Illum{self.illum}", f"Exp{self.exp}", f"view{k}.png")
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(p)
            if self.scale != 1.0:
                img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            self._views[k] = img
        return self._views[k]

    def gt_pair(self, a, b):
        if self.d1 is None:
            return None
        if (a, b) == (1, 5):
            return self.d1.copy()
        h, w = self.d1.shape
        ys, xs = np.nonzero(self.d1 > 0)
        d = self.d1[ys, xs]
        xa = np.round(xs - d * (a - 1) / 4.0).astype(int)
        ok = (xa >= 0) & (xa < w)
        ys, xa, d = ys[ok], xa[ok], d[ok]
        order = np.argsort(d)                     
        out = np.zeros((h, w), np.float32)
        out[ys[order], xa[order]] = d[order] * (b - a) / 4.0
        l, r = np.roll(out, 1, 1), np.roll(out, -1, 1)
        hole = (out == 0) & (l > 0) & (r > 0) & (np.abs(l - r) < 1.5)
        out[hole] = 0.5 * (l[hole] + r[hole])
        return out

    def z_median(self):
        """Типова відстань до об'єкта, мм (за еталоном пари 1-5)."""
        g = self.d1[self.d1 > 0]
        return float(np.median(self.f * BASE_FULL_MM / (g + self.dmin)))


def dmin_for(ds, a, b):
    return ds.dmin * (b - a) / 4.0


def num_disp_from_gt(gt, default=112):
    if gt is None or not (gt > 0).any():
        return default
    return int(np.ceil((np.percentile(gt[gt > 0], 99.5) + 16) / 16.0) * 16)


def compute_disparity(L, R, num_disp, block=5, mode="sgbm"):
    """Диспаритет, px (float32). Невалідні пікселі = NaN."""
    num_disp = int(np.ceil(num_disp / 16.0) * 16)
    block = block | 1
    if mode == "bm":
        m = cv2.StereoBM_create(numDisparities=num_disp, blockSize=max(block, 5))
        d = m.compute(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY),
                      cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
    else:
        m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=block,
                                  P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2,
                                  disp12MaxDiff=1, uniquenessRatio=10,
                                  speckleWindowSize=100, speckleRange=32)
        d = m.compute(L, R).astype(np.float32) / 16.0     
    d[d <= 0] = np.nan
    return d


def disparity_to_cloud(disp, imgL, f, B, dmin_eff):
    """Матриця Q + reprojectImageTo3D (як у відео, але з реальними f, B, dmin)."""
    h, w = disp.shape
    cx, cy = 0.5 * w, 0.5 * h
    Q = np.float32([[1, 0, 0, -cx],
                    [0, -1, 0, cy],
                    [0, 0, 0, -f],
                    [0, 0, 1.0 / B, dmin_eff / B]])
    d = np.nan_to_num(disp, nan=0.0)
    pts = cv2.reprojectImageTo3D(d, Q)
    mask = np.isfinite(disp) & np.isfinite(pts).all(axis=2) & (d + dmin_eff > 0.5)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    return pts[mask], colors[mask]


def write_ply(fn, verts, colors, ascii_fmt=True):
    verts, colors = verts.reshape(-1, 3), colors.reshape(-1, 3)
    if ascii_fmt:                                      
        with open(fn, "wb") as f:
            f.write((PLY_HEADER % ("ascii", len(verts))).encode())
            np.savetxt(f, np.hstack([verts, colors]), fmt="%f %f %f %d %d %d ")
    else:
        rec = np.empty(len(verts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = verts.astype(np.float32), colors.astype(np.uint8)
        with open(fn, "wb") as f:
            f.write((PLY_HEADER % ("binary_little_endian", len(verts))).encode())
            f.write(rec.tobytes())


def show_open3d(ply, title="3D"):
    try:
        import open3d as o3d
    except ImportError:
        print("open3d не встановлено (pip install open3d) - показую 3D-модель через matplotlib")
        return False
    pcd = o3d.io.read_point_cloud(ply)
    print(pcd)
    print(np.asarray(pcd.points))
    c = np.asarray(pcd.points).mean(axis=0)
    o3d.visualization.draw_geometries([pcd], window_name=title, zoom=0.5, lookat=c.tolist(),
                                      front=[0.0, 0.0, 1.0], up=[0.0, 1.0, 0.0])
    return True


def fig_3d(pts, col, title="3D-модель (обертайте мишею)"):
    """Резервна інтерактивна 3D-модель, якщо Open3D не встановлено."""
    from mpl_toolkits.mplot3d import Axes3D  
    s = max(1, len(pts) // 30000)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pts[::s, 0], pts[::s, 2], pts[::s, 1], c=col[::s] / 255.0, s=0.6)
    ax.set_xlabel("X, мм"); ax.set_ylabel("Z, мм"); ax.set_zlabel("Y, мм")
    ax.set_title(title)
    fig.canvas.manager.set_window_title(title)
    return fig


def display_all(info, args):
    """Вивести усі результати на екран: графіки, а потім 3D-модель."""
    if args.no_show:
        return
    if FIGS:
        plt.show(block=False)
        plt.pause(0.5)
    if info is not None and not args.no3d:
        ply, pts, col = info
        if not show_open3d(ply, "3D-модель (Open3D)"):
            fig_3d(pts, col)
            plt.show(block=False)
            plt.pause(0.5)
    if plt.get_fignums():
        print("\nВікна з результатами відкрито. Закрийте їх, щоб завершити програму.")
        plt.show()


def render_cloud(ax_xy, ax_xz, pts, col, title=""):
    s = max(1, len(pts) // 40000)
    ax_xy.scatter(pts[::s, 0], pts[::s, 1], c=col[::s] / 255.0, s=0.4)
    ax_xz.scatter(pts[::s, 0], pts[::s, 2], c=col[::s] / 255.0, s=0.4)
    ax_xy.set_title(title + "  вид спереду (X-Y), мм", fontsize=8)
    ax_xz.set_title("вид зверху (X-Z), мм", fontsize=8)
    for a in (ax_xy, ax_xz):
        a.set_aspect("equal"); a.tick_params(labelsize=7)


def save_fig(fig, path, dpi=130):
    """Зберегти фігуру у файл і залишити для показу у вікні."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    FIGS.append(fig)


def colorize(disp, vmax):
    d = np.nan_to_num(disp, nan=0.0)
    img = cv2.applyColorMap((np.clip(d / vmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[~np.isfinite(disp)] = 0
    return img


def evaluate(disp, gt, f, B, dmin_eff, tol=1.0):
    nan = float("nan")
    if gt is None:
        return dict(cover=float(np.isfinite(disp).mean() * 100), rmse=nan, mae=nan,
                    bias=nan, bad=nan, zrel=nan, zrmse=nan)
    vg = gt > 0
    m = vg & np.isfinite(disp)
    if m.sum() < 50:
        return dict(cover=0.0, rmse=nan, mae=nan, bias=nan, bad=nan, zrel=nan, zrmse=nan)
    e = disp[m] - gt[m]
    zest = f * B / (disp[m] + dmin_eff)
    zgt = f * B / (gt[m] + dmin_eff)
    return dict(cover=float(m.sum() / vg.sum() * 100),
                rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                bias=float(np.median(e)), bad=float(np.mean(np.abs(e) > tol) * 100),
                zrel=float(np.median(np.abs(zest - zgt) / zgt) * 100),
                zrmse=float(np.sqrt(np.mean((zest - zgt) ** 2))))


def run_pair(ds, a, b, block=5, mode="sgbm", nd_mult=1.0, nd=None):
    """Повний цикл для пари (a,b): matching -> метрики -> хмара."""
    L, R = ds.view(a), ds.view(b)
    gt = ds.gt_pair(a, b)
    if nd is None:
        nd = int(np.ceil(num_disp_from_gt(gt) * nd_mult / 16.0) * 16)
    disp = compute_disparity(L, R, nd, block, mode)
    B = STEP_MM * (b - a)
    de = dmin_for(ds, a, b)
    met = evaluate(disp, gt, ds.f, B, de, tol=1.0)
    return dict(L=L, disp=disp, gt=gt, nd=nd, B=B, dmin_eff=de, met=met)


def geometry(ds, a, b, zobj):
    B = STEP_MM * (b - a)
    theta = np.degrees(2 * np.arctan(B / (2 * zobj)))
    xmid = STEP_MM * ((a + b) / 2.0 - CENTER_VIEW)
    phi = np.degrees(np.arctan(xmid / zobj))
    dz = zobj ** 2 * 1.0 / (ds.f * B)      
    return B, theta, phi, dz


def mode_recon(ds, args, out):
    a, b = args.left, args.right
    print(f"\n=== Реконструкція: view{a} - view{b} ===")
    r = run_pair(ds, a, b, args.window, "sgbm")
    print(f"f = {ds.f:.1f} px, база = {r['B']:.0f} мм, dmin(ефект.) = {r['dmin_eff']:.1f} px, numDisparities = {r['nd']}")
    pts, col = disparity_to_cloud(r["disp"], r["L"], ds.f, r["B"], r["dmin_eff"])
    ply = os.path.join(out, f"cloud_{a}_{b}.ply")
    write_ply(ply, pts, col, ascii_fmt=True)
    print(f"{ply} saved ({len(pts)} точок)")
    m = r["met"]
    if r["gt"] is not None:
        print("Порівняння з еталоном: RMSE={rmse:.2f} px, MAE={mae:.2f} px, зсув(медіана)={bias:+.2f} px, "
              "bad>1px={bad:.1f}%, покриття={cover:.1f}%, відн. похибка Z={zrel:.1f}%".format(**m))
        if abs(m["bias"]) > 3:
            print("  УВАГА: великий систематичний зсув. Перевірте масштаб еталона (--gt_scale) та dmin.")
    cv2.imwrite(os.path.join(out, "left.png"), r["L"])
    vmax = max(r["nd"], 1)
    cv2.imwrite(os.path.join(out, "disparity.png"), colorize(r["disp"], vmax))
    gt = r["gt"]
    if gt is not None:
        cv2.imwrite(os.path.join(out, "gt.png"), colorize(np.where(gt > 0, gt, np.nan), vmax))
        err = np.abs(r["disp"] - gt)
        err[~(gt > 0)] = np.nan
        cv2.imwrite(os.path.join(out, "error.png"), colorize(err, 8))

    cmap = plt.get_cmap("jet").copy(); cmap.set_bad("black")
    fig = plt.figure(figsize=(15, 8))
    fig.canvas.manager.set_window_title(f"Результат реконструкції view{a}-view{b}")
    gs = fig.add_gridspec(2, 3)
    ax = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    ax[0].imshow(cv2.cvtColor(r["L"], cv2.COLOR_BGR2RGB)); ax[0].set_title(f"Ліве зображення (view{a})")
    im = ax[1].imshow(np.ma.masked_invalid(r["disp"]), cmap=cmap, vmin=0, vmax=vmax)
    ax[1].set_title("Карта диспаритету SGBM, px"); fig.colorbar(im, ax=ax[1], fraction=0.046)
    if gt is not None:
        im = ax[2].imshow(np.ma.masked_where(~(gt > 0), gt), cmap=cmap, vmin=0, vmax=vmax)
        ax[2].set_title("Еталон (disp1), px"); fig.colorbar(im, ax=ax[2], fraction=0.046)
        im = ax[3].imshow(np.ma.masked_invalid(err), cmap=cmap, vmin=0, vmax=8)
        ax[3].set_title("Модуль похибки, px"); fig.colorbar(im, ax=ax[3], fraction=0.046)
    else:
        ax[2].axis("off"); ax[3].axis("off")
    for a_ in ax[:4]:
        a_.axis("off")
    render_cloud(ax[4], ax[5], pts, col, f"B = {r['B']:.0f} мм")
    if gt is not None:
        fig.suptitle("view{}-view{}, B = {:.0f} мм | RMSE = {rmse:.2f} px, bad>1px = {bad:.1f}%, "
                     "покриття = {cover:.1f}%, відн. похибка Z = {zrel:.1f}%".format(a, b, r["B"], **m),
                     fontsize=11)
    save_fig(fig, os.path.join(out, "recon_dashboard.png"), 130)
    return ply, pts, col


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_table(rows, title):
    print(f"\n--- {title} ---")
    print(f"{'пара':<8}{'B,мм':>6}{'θ,°':>7}{'φ,°':>7}{'cover%':>8}{'RMSE':>7}{'MAE':>7}"
          f"{'bias':>7}{'bad%':>7}{'Zrel%':>7}{'dZтеор':>8}")
    for r in rows:
        print(f"{r['pair']:<8}{r['B']:>6.0f}{r['theta']:>7.2f}{r['phi']:>7.2f}{r['cover']:>8.1f}"
              f"{r['rmse']:>7.2f}{r['mae']:>7.2f}{r['bias']:>7.2f}{r['bad']:>7.1f}{r['zrel']:>7.1f}{r['dz_theory']:>8.1f}")


def mode_study(ds, args, out):
    if ds.d1 is None:
        raise SystemExit("Для дослідження потрібен еталон disp1.png")
    if not (ds.has(1) and ds.has(5)):
        raise SystemExit("Потрібні щонайменше view1.png та view5.png")
    zobj = ds.z_median()
    print(f"\n=== Дослідження. Типова відстань до об'єкта Z ≈ {zobj:.0f} мм ===")
    cols = ["series", "pair", "B", "theta", "phi", "nd", "cover", "rmse", "mae", "bias", "bad",
            "zrel", "zrmse", "dz_theory"]
    plydir = os.path.join(out, "ply"); os.makedirs(plydir, exist_ok=True)
    cache = {}

    def do(a, b, series, **kw):
        key = (a, b, tuple(sorted(kw.items())))
        r = cache.get(key) or run_pair(ds, a, b, **kw)
        cache[key] = r
        B, th, ph, dz = geometry(ds, a, b, zobj)
        row = dict(series=series, pair=f"{a}-{b}", B=B, theta=th, phi=ph, nd=r["nd"],
                   dz_theory=dz, **r["met"])
        return row, r

    base_rows, base_res = [], []
    for k in range(2, 7):
        if not ds.has(k):
            print(f"  (пропуск пари 1-{k}: немає view{k}.png)")
            continue
        row, r = do(1, k, "база")
        base_rows.append(row); base_res.append((k, r))
        pts, col = disparity_to_cloud(r["disp"], r["L"], ds.f, r["B"], r["dmin_eff"])
        write_ply(os.path.join(plydir, f"cloud_1_{k}.ply"), pts, col, ascii_fmt=False)
    print_table(base_rows, "1. Вплив БАЗИ (view1 - view k)")

    ang_rows = []
    for a in range(0, 5):
        if not (ds.has(a) and ds.has(a + 2)):
            print(f"  (пропуск пари {a}-{a + 2}: немає view{a}.png або view{a + 2}.png)")
            continue
        row, _ = do(a, a + 2, "ракурс")
        ang_rows.append(row)
    print_table(ang_rows, "2. Вплив РАКУРСУ (база 80 мм, пара зсувається вздовж рейки)")

    par_rows = []
    for blk in (3, 5, 9, 15):
        row, _ = do(1, 5, f"blockSize={blk}", block=blk); par_rows.append(row)
    for mult in (0.5, 1.0, 1.5):
        row, _ = do(1, 5, f"numDisp×{mult}", nd_mult=mult); par_rows.append(row)
    for mode in ("bm", "sgbm"):
        row, _ = do(1, 5, f"алгоритм={mode.upper()}", block=9, mode=mode); par_rows.append(row)
    print(f"\n--- 3. Вплив параметрів алгоритму (пара 1-5, B=160 мм) ---")
    print(f"{'налаштування':<18}{'cover%':>8}{'RMSE':>8}{'bad%':>8}{'Zrel%':>8}")
    for r in par_rows:
        print(f"{r['series']:<18}{r['cover']:>8.1f}{r['rmse']:>8.2f}{r['bad']:>8.1f}{r['zrel']:>8.1f}")

    write_csv(os.path.join(out, "study_baseline.csv"), base_rows, cols)
    write_csv(os.path.join(out, "study_angle.csv"), ang_rows, cols)
    write_csv(os.path.join(out, "study_params.csv"), par_rows, cols)

    fig, axs = plt.subplots(2, 1, figsize=(13, 6.5))
    fig.canvas.manager.set_window_title("Підсумкові таблиці дослідження")
    heads = ["пара", "B, мм", "θ, °", "φ, °", "покриття, %", "RMSE, px", "bad>1px, %", "відн. Z, %", "ΔZ теор., мм"]
    for a_, rows, ttl in ((axs[0], base_rows, "Вплив бази (ліва камера view1)"),
                          (axs[1], ang_rows, "Вплив ракурсу (база 80 мм)")):
        a_.axis("off"); a_.set_title(ttl, fontsize=11)
        if rows:
            cells = [[r["pair"], f"{r['B']:.0f}", f"{r['theta']:.2f}", f"{r['phi']:.2f}", f"{r['cover']:.1f}",
                      f"{r['rmse']:.2f}", f"{r['bad']:.1f}", f"{r['zrel']:.1f}", f"{r['dz_theory']:.1f}"]
                     for r in rows]
            t = a_.table(cellText=cells, colLabels=heads, loc="center", cellLoc="center")
            t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.6)
    save_fig(fig, os.path.join(out, "fig_tables.png"), 130)

    Bs = [r["B"] for r in base_rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(Bs, [r["rmse"] for r in base_rows], "o-", label="RMSE, px")
    ax[0].plot(Bs, [r["mae"] for r in base_rows], "s--", label="MAE, px")
    ax[0].set_xlabel("база B, мм"); ax[0].set_title("Похибка диспаритету"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(Bs, [r["zrel"] for r in base_rows], "o-", label="відн. похибка Z, % (факт)")
    ax[1].plot(Bs, [r["dz_theory"] / zobj * 100 for r in base_rows], "k--", label="теорія для dd=1 px")
    ax[1].set_xlabel("база B, мм"); ax[1].set_title("Похибка глибини"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(Bs, [r["cover"] for r in base_rows], "o-", label="покриття еталона, %")
    ax[2].plot(Bs, [r["bad"] for r in base_rows], "s--", label="bad>1px, %")
    ax[2].set_xlabel("база B, мм"); ax[2].set_title("Покриття та частка помилок"); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.suptitle("Вплив бази стереопари (Bowling2, ліва камера view1)")
    save_fig(fig, os.path.join(out, "fig_baseline.png"), 130)

    ph = [r["phi"] for r in ang_rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(ph, [r["rmse"] for r in ang_rows], "o-"); ax[0].set_title("RMSE диспаритету, px")
    ax[1].plot(ph, [r["zrel"] for r in ang_rows], "o-"); ax[1].set_title("Відн. похибка Z, %")
    ax[2].plot(ph, [r["cover"] for r in ang_rows], "o-"); ax[2].set_title("Покриття еталона, %")
    for a_ in ax:
        a_.set_xlabel("ракурс φ, °"); a_.grid(alpha=.3)
    fig.suptitle("Вплив ракурсу на об'єкт (B = 80 мм, пара зсувається вздовж рейки)")
    save_fig(fig, os.path.join(out, "fig_angle.png"), 130)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a_, key, ttl in zip(ax, ("rmse", "bad", "cover"), ("RMSE, px", "bad>1px, %", "покриття, %")):
        a_.barh([r["series"] for r in par_rows], [r[key] for r in par_rows], color="#4c72b0")
        a_.set_title(ttl); a_.invert_yaxis(); a_.tick_params(labelsize=8); a_.grid(alpha=.3, axis="x")
    fig.suptitle("Вплив параметрів алгоритму (пара view1-view5)")
    save_fig(fig, os.path.join(out, "fig_params.png"), 130)

    vmax = max(r["nd"] for _, r in base_res)
    n = len(base_res)
    fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 6))
    for i, (k, r) in enumerate(base_res):
        ax[0, i].imshow(cv2.cvtColor(colorize(r["disp"], vmax), cv2.COLOR_BGR2RGB))
        ax[0, i].set_title(f"B = {r['B']:.0f} мм", fontsize=9); ax[0, i].axis("off")
        ax[1, i].imshow(cv2.cvtColor(colorize(np.where(r["gt"] > 0, r["gt"], np.nan), vmax), cv2.COLOR_BGR2RGB))
        ax[1, i].set_title("еталон", fontsize=9); ax[1, i].axis("off")
    fig.suptitle("Карти диспаритету для різних баз (зверху) та еталон (знизу)")
    save_fig(fig, os.path.join(out, "fig_disparity_baselines.png"), 120)

    show = [base_res[0], base_res[2], base_res[4]]         
    fig, ax = plt.subplots(2, len(show), figsize=(4.5 * len(show), 8))
    for i, (k, r) in enumerate(show):
        pts, col = disparity_to_cloud(r["disp"], r["L"], ds.f, r["B"], r["dmin_eff"])
        render_cloud(ax[0, i], ax[1, i], pts, col, f"B = {r['B']:.0f} мм")
    save_fig(fig, os.path.join(out, "fig_clouds_baselines.png"), 120)
    print(f"\nТаблиці (study_*.csv), графіки (fig_*.png) та хмари точок збережено в '{out}'")


def mode_custom(args, out):
    """Власна стереопара: два зображення + відомі f (px) та база (мм)."""
    L, R = cv2.imread(args.left_img), cv2.imread(args.right_img)
    if L is None or R is None:
        raise SystemExit("Не вдалося прочитати зображення")
    nd = args.num_disp or 128
    disp = compute_disparity(L, R, nd, args.window, "sgbm")
    pts, col = disparity_to_cloud(disp, L, args.focal, args.baseline, 0.0)
    ply = os.path.join(out, "custom.ply")
    write_ply(ply, pts, col)
    cv2.imwrite(os.path.join(out, "custom_disparity.png"), colorize(disp, nd))
    print(f"{ply} saved ({len(pts)} точок)")
    return ply, pts, col

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="all", choices=["recon", "study", "all", "custom"])
    ap.add_argument("--root", default="Bowling2")
    ap.add_argument("--illum", type=int, default=2)
    ap.add_argument("--exp", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.5, help="1.0 повний розмір, 0.5 половина")
    ap.add_argument("--gt_scale", type=float, default=1.0, help="поправка масштабу еталона, якщо є зсув")
    ap.add_argument("--left", type=int, default=1)
    ap.add_argument("--right", type=int, default=5)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--no3d", action="store_true", help="не відкривати 3D-вікно")
    ap.add_argument("--no_show", action="store_true", help="не показувати вікна (лише зберегти файли)")
    ap.add_argument("--view3d", action="store_true", help=argparse.SUPPRESS)   
    ap.add_argument("--out", default="results")
    ap.add_argument("--left_img"); ap.add_argument("--right_img")
    ap.add_argument("--focal", type=float, default=700.0)
    ap.add_argument("--baseline", type=float, default=60.0)
    ap.add_argument("--num_disp", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    info = None
    if args.mode == "custom":
        info = mode_custom(args, args.out)
    else:
        ds = Dataset(args.root, args.illum, args.exp, args.scale, args.gt_scale)
        if args.mode in ("recon", "all"):
            info = mode_recon(ds, args, args.out)
        if args.mode in ("study", "all"):
            mode_study(ds, args, args.out)
    display_all(info, args)