import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib import ticker
from scipy.odr import ODR, Model, RealData
from sklearn.metrics import r2_score
from scipy import stats

# region ### add the sys.path to search for custom modules ###
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent

plt.rcParams.update(
    {
        "font.size": 7,
        "font.family": "Arial",
        "legend.fontsize": 6,
        "axes.formatter.limits": (-3, 6),
        "axes.formatter.use_mathtext": True,
    }
)


def linear_function(p, x):
    return p[0] + p[1] * x


def fit_regression(x, y, xerr, yerr):
    """Fit ODR regression and compute confidence intervals."""
    x_fit = np.logspace(np.log10(np.min(lower_x)), np.log10(np.max(upper_x)), 100)
    t_val = 1.96
    y_arr = np.array(y)
    xerr_1d = np.mean(np.abs(xerr), axis=0)
    yerr_1d = np.mean(np.abs(yerr), axis=0)
    log_x = np.log10(x)
    log_y = np.log10(y_arr)

    sx = xerr_1d / (x * np.log(10))
    sy = yerr_1d / (y_arr * np.log(10))

    data = RealData(log_x, log_y, sx=sx, sy=sy)
    model = Model(linear_function)
    odr_inst = ODR(data, model, beta0=[0, 1])
    output = odr_inst.run()

    log_x_fit = np.log10(x_fit)
    log_y_fit = linear_function(output.beta, log_x_fit)
    y_fit = 10**log_y_fit

    residuals = log_y - linear_function(output.beta, log_x)
    residual_std = np.std(residuals, ddof=1)
    mean_log_x = np.mean(log_x)
    sxx = np.sum((log_x - mean_log_x) ** 2)
    y_ci_log = t_val * residual_std * np.sqrt(1 / len(log_x) + (log_x_fit - mean_log_x) ** 2 / sxx)
    y_lower = 10 ** (log_y_fit - y_ci_log)
    y_upper = 10 ** (log_y_fit + y_ci_log)

    return output.beta, output.sd_beta, y_fit, y_lower, y_upper


fig = plt.figure(figsize=(5.5, 3))
gs = gridspec.GridSpec(1, 2)
xlim_l = 1e4
xlim_u = 1e6
xlim_major = 10_000
xlim_minor = 2_500
########################################################################################################
data = np.load("./img1.npz")
ax = plt.subplot(gs[0])

print(data["target_x"], data["target_y"])
ax.set_title(label="(a)", loc="left", fontsize=8, fontweight="bold")

ax.errorbar(
    data["target_x"],
    data["target_y"],
    yerr=data["target_y_err"],
    xerr=data["target_x_err"],
    fmt="o",
    color="grey",
    label="Measured CIF (2019)",
    markersize=5,
    markeredgecolor="k",
    capsize=2,
    elinewidth=2,
    ecolor="grey",
    zorder=2,
    alpha=1,
)
ax.errorbar(
    data["inv_x"],
    data["inv_y"],
    yerr=data["inv_y_err"],
    xerr=data["inv_x_err"],
    fmt="o",
    color="C0",
    label="Inverted CIF (2019)",
    markersize=5,
    markeredgecolor="k",
    capsize=2,
    elinewidth=2,
    ecolor="C0",
    zorder=3,
    alpha=0.8,
)

x = np.array(data["target_x"])
y = np.array(data["target_y"])
# x, y = x, y
lower_x = x - x * 0.5
upper_x = x + x * 0.5
x_err = data["target_x_err"]
y_err = data["target_y_err"]
xerr_1d = np.mean(np.abs(x_err), axis=0)
yerr_1d = np.mean(np.abs(y_err), axis=0)
log_x_lower, log_x, log_x_upper = np.log10(lower_x), np.log10(x), np.log10(upper_x)
log_y = np.log10(y)
sx = xerr_1d / (x * np.log(10))
sxl = xerr_1d / (lower_x * np.log(10))
sxu = xerr_1d / (upper_x * np.log(10))
sy = yerr_1d / (y * np.log(10))

# Lower Regression Line
data = RealData(log_x_lower, log_y, sx=sxl, sy=sy)
model = Model(linear_function)
odr_inst = ODR(data, model, beta0=[0, 1])
output = odr_inst.run()
p_fit_l = output.beta
p_std_l = output.sd_beta

x_fit = np.logspace(np.log10(xlim_l), np.log10(xlim_u), 100)
log_x_fit = np.log10(x_fit)
log_y_fit = linear_function(p_fit_l, log_x_fit)
y_fit_l = 10**log_y_fit

ax.plot(x_fit, y_fit_l, "k--", linewidth=2, alpha=0.7, zorder=0)

# Upper Regression Line
data = RealData(log_x_upper, log_y, sx=sxu, sy=sy)
model = Model(linear_function)
odr_inst = ODR(data, model, beta0=[0, 1])
output = odr_inst.run()
p_fit_u = output.beta
p_std_u = output.sd_beta

x_fit = np.logspace(np.log10(xlim_l), np.log10(xlim_u), 100)
log_x_fit = np.log10(x_fit)
log_y_fit = linear_function(p_fit_u, log_x_fit)
y_fit_u = 10**log_y_fit

ax.plot(x_fit, y_fit_u, "k--", linewidth=2, alpha=0.7, zorder=0)

# Central Regression Line
data = RealData(log_x, log_y, sx=sx, sy=sy)
model = Model(linear_function)
odr_inst = ODR(data, model, beta0=[0, 1])
output = odr_inst.run()

p_fit = output.beta
p_std = output.sd_beta

x_fit = np.logspace(np.log10(xlim_l), np.log10(xlim_u), 100)
log_x_fit = np.log10(x_fit)
log_y_fit = linear_function(p_fit, log_x_fit)
y_fit = 10**log_y_fit
t_val = 1.96
residuals = log_y - linear_function(p_fit, log_x)
residual_std = np.std(residuals, ddof=1)
mean_log_x = np.mean(log_x)
sxx = np.sum((log_x - mean_log_x) ** 2)
y_ci_log = t_val * residual_std * np.sqrt(1 / len(log_x) + (log_x_fit - mean_log_x) ** 2 / sxx)
y_lower = 10 ** (log_y_fit - y_ci_log)
y_upper = 10 ** (log_y_fit + y_ci_log)

ax.plot(x_fit, y_fit, "k-", linewidth=2, alpha=0.7, zorder=0)
ax.fill_between(x_fit, y_lower, y_upper, alpha=0.2, color="grey", label="95% CI", zorder=0)

r2_log = r2_score(log_y, linear_function(p_fit, log_x))
corr_log, _ = stats.pearsonr(log_x, log_y)
info_text = (
    f"Slope: {p_fit[1]:.3f} ± {p_std[1]:.3f}\n"
    f"Intercept: {p_fit[0]:.3f} ± {p_std[0]:.3f}\n"
    f"Log-log R²: {r2_log:.3f}\n"
    f"Pearson r: {corr_log:.3f}"
)
ax.text(
    0.55,
    0.35,
    info_text,
    transform=ax.transAxes,
    fontsize=5,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
)

ax.legend(loc="best", fontsize=7)
ax.grid(True, linestyle="--", alpha=0.5, which="both")

########################################################################################################
data = np.load("./img2.npz")
ax = plt.subplot(gs[1])

ax.set_title(label="(b)", loc="left", fontsize=8, fontweight="bold")

ax.plot(x_fit, y_fit_l, "k--", linewidth=2, alpha=0.7, zorder=0)
ax.plot(x_fit, y_fit_u, "k--", linewidth=2, alpha=0.7, zorder=0)
ax.plot(x_fit, y_fit, "k-", linewidth=2, alpha=0.7, zorder=0)
ax.fill_between(x_fit, y_lower, y_upper, alpha=0.2, color="grey", label="95% CI", zorder=0)

for year, color in zip(["2020", "2021", "2022", "2023"], ["C1", "C2", "C3", "C4"]):
    ax.errorbar(
        data[f"{year}_x"],
        data[f"{year}_y"],
        yerr=data[f"{year}_y_err"],
        xerr=data[f"{year}_x_err"],
        fmt="o",
        color=color,
        label=f"CIF ({year})",
        markersize=5,
        markeredgecolor="k",
        capsize=2,
        elinewidth=2,
        ecolor=color,
        zorder=2,
        alpha=0.8,
    )

ax.text(0.30, 0.7, "1", transform=ax.transAxes, fontsize=8, verticalalignment="top", zorder=5)
ax.text(0.31, 0.82, "2", transform=ax.transAxes, fontsize=8, verticalalignment="top", zorder=5)
ax.text(0.4, 0.82, "3", transform=ax.transAxes, fontsize=8, verticalalignment="top", zorder=5)
ax.text(0.48, 0.22, "4", transform=ax.transAxes, fontsize=8, verticalalignment="top", zorder=5)
ax.text(0.53, 0.47, "5", transform=ax.transAxes, fontsize=8, verticalalignment="top", zorder=5)
for a in gs:
    ax = plt.subplot(a)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.tick_params(axis="both", which="major", labelsize=7, length=6)
    ax.tick_params(axis="both", which="minor", labelsize=7, length=3)
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.set_ylabel("Volume (m³)", weight="bold")
    ax.set_xlabel("Cumulative Impact Force (kNs)", weight="bold")
    ax.set_xlim(xlim_l, xlim_u)
    ax.set_ylim(1e3, 5e5)
    ax.legend(loc="best", fontsize=6)
    ax.grid(True, linestyle="--", alpha=0.5, which="both")
    ax.text(0.35, 0.97, "Lower\nBound", transform=ax.transAxes, fontsize=6, verticalalignment="top")
    ax.text(0.55, 0.97, "log-log\nfit", transform=ax.transAxes, fontsize=6, verticalalignment="top")
    ax.text(0.80, 0.97, "Upper\nBound", transform=ax.transAxes, fontsize=6, verticalalignment="top")


plt.tight_layout()
png_path = Path(current_dir) / "Figure-6.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(png_path, dpi=600)
# plt.show()
plt.close(fig)
