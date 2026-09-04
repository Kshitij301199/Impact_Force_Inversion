import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
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


fig = plt.figure(figsize=(5.5, 3))
gs = gridspec.GridSpec(1, 2)
xlim_l = 1e4
xlim_u = 1e6
xlim_major = 10_000
xlim_minor = 2_500


########################################################################################################
def linear_function(p, x):
    return p[0] + p[1] * x


data = np.load("./fig8-cache/fig1.npz")
ax = plt.subplot(gs[0])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title(label="(a)", loc="left", fontsize=8, fontweight="bold")
ax.set_xlabel("Cumulative Impact Force (kNs)", fontweight="bold")
ax.set_ylabel("Cumulative seismically derived vertical forces (kN)", fontweight="bold")
ax.set_ylim(3e2, 1e4)
ax.set_xlim(1e4, 1e6)
ax.grid(True, linestyle="--", alpha=0.5, which="both", zorder=0)
for year, c in zip([2019, 2020, 2021, 2022, 2023], ["C0", "C1", "C2", "C3", "C4"]):
    ax.errorbar(
        x=data[f"{year}_x"],
        y=data[f"{year}_y"],
        xerr=data[f"{year}_x_err"],
        yerr=None,
        capsize=2,
        ls="none",
        elinewidth=2,
        color=c,
        mfc=c,
        marker="o",
        markeredgecolor="k",
        markersize=5,
        label=f"{year}",
        alpha=1,
    )
x = np.array(data["x"])
y = np.array(data["y"])
xerr = data["x_err"]
yerr = np.zeros_like(y) + 1e-13
xerr_1d = np.mean(np.abs(xerr), axis=0)
yerr_1d = np.mean(np.abs(yerr), axis=0)
log_x = np.log10(x)
log_y = np.log10(y)
sx = xerr_1d / (x * np.log(10))
sy = yerr_1d / (y * np.log(10))

data = RealData(log_x, log_y, sx=sx, sy=sy)
model = Model(linear_function)
odr_inst = ODR(data, model, beta0=[0, 1])
output = odr_inst.run()

p_fit = output.beta
p_std = output.sd_beta

x_fit = np.logspace(np.log10(1e4), np.log10(1e6), 1000)
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

ax.plot(x_fit, y_fit, "k-", linewidth=2, alpha=0.5, zorder=0)
ax.fill_between(x_fit, y_lower, y_upper, alpha=0.2, color="grey", zorder=0)

r2_log = r2_score(log_y, linear_function(p_fit, log_x))
corr_log, _ = stats.pearsonr(log_x, log_y)

info_text = (
    f"Slope: {p_fit[1]:.3f} ± {p_std[1]:.3f}\n"
    f"Intercept: {p_fit[0]:.3f} ± {p_std[0]:.3f}\n"
    f"Log-log R²: {r2_log:.3f}\n"
    f"Pearson r: {corr_log:.3f}"
)
ax.text(
    0.05,
    0.95,
    info_text,
    transform=ax.transAxes,
    fontsize=5,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
)
ax.legend(loc="best", fontsize=6)


########################################################################################################
def linear_func(beta, x):
    return beta[0] * x + beta[1]


data = np.load("./fig8-cache/fig2.npz")
ax = plt.subplot(gs[1])
ax.set_title(label="(b)", loc="left", fontsize=8, fontweight="bold")
ax.set_xlabel("Cumulative Impact Force (kNs)", fontweight="bold")
ax.set_ylabel("Downstream Reach Change (m³/m)", fontweight="bold")
ax.set_ylim(-8, 8)
ax.set_xlim(0, 2e5)
ax.axhspan(0, 8, facecolor="green", alpha=0.1)
ax.axhspan(-8, 0, facecolor="red", alpha=0.1)
ax.text(
    0.98,
    0.52,
    "Deposition",
    transform=ax.transAxes,
    fontsize=6,
    verticalalignment="center",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0),
)
ax.text(
    0.98,
    0.48,
    "Erosion",
    transform=ax.transAxes,
    fontsize=6,
    verticalalignment="center",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0),
)
ax.grid(True, linestyle="--", alpha=0.5, which="both", zorder=0)

for year, c in zip([2019, 2020, 2021, 2022, 2023], ["C0", "C1", "C2", "C3", "C4"]):
    ax.errorbar(
        x=data[f"x_{year}"],
        y=data[f"y_{year}"],
        xerr=data[f"x_err_{year}"],
        yerr=None,
        capsize=2,
        ls="none",
        elinewidth=2,
        color=c,
        mfc=c,
        marker="o",
        markeredgecolor="k",
        markersize=5,
        label=f"{year}",
        alpha=1,
        zorder=1,
    )
ax.errorbar(
    x=data["outlier_x"],
    y=data["outlier_y"],
    xerr=None,
    yerr=None,
    capsize=2,
    ls="none",
    elinewidth=2,
    color="k",
    mfc="k",
    marker="o",
    markeredgecolor="k",
    markersize=5,
    label="Outlier",
    alpha=1,
    zorder=2,
)

sel_x = np.array(data["sel_x"])
sel_y = np.array(data["sel_y"])
sel_x_err = data["sel_x_err"]
sel_x_err_mean = sel_x_err.mean(axis=0)

data = RealData(sel_x, sel_y, sx=sel_x_err_mean)
model = Model(linear_func)
odr_inst = ODR(data, model, beta0=[-0.001, 3.0])
output = odr_inst.run()

slope, intercept = output.beta
slope_err, intercept_err = output.sd_beta

# Calculate prediction line and confidence interval
plot_x = np.linspace(0, 200_000, 1000)
y_pred = slope * plot_x + intercept

# Calculate confidence interval
residuals = sel_y - (slope * sel_x + intercept)
residual_std = np.std(residuals, ddof=1)
mean_sel_x = np.mean(sel_x)
sxx = np.sum((sel_x - mean_sel_x) ** 2)
t_val = 1.96

y_ci = t_val * residual_std * np.sqrt(1 / len(sel_x) + (plot_x - mean_sel_x) ** 2 / sxx)
y_lower = y_pred - y_ci
y_upper = y_pred + y_ci

ax.fill_between(plot_x, y_lower, y_upper, alpha=0.2, color="grey", zorder=0)
ax.plot(plot_x, slope * plot_x + intercept, linestyle="--", color="k", alpha=0.6, zorder=0)

r2_log = r2_score(sel_y, linear_function([slope, intercept], sel_x))
corr_log, _ = stats.pearsonr(sel_x, sel_y)

info_text = f"Pearson r: {corr_log:.3f}"
ax.text(
    0.05,
    0.05,
    info_text,
    transform=ax.transAxes,
    fontsize=6,
    verticalalignment="center",
    horizontalalignment="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0),
)
ax.legend(loc="upper left", fontsize=6)

plt.tight_layout()
png_path = Path(current_dir) / "Figure-8.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(png_path, dpi=600)
# plt.show()
plt.close(fig)
