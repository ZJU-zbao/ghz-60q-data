#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import seaborn as sns

root_path = os.path.abspath(".")

colors_iter = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
]


def plot_fig1():
    data_load = sio.loadmat(f"{root_path}/data/fig1.mat", simplify_cells=True)
    plt.figure(figsize=(8, 4))
    for processor in [1, 2]:
        plt.errorbar(
            data_load[f"qnum_processor_{processor}"],
            data_load[f"fid_processor_{processor}"][0],
            data_load[f"fid_processor_{processor}"][1],
            label=f"processor {processor}",
        )
    plt.title("fig1.b")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.errorbar(
        data_load["phi"],
        data_load["k_phi"][0],
        data_load["k_phi"][1],
    )
    plt.subplot(212)
    plt.errorbar(
        data_load["q"],
        data_load["kf_q"][0],
        data_load["kf_q"][1],
    )
    plt.suptitle("fig1.d")
    plt.show()


def plot_fig2():
    data_load = sio.loadmat(f"{root_path}/data/fig2.mat", simplify_cells=True)
    plt.figure(figsize=(6, 10))
    plt.subplot(311)
    plt.errorbar(data_load["phi"], data_load["k_phi_0"][0], data_load["k_phi_0"][1])
    plt.title("t=0T")
    for i in [2, 3]:
        plt.subplot(3, 1, i)
        for evo in ["dtc", "thermal"]:
            plt.errorbar(
                data_load["phi"],
                data_load[f"k_phi_{evo}"][0, i - 2],
                data_load[f"k_phi_{evo}"][1, i - 2],
                label=evo,
            )
        plt.title(f"t={i-1}T")
        if i == 2:
            plt.legend()
    plt.show()


def plot_fig3():
    data_load = sio.loadmat(f"{root_path}/data/fig3.mat", simplify_cells=True)
    plt.figure(figsize=(8, 4))
    plt.errorbar(
        data_load["t"],
        data_load["kf_DTC"][0],
        data_load["kf_DTC"][1],
        label="DTC",
        color="tab:red",
    )
    plt.plot(
        data_load["t"],
        (1 - data_load["ep_DTC"]) ** (data_load["N"] * data_load["t"])
        * data_load["const"]
        * data_load["kf_DTC"][0][0],
        ls="--",
        color="tab:red",
    )
    plt.errorbar(
        data_load["t"],
        data_load["kf_Rabi"][0],
        data_load["kf_Rabi"][1],
        label="Rabi",
        color="tab:blue",
    )
    plt.plot(
        data_load["t"],
        (1 - data_load["ep_Rabi"]) ** (data_load["N"] * data_load["t"])
        * np.exp(-data_load["N"] * data_load["lambda_eff"] ** 2 * data_load["t"] ** 2)
        * data_load["kf_Rabi"][0][0],
        ls="--",
        color="tab:blue",
    )
    plt.ylim([3e-6, 2e-1])
    plt.yscale("log")
    plt.legend()
    plt.title("fig3")
    plt.show()


def plot_fig4():
    data_load = sio.loadmat(f"{root_path}/data/fig4.mat", simplify_cells=True)
    qnum = data_load["qnum"]
    xx, yy = np.meshgrid(np.arange(1, qnum + 1), np.arange(1, qnum + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for idx, fig_idx in enumerate(["a", "d"]):
        pcm = axes[idx].pcolormesh(
            xx,
            yy,
            np.flipud(data_load[f"gjk_{fig_idx}"]),
            shading="nearest",
            norm=mpl.colors.SymLogNorm(
                linthresh=0.01, linscale=0.01, vmin=0.1, vmax=1, base=10
            ),
            cmap="RdBu",
        )
        axes[idx].set_yticks([1, 18, 36], [r"$Q_{36}$", r"$Q_{19}$", r"$Q_{1}$"])
        axes[idx].set_xticks([1, 19, 36], [r"$Q_{1}$", r"$Q_{19}$", r"$Q_{36}$"])
        axes[idx].set_title(f"fig4.{fig_idx}")
        fig.colorbar(pcm, ax=axes[idx])
    plt.show()

    xx, yy = np.meshgrid(
        data_load["t"],
        np.arange(1, qnum + 1),
    )
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    pcm = ax.pcolormesh(
        xx,
        yy,
        np.flipud(data_load["gj_b"].T),
        shading="nearest",
        norm=mpl.colors.SymLogNorm(
            linthresh=0.01, linscale=0.01, vmin=0.01, vmax=1, base=10
        ),
        cmap="RdBu",
    )
    ax.plot(data_load["t"], 18 + data_load["t"] * data_load["vb"], "--", color="k")
    ax.plot(data_load["t"], 18 - data_load["t"] * data_load["vb"], "--", color="k")
    ax.set_yticks([1, 18, 36], [r"$Q_{36}$", r"$Q_{19}$", r"$Q_{1}$"])
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_title("fig4.b")
    fig.colorbar(pcm, ax=ax)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
    for i, fig_idx in enumerate(["e", "f"]):
        axes[i].errorbar(
            data_load["t_a"],
            data_load["kf_DTC"][0],
            data_load["kf_DTC"][1],
            color="gray",
        )
        axes[i].set_ylim([1e-9, 2e-1])
        axes[i].set_yscale("log")
        axes[i].set_title(f"fig4.{fig_idx}")
    axes[0].errorbar(
        data_load["t_b"],
        data_load["kf_e"][0],
        data_load["kf_e"][1],
    )
    axes[1].errorbar(
        data_load["t_b"],
        data_load["kf_f"][0],
        data_load["kf_f"][1],
    )
    for i in range(2):
        axes[i].set_ylim([3e-6, 2e-1])
    fig.tight_layout()
    plt.show()


def plot_figS6():
    data_load = sio.loadmat(f"{root_path}/data/figS6.mat", simplify_cells=True)
    fig, axes = plt.subplots(2, 5, figsize=(16, 4), dpi=100)
    colors = ["#548ec4", "#d1464f"]
    for pi, processor in enumerate([1, 2]):
        qnums = data_load[f"qnums processor {processor}"]
        for qi, qnum in enumerate(qnums):
            probs = data_load[f"{qnum}q processor {processor}"]
            for i in range(2):
                axes[pi, qi].text(0.1, 0.5, f"{qnum} Qubits")
                axes[pi, qi].bar(
                    [i],
                    probs[0][i],
                    width=0.5,
                    yerr=probs[1][i],
                    error_kw={"ecolor": "0.2", "capsize": 6},
                    alpha=0.9,
                    color=colors[i],
                )
                axes[pi, qi].tick_params(axis="x", pad=2)

            axes[pi, qi].set_xticks(
                [0, 1], [r"$P_{|1010\dots\sf\rangle}$", r"$P_{|0101\dots\sf\rangle}$"]
            )
            axes[pi, qi].set_yticks(
                [0, 0.2, 0.4, 0.6], labels=["0", "0.2", "0.4", "0.6"]
            )
            axes[pi, qi].set_ylim([0, 0.6])
            if qi == 0:
                axes[pi, qi].set_ylabel("Probability")
            else:
                plt.setp(axes[pi, qi].get_yticklabels(), visible=False)
            axes[pi, qi].tick_params(length=0)
            axes[pi, qi].grid(alpha=0.4)
        axes[0, 2].set_title(r"Processor I", fontname="Arial")
        axes[1, 2].set_title(r"Processor II", fontname="Arial")
    fig.tight_layout()


def plot_figS7():
    data_load = sio.loadmat(f"{root_path}/data/figS7.mat", simplify_cells=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 3), dpi=100)
    for plot_type in ["sparse", "dense"]:
        axes[0].errorbar(
            data_load[plot_type + "_phi"],
            data_load[plot_type + "_k"][0],
            data_load[plot_type + "_k"][1],
            fmt="o-",
            mfc="none",
            label=plot_type,
            alpha=0.8,
        )
        axes[1].errorbar(
            data_load[plot_type + "_q"],
            data_load[plot_type + "_kf"][0],
            data_load[plot_type + "_kf"][1],
            fmt="o",
            mfc="none",
            label=plot_type,
            alpha=0.8,
        )
    axes[0].set_xticks([0, np.pi, 2 * np.pi], [0, r"$\pi$", r"$2\pi$"])
    axes[1].set_xticks([-36, 0, 36])
    axes[0].grid()
    axes[1].grid()
    axes[1].legend()
    fig.tight_layout()
    plt.show()


def plot_figS8():
    data_load = sio.loadmat(f"{root_path}/data/figS8.mat", simplify_cells=True)
    fig, axes = plt.subplots(2, 5, figsize=(16, 3), dpi=100)
    qnums = data_load["qnums"]
    for qi, qnum in enumerate(qnums):
        axes[0, qi].errorbar(
            data_load[f"{qnum}q"]["phi"],
            data_load[f"{qnum}q"]["k"][0],
            data_load[f"{qnum}q"]["k"][1],
            fmt="o",
            mfc="none",
        )
        axes[1, qi].errorbar(
            data_load[f"{qnum}q"]["q"],
            data_load[f"{qnum}q"]["kf"][0],
            data_load[f"{qnum}q"]["kf"][1],
            fmt="o",
            mfc="none",
        )
        axes[0, qi].set_title(f"{qnum}q")
    plt.show()


def plot_figS9():
    data_load = sio.loadmat(f"{root_path}/data/figS9.mat", simplify_cells=True)
    fig, axes = plt.subplots(2, 5, figsize=(16, 3), dpi=100)
    qnums = data_load["qnums"]
    for qi, qnum in enumerate(qnums):
        axes[0, qi].errorbar(
            data_load[f"{qnum}q"]["phi"],
            data_load[f"{qnum}q"]["k"][0],
            data_load[f"{qnum}q"]["k"][1],
            fmt="o",
            mfc="none",
        )
        axes[1, qi].errorbar(
            data_load[f"{qnum}q"]["q"],
            data_load[f"{qnum}q"]["kf"][0],
            data_load[f"{qnum}q"]["kf"][1],
            fmt="o",
            mfc="none",
        )
        axes[0, qi].set_title(f"{qnum}q")
    plt.show()


def plot_figS10():
    data_load = sio.loadmat(f"{root_path}/data/figS10.mat", simplify_cells=True)
    qnums = data_load["qnums"]
    fig = plt.figure(figsize=(12, 8), dpi=100)
    axes = fig.subplots(5, 1)
    axes = axes.flatten()
    for qi, qnum in enumerate(qnums):
        data_dic = data_load[f"{qnum}q"]
        axes[qi].errorbar(
            data_dic["gamma"],
            data_dic["parity"][0],
            data_dic["parity"][1],
            fmt="o-",
            mfc="none",
        )
        if qnum in [14, 20]:
            for load_part in ["a", "b"]:
                axes[qi].plot(
                    data_dic[f"gamma_{load_part}"],
                    data_dic[f"parity_{load_part}"],
                    ".-",
                    color="gray",
                )
        axes[qi].set_title(f"{qnum}")
        axes[qi].set_xticks([-np.pi / 2, 0, np.pi / 2], [r"$\pi/2$", 0, r"$\pi/2$"])
        axes[qi].set_ylim([-1.05, 1.05])
    fig.tight_layout()
    plt.show()


def plot_figS11():
    data_load = sio.loadmat(f"{root_path}/data/figS11.mat", simplify_cells=True)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot()
    colors = [
        "#866aa0",
        "#4d818e",
        "#d3915f",
    ]

    kw = {
        "markersize": 4.5,
        "lw": 1,
        "capsize": 2.5,
        "mfc": "none",
        "alpha": 1,
    }

    ax.errorbar(
        data_load["sim_qnums"],
        data_load["sim_fids"][0],
        data_load["sim_fids"][1],
        fmt="s--",
        label="Sim. (noisy)",
        color=colors[0],
        **kw,
    )
    ax.errorbar(
        data_load["exp_qnums_parity"],
        data_load["exp_parity_fids"][0],
        data_load["exp_parity_fids"][1],
        fmt="o-",
        label="Exp. (Parity)",
        color=colors[1],
        **kw,
    )
    ax.errorbar(
        data_load["exp_qnums_mqc"],
        data_load["exp_mqc_fids"][0],
        data_load["exp_mqc_fids"][1],
        fmt="^-",
        label="Exp. (MQC)",
        color=colors[2],
        **kw,
    )
    y_low = 0.5
    y_high = 1.02
    mypal1 = sns.cubehelix_palette(n_colors=9, rot=0.4, gamma=1)
    fill_xs = [1.5, 2.5, 4.5, 9.5, 14.5, 20.5, 28.5, 36.5]
    for i in range(len(fill_xs) - 1):
        ax.fill_betweenx(
            y=[y_low, y_high],
            x1=fill_xs[i],
            x2=fill_xs[i + 1],
            color=mypal1[i],
            alpha=0.35,
            linewidth=0,
        )
        ax.plot(
            [(fill_xs[i] + fill_xs[i + 1]) * 0.5] * 2,
            [y_high, y_high * 0.987],
            "-",
            lw=0.8,
            color="k",
        )
        ax.text((fill_xs[i] + fill_xs[i + 1]) * 0.5, y_high, str(i + 1), ha="center")
    ax.text(
        (fill_xs[0] + fill_xs[-1]) * 0.5,
        y_high + 0.02,
        "Number of CZ Layers",
        ha="center",
    )
    ax.grid(axis="y", alpha=0.5)
    ax.set_xlabel(r"Qubit Number $N$", labelpad=-0.03)
    ax.set_ylabel(r"$\mathcal{F}$", labelpad=-0.03, usetex=True)
    ax.set_ylim([0.7, 1.01])
    ax.set_xlim([1.5, 36.5])
    ax.set_yticks([0.7, 0.8, 0.9, 1.0], [0.7, 0.8, 0.9, 1.0])
    ax.set_xticks(data_load["exp_qnums_mqc"])

    ax.legend(
        loc="upper right",
        ncols=1,
        borderpad=0.17,
        labelspacing=0.5,
        handletextpad=0.5,
        handlelength=1.2,
        framealpha=0.0,
        prop={"size": 8},
    )

    ax.tick_params(
        length=1.5, direction="in", bottom=True, top=False, left=True, right=True
    )
    plt.show()


def plot_figS13():
    data_load = sio.loadmat(f"{root_path}/data/figS13.mat", simplify_cells=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=100)
    axes = axes.flatten()
    ms = data_load["ms"]
    for pi, plot_i in enumerate(["phi_4", "phi_pi"]):
        for pj, plot_j in enumerate(["delta_m", "delta_p", "phi"]):
            axes[0].errorbar(
                ms,
                data_load[f"lin_{plot_j}_{plot_i}"][0],
                data_load[f"lin_{plot_j}_{plot_i}"][1],
                fmt="o-",
                mfc="none",
                color=colors_iter[pi],
                alpha=pj * 0.3 + 0.3,
                capsize=5,
            )
    axes[0].set_xlabel("m")
    axes[0].set_ylabel(r"$\varphi$")

    axes[1].errorbar(
        data_load["detuning"],
        data_load["phis"][0],
        data_load["phis"][1],
        fmt="o-",
        mfc="none",
        capsize=5,
    )
    for name_i in ["4", "pi"]:
        axes[1].errorbar(
            data_load[f"detune_{name_i}"],
            data_load[f"phi_fit_phi_{name_i}"][0],
            data_load[f"phi_fit_phi_{name_i}"][1],
            fmt="*",
            mfc="none",
            capsize=5,
            markersize=20,
        )
        axes[2].errorbar(
            data_load["ms"],
            data_load[f"phi_m_phi_{name_i}"][0],
            data_load[f"phi_m_phi_{name_i}"][1],
            fmt="o-",
            mfc="none",
            capsize=5,
        )
    data = [np.pi, np.pi * 2 - 4]
    for ni, name_i in enumerate(["cz_phi", "cp_phi"]):
        axes[3].plot(
            range(len(data_load[name_i])),
            data_load[name_i],
            "o",
            mfc="none",
        )
    axes[1].set_xlabel(r"$\Delta$")
    axes[1].set_ylabel(r"$\phi$")
    axes[2].set_yticks([-4 + np.pi * 2, np.pi], ["-4", r"$\pi$"])
    axes[3].set_yticks([-4 + np.pi * 2, np.pi], ["-4", r"$\pi$"])

    for i in range(4):
        axes[i].grid()
    fig.tight_layout()
    plt.show()


def plot_TableS3():
    data_load = sio.loadmat(f"{root_path}/data/TableS3.mat", simplify_cells=True)
    changecolor = mpl.colors.Normalize(vmin=0, vmax=1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), dpi=100)
    for ai, data_name in enumerate(["mbl", "pLSB", "scar"]):
        ipr = data_load[f"ipr_{data_name}"]
        pcm = axes[ai].scatter(
            ipr[:, 0],
            ipr[:, 1],
            c=ipr[:, 2],
            cmap="RdBu_r",
            s=ipr[:, 2] * 100 + 1,
            norm=changecolor,
        )
        axes[ai].set_title(data_name)
        fig.colorbar(pcm, ax=axes[ai])
    fig.tight_layout()
    plt.show()


def plot_figS15():
    data_load = sio.loadmat(f"{root_path}/data/figS15.mat", simplify_cells=True)
    changecolor = mpl.colors.Normalize(vmin=0, vmax=1)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), dpi=100)
    for ai, data_name in enumerate(["scar", "mbl1", "mbl2", "mbl3"]):
        ipr = data_load[f"ipr_{data_name}"]
        pcm = axes[0, ai].scatter(
            ipr[:, 0],
            ipr[:, 1],
            c=ipr[:, 2],
            cmap="RdBu_r",
            s=ipr[:, 2] * 100 + 1,
            norm=changecolor,
        )
        fig.colorbar(pcm, ax=axes[0, ai])
        axes[1, ai].plot(data_load[f"ji_{data_name}"])
    fig.tight_layout()
    plt.show()


def plot_figS16():
    data_load = sio.loadmat(f"{root_path}/data/figS16.mat", simplify_cells=True)
    lmbda_ab = data_load["lmbda_ab"]
    lmbda_cd = data_load["lmbda_cd"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), dpi=100)
    for li, L in enumerate([8, 14]):
        axes[li].plot(
            lmbda_ab,
            data_load[f"ipr_catscar_{L}"],
            "o-",
            alpha=0.8,
            mfc="none",
            label="Cat scar",
        )
        for ni, data_name in enumerate(["highest 10%", "average", "lowest 10%"]):
            axes[li].plot(
                lmbda_ab,
                data_load[f"ipr_mbl_{L}"][ni],
                "o-",
                alpha=0.8,
                mfc="none",
                label=f"MBL {data_name}",
            )
        axes[li].set_ylim([-0.05, 0.52])
        axes[li].set_title(f"L={L}")

    for li, L in enumerate(data_load["qnums"]):
        axes[2].plot(lmbda_cd, data_load["ea_mbl"][li], "o-", alpha=0.8, label=f"L={L}")
        axes[3].plot(
            lmbda_cd, data_load["ea_scar"][li], "o-", alpha=0.8, label=f"L={L}"
        )
    axes[2].set_title("MBL")
    axes[3].set_title("scar")
    for i in range(4):
        axes[i].set_xscale("log")
        axes[i].set_xlabel(r"$\lambda$")
        axes[i].legend()
    fig.tight_layout()
    plt.show()


def plot_figS19():
    data_load = sio.loadmat(f"{root_path}/data/figS19.mat", simplify_cells=True)
    qnum = data_load["qnum"]
    xx, yy = np.meshgrid(data_load["t"], np.arange(1, qnum + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))

    for sample in range(3):
        ax = axes[sample]
        vb = data_load["vb"][sample]
        pcm = ax.pcolormesh(
            xx,
            yy,
            np.flipud(data_load["gj"][sample]),
            shading="nearest",
            cmap="RdBu",
        )
        ax.plot(
            data_load["t"],
            11 + data_load["t"] * vb,
            "--",
            color="k",
        )
        ax.plot(
            data_load["t"],
            11 - data_load["t"] * vb,
            "--",
            color="k",
        )
        ax.set_yticks([1, 11, 20], [r"$Q_{20}$", r"$Q_{10}$", r"$Q_{1}$"])
        ax.set_xticks(np.arange(0, 101, 20))
        fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.show()


def plot_figS20():
    data_load = sio.loadmat(f"{root_path}/data/figS20.mat", simplify_cells=True)
    qnum = data_load["qnum"]
    qnum_sim = data_load["qnum_sim"]
    fig, axes = plt.subplots(5, 4, figsize=(15, 12))
    for sample in range(10):
        ax = axes[sample % 5, sample // 5 * 2]
        vb = data_load["vb"][sample]
        xx, yy = np.meshgrid(data_load["t"], np.arange(1, qnum + 1))
        pcm = ax.pcolormesh(
            xx,
            yy,
            np.flipud(data_load["gj_flip1"][sample].T),
            shading="nearest",
            norm=mpl.colors.SymLogNorm(
                linthresh=0.01, linscale=0.01, vmin=0.01, vmax=1, base=10
            ),
            cmap="RdBu",
        )
        ax.plot(
            data_load["t"],
            18 + data_load["t"] * vb,
            "--",
            color="k",
        )
        ax.plot(
            data_load["t"],
            18 - data_load["t"] * vb,
            "--",
            color="k",
        )
        ax.set_yticks([1, 18, 36], [r"$Q_{36}$", r"$Q_{19}$", r"$Q_{1}$"])
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_title(f"sample:{sample+1}, vb = {np.round(vb,4)}")
        fig.colorbar(pcm, ax=ax)

        xx, yy = np.meshgrid(data_load["t"], np.arange(1, qnum_sim + 1))
        ax = axes[sample % 5, sample // 5 * 2 + 1]
        pcm = ax.pcolormesh(
            xx,
            yy,
            np.flipud(data_load["gj_sim"][sample]),
            shading="nearest",
            vmin=0,
            vmax=1.0,
            cmap="RdBu",
        )
        ax.plot(
            data_load["t"],
            11 + data_load["t"] * vb,
            "--",
            color="k",
        )
        ax.plot(
            data_load["t"],
            11 - data_load["t"] * vb,
            "--",
            color="k",
        )
        ax.set_yticks([1, 11, 20], [r"$Q_{20}$", r"$Q_{10}$", r"$Q_{1}$"])
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_title(f"sample:{sample+1}, vb = {np.round(vb,4)}")
        fig.colorbar(pcm, ax=ax)
    fig.tight_layout()
    plt.show()


def plot_figS21():
    data_load = sio.loadmat(f"{root_path}/data/figS21.mat", simplify_cells=True)
    qnum = data_load["qnum"]
    xx, yy = np.meshgrid(
        data_load["t"],
        np.arange(1, qnum + 1),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8, 4))
    axes = axes.flatten()
    for pi, plot_name in enumerate(["flip1_flip", "flip1", "flip2", "flip4"]):
        pcm = axes[pi].pcolormesh(
            xx,
            yy,
            data_load[plot_name].T,
            shading="nearest",
            norm=mpl.colors.SymLogNorm(
                linthresh=0.01, linscale=0.01, vmin=0.01, vmax=1, base=10
            ),
            cmap="RdBu",
        )
        if pi == 0:
            axes[pi].set_yticks([1, 36], [r"$Q_{36}$", r"$Q_{1}$"])
        if pi == 1:
            axes[pi].set_yticks([1, 18, 36], [r"$Q_{36}$", r"$Q_{19}$", r"$Q_{1}$"])
        if pi == 2:
            axes[pi].set_yticks(
                [1, 9, 27, 36], [r"$Q_{36}$", r"$Q_{27}$", r"$Q_{9}$", r"$Q_{1}$"]
            )
        if pi == 3:
            axes[pi].set_yticks(
                [1, 5, 14, 23, 32, 36],
                [
                    r"$Q_{36}$",
                    r"$Q_{32}$",
                    r"$Q_{23}$",
                    r"$Q_{14}$",
                    r"$Q_{5}$",
                    r"$Q_{1}$",
                ],
            )
        axes[pi].set_xticks(np.arange(0, 101, 20))
        fig.colorbar(pcm, ax=axes[pi])
    plt.show()


def plot_figS22():
    data_load = sio.loadmat(f"{root_path}/data/figS22.mat", simplify_cells=True)
    ts = data_load["ts"]
    plt.figure(figsize=(8, 3))
    for plot_name in ["dtc", "rabi", "idle"]:
        plot_data = data_load[plot_name]
        plt.errorbar(
            ts,
            plot_data["mean"],
            plot_data["std"],
            fmt="o-",
            mfc="none",
            capsize=5,
            label=plot_name,
        )
    plt.legend()


def plot_figS23():
    data_load = sio.loadmat(f"{root_path}/data/figS23.mat", simplify_cells=True)
    iqs = data_load["iqs"]
    fig, axes = plt.subplots(3, 4, figsize=(10, 5))
    for pi, plot_name in enumerate(["dtc", "rabi", "idle"]):
        plot_data = data_load[plot_name]
        for ti, t in enumerate([0, 6, 16, 22]):
            axes[pi, ti].errorbar(
                iqs,
                plot_data["mean"][:, ti],
                plot_data["std"][:, ti],
                fmt="o-",
                mfc="none",
                capsize=5,
                color=colors_iter[pi],
            )
            axes[pi, ti].set_title(plot_name + f" t={t}")
            axes[pi, ti].set_ylim([-0.05, 1])
            axes[pi, ti].grid()
    fig.tight_layout()


def plot_figS24():
    data_load = sio.loadmat(f"{root_path}/data/figS24.mat", simplify_cells=True)
    fig,axes = plt.subplots(1,2)
    for pi,plot_name in enumerate(['PBC','OBC']):
        axes[pi].plot(data_load[plot_name][0],data_load[plot_name][1],'.')
        axes[pi].set_title(plot_name)
        
def plot_figS25():
    data_load = sio.loadmat(f"{root_path}/data/figS25.mat", simplify_cells=True)
    plt.plot(data_load['ipr'][0],data_load['ipr'][1],'.')
    

def plot_figS26():
    data_load = sio.loadmat(f"{root_path}/data/figS26.mat", simplify_cells=True)
    fig,axes = plt.subplots(1,2)
    for pi,plot_type in enumerate(['LRE','SRE']):
        for l in [8,10,12]:
            axes[pi].plot(data_load[f'L{l}_{plot_type}'][:,0],data_load[f'L{l}_{plot_type}'][:,1],'o',label=f'N={l}')
        axes[pi].set_xscale('log')
        axes[pi].set_xlabel('t/T')
        axes[pi].set_title(plot_type)
        axes[pi].legend()
    
    
def plot_figS27():
    data_load = sio.loadmat(f"{root_path}/data/figS27.mat", simplify_cells=True)
    ts = data_load['ts']
    fig,axes = plt.subplots(2,2)
    for l,L in enumerate([10,14]):
        for pi, plot_type in enumerate(['C(t)','Kf(2N,t)']):
            for i in range(4):
                axes[l,pi].plot(ts,data_load[f'L{L}_{plot_type}'][i],'o',label=f'{i}')
            axes[l,pi].legend()
            axes[l,pi].set_xlabel('t/T')
            axes[l,pi].set_ylabel(plot_type)
            axes[l,pi].set_title(f'N={L}')
    fig.tight_layout()
            
