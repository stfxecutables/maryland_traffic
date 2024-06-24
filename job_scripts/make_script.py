ALWAYS_DROP = ["article", "acc_blame"]
CLUSTERED = [
    "chrg_sect_mtch",
    "chrg_title_mtch",
    "chrg_title",
    "outcome",
    "search_conducted",
    "search_disposition",
    "search_type",
    "stop_chrg_title",
    "violation_type",
]

TARGETS = ["outcome", "violation_type", "search_conducted", "chrg_title_mtch"]

if __name__ == "__main__":
    all_targs = []
    all_drops = []
    all_outs = []
    for target in TARGETS:
        drops = CLUSTERED.copy()
        drops.remove(target)
        for label, feats in {
            "race+sex": (),
            "nosex": ("race",),
            "norace": ("sex",),
            "norace+nosex": ("race", "sex"),
        }.items():
            all_drops.append(sorted([*drops, *feats]))
            all_outs.append(f"{target}__{label}")
            all_targs.append(target)

    for i, drops in enumerate(all_drops):
        s = ",".join(drops)
        print(f"DROPS{i:02d}={s}")
    for i, out in enumerate(all_outs):
        print(f"OUT{i:02d}={out}")
    for i, targ in enumerate(all_targs):
        print(f"TARGET{i:02d}={targ}")
