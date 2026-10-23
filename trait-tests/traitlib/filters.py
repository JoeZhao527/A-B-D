"""Population filters shared by adjustment, trait tests and the model x trait matrix."""


def harness_filter(df, min_models=2):
    """Drop trajectories on harnesses used by fewer than `min_models` distinct models.

    Such a harness dummy is collinear with its single model, so no model of the form
    y ~ C(harness) + model can separate the scaffold from the model there. Returns
    (filtered df, sorted list of dropped harnesses). min_models <= 1 disables."""
    if min_models <= 1 or "harness" not in df.columns:
        return df, []
    hm = df.groupby("harness")["model"].nunique()
    drop = sorted(hm[hm < min_models].index)
    return df[~df["harness"].isin(drop)].reset_index(drop=True), drop


def backbone_models(df, min_ctx=3):
    """Models spanning >= min_ctx task types (the T2 population)."""
    cov = df.groupby("model")["task_type"].nunique()
    return set(cov[cov >= min_ctx].index)


def cell_key(df):
    return df["model"].astype(str) + " @ " + df["task_type"].astype(str)
