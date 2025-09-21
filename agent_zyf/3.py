import os
import re
import glob
from typing import Dict, List, Tuple, Optional

import pandas as pd



def _unique_name_smiles(df: pd.DataFrame, prefix: str) -> Dict[str, str]:
    name_col = f"{prefix}_name"
    smile_col = f"{prefix}_SMILE"
    if name_col not in df.columns or smile_col not in df.columns:
        return {}
    sub = df[[name_col, smile_col]].dropna().drop_duplicates()
    return dict(zip(sub[name_col].astype(str), sub[smile_col].astype(str)))


def _unique_ratios(df: pd.DataFrame, prefix: str) -> List[float]:
    ratio_col = f"{prefix}_ratio"
    if ratio_col not in df.columns:
        return []
    vals = (
        df[ratio_col]
        .apply(lambda x: None if pd.isna(x) or (isinstance(x, str) and x.strip() == "" ) else x)
        .dropna()
        .astype(float)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return vals


def _parse_conditions(cond_str: str) -> Dict[str, List[str]]:
    """Parse user condition string like 'SubstanceA_name=a|b;SubstanceB_ratio=0.2|0.3'."""
    cond_str = cond_str.strip()
    if not cond_str:
        return {}
    out: Dict[str, List[str]] = {}
    for part in cond_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        vs = [itm.strip() for itm in v.split("|") if itm.strip() != ""]
        if k and vs:
            out[k] = vs
    return out

def get_all_prefixes() -> List[str]:
    """自动获取根目录下所有 features_*.csv 的 prefix 列表"""
    files = glob.glob("features_*.csv")
    prefixes = []
    for f in files:
        # 文件名如 features_SubstanceA.csv
        basename = os.path.basename(f)
        if basename.startswith("features_") and basename.endswith(".csv"):
            prefix = basename[len("features_"):-len(".csv")]
            prefixes.append(prefix)
    return prefixes

def _filter_and_save_features(prefix: str) -> Optional[str]:
    src_path = f"features_{prefix}.csv"
    if not os.path.exists(src_path):
        print(f"跳过: 未找到 {os.path.basename(src_path)}")
        return None
    try:
        df = pd.read_csv(src_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"读取 {src_path} 出错: {e}")
        return None
    selected = pd.read_csv(f"top_ranked_features_{prefix}.csv", encoding='utf-8-sig')
    selected_features = selected.iloc[:, 0].astype(str).tolist()
    ratio_col = f"{prefix}_ratio"
    keep = [c for c in selected_features if c in df.columns]
    if ratio_col in df.columns and ratio_col not in keep:
        keep.append(ratio_col)
    if not keep:
        print(f"警告: 没有匹配到任何特征列，全部保留。")
        keep = df.columns.tolist()
    df_filtered = df[keep].copy()
    out_path = f"filtered_features_{prefix}.csv"
    try:
        df_filtered.to_csv(out_path, encoding="utf-8-sig", index=False)
        print(f"已生成筛选后的特征: {os.path.basename(out_path)} (列数: {df_filtered.shape[1]})")
        return out_path
    except Exception as e:
        print(f"写入 {out_path} 出错: {e}")
        return None


def _load_filtered_features_per_label(prefix: str, example_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    preferred = [
        f"filtered_features_{prefix}.csv",
        f"{prefix}_filtered_features.csv",
    ]
    feat_path = None
    for p in preferred:
        if os.path.exists(p):
            feat_path = p
            break
    if feat_path is None:
        candidate = f"features_{prefix}.csv"
        if not os.path.exists(candidate):
            return None
        feat_path = candidate

    feats_df = pd.read_csv(feat_path, encoding="utf-8-sig")
    name_col = f"{prefix}_name"
    if name_col not in example_df.columns:
        return None
    if len(example_df) < len(feats_df):
        feats_df = feats_df.iloc[: len(example_df)].copy()
    labels_series = example_df[name_col].astype(str).reset_index(drop=True)
    feats_df = feats_df.reset_index(drop=True)
    feats_df[name_col] = labels_series[: len(feats_df)]

    # 按物质名分组取均值
    comp = feats_df.groupby(name_col, as_index=True).mean(numeric_only=True)
    if comp.empty:
        return None

    # 去掉所有值都一样的列（BayBE要求）
    nunique = comp.nunique(axis=0)
    keep_cols = nunique[nunique > 1].index.tolist()
    if not keep_cols:
        return None
    comp = comp[keep_cols]

    return comp

def _build_parameters(example_df: pd.DataFrame, conditions: Dict[str, List[str]]):
    from baybe.parameters.custom import CustomDiscreteParameter
    from baybe.parameters import NumericalDiscreteParameter

    parameters = []
    fixed_assignments: Dict[str, object] = {}
    name_to_smiles_all: Dict[str, Dict[str, str]] = {}

    prefixes = get_all_prefixes()
    for prefix in prefixes:
        names_map = _unique_name_smiles(example_df, prefix)
        #print(names_map)
        name_to_smiles_all[prefix] = names_map

        name_key = f"{prefix}_name"
        if name_key in conditions:
            allow = set(conditions[name_key])
            names_map = {k: v for k, v in names_map.items() if k in allow}
        #print(names_map)
        if len(names_map) >= 2:
            comp = _load_filtered_features_per_label(prefix, example_df)
            if comp is None:
                labels = list(names_map.keys())
                ohe_cols = [f"ohe_{i}" for i in range(len(labels))]
                comp = pd.DataFrame(0.0, index=labels, columns=ohe_cols)
                for i, lb in enumerate(labels):
                    comp.loc[lb, ohe_cols[i]] = 1.0

            comp_sel = comp.copy()
            comp_sel.index = comp_sel.index.astype(str)
            #print(name_key)
            #print(len(name_key))
            #print(comp_sel)
            #print(len(comp_sel))
            param = CustomDiscreteParameter(name=name_key, data=comp_sel, decorrelate=False)
            parameters.append(param)
            #print(parameters)
        elif len(names_map) == 1:
            only_val = next(iter(names_map.keys()))
            fixed_assignments[name_key] = only_val

        ratios = _unique_ratios(example_df, prefix)
        ratio_key = f"{prefix}_ratio"
        if ratio_key in conditions:
            allow_raw = conditions[ratio_key]
            allow_vals: List[float] = []
            for t in allow_raw:
                try:
                    allow_vals.append(float(t))
                except Exception:
                    pass
            if allow_vals:
                ratios = [x for x in ratios if x in set(allow_vals)]

        if len(ratios) >= 2:
            tol = 1e-6
            parameters.append(NumericalDiscreteParameter(name=ratio_key, values=ratios, tolerance=tol))
        elif len(ratios) == 1:
            fixed_assignments[ratio_key] = ratios[0]

    return parameters, fixed_assignments, name_to_smiles_all


def _build_campaign(parameters, targets_spec: Optional[Dict[str, List[str]]] = None) -> object:
    from baybe import Campaign
    from baybe.objectives import SingleTargetObjective, DesirabilityObjective
    from baybe.targets import NumericalTarget
    from baybe.searchspace import SearchSpace

    if targets_spec:
        # Parse target spec like: Target_alpha_tg=MAX(0,100) or MIN(0,300) or MATCH(95,105)
        targets = []
        for key, vals in targets_spec.items():
            # Take the last spec in case of multiple
            spec = vals[-1]
            mode = "MAX"
            bounds = (0.0, 100.0)
            m = re.match(r"^(MAX|MIN|MATCH)\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)$", spec, flags=re.I)
            if m:
                mode = m.group(1).upper()
                bounds = (float(m.group(2)), float(m.group(3)))
            else:
                # If only mode given without bounds
                mm = re.match(r"^(MAX|MIN|MATCH)$", spec, flags=re.I)
                if mm:
                    mode = mm.group(1).upper()
            t = NumericalTarget(name=key, mode=mode, bounds=bounds if mode != "MAX" and mode != "MIN" else None)
            targets.append(t)
        if not targets:
            targets = [NumericalTarget(name="Target_alpha_tg", mode="MAX")]
        objective = DesirabilityObjective(targets=targets)
    else:
        target = NumericalTarget(name="Target_alpha_tg", mode="MAX")
        objective = SingleTargetObjective(target=target)

    searchspace = SearchSpace.from_product(parameters=parameters)
    return Campaign(searchspace=searchspace, objective=objective)


def _choose_example_columns(example_df: pd.DataFrame) -> List[str]:
    return list(example_df.columns)


def _append_recommendation_row(
    example_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    fixed_assignments: Dict[str, object],
    name_to_smiles_all: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Append one recommended row to example_df keeping the same column order.

    Target columns will be set to empty strings.
    """
    cols = _choose_example_columns(example_df)
    new_row: Dict[str, object] = {c: "" for c in cols}

    # Number column if present
    number_col = None
    for c in cols:
        if c.lower() == "number":
            number_col = c
            break
    if number_col is not None:
        try:
            max_num = pd.to_numeric(example_df[number_col], errors="coerce").fillna(0).astype(int).max()
            new_row[number_col] = max_num + 1
        except Exception:
            new_row[number_col] = ""

    # Fill parameters from recommendation or fixed defaults
    rec = rec_df.iloc[0].to_dict() if len(rec_df) > 0 else {}

    prefixes = get_all_prefixes()
    for prefix in prefixes:
        name_key = f"{prefix}_name"
        smile_key = f"{prefix}_SMILE"
        ratio_key = f"{prefix}_ratio"

        # Name
        val_name = rec.get(name_key, fixed_assignments.get(name_key, None))
        if val_name is not None:
            new_row[name_key] = val_name
            new_row[smile_key] = name_to_smiles_all.get(prefix, {}).get(str(val_name), "")

        # Ratio
        val_ratio = rec.get(ratio_key, fixed_assignments.get(ratio_key, None))
        if val_ratio is not None:
            new_row[ratio_key] = val_ratio

    # Leave target columns blank explicitly if they exist
    for c in cols:
        if re.match(r"^Target_", str(c)):
            new_row[c] = ""

    # Any other passthrough columns remain blank

    return pd.concat([example_df, pd.DataFrame([new_row], columns=cols)], ignore_index=True)


def main():
    print("请输入条件字符串：\n- 参数约束示例: SubstanceA_name=南亚127e;SubstanceB_ratio=0.3|0.2\n- 目标约束示例: Target_alpha_tg=MAX(0,100);Target_beta_impactstrength=MIN(0,300)\n直接回车跳过：")
    try:
        cond_str = input().strip()
    except Exception:
        cond_str = ""
    all_conditions = _parse_conditions(cond_str)
    # Split into parameter and target specs
    target_conditions = {k: v for k, v in all_conditions.items() if k.startswith("Target_")}
    param_conditions = {k: v for k, v in all_conditions.items() if not k.startswith("Target_")}

    example_df = pd.read_csv('example.csv', encoding='utf-8-sig')

    prefixes = get_all_prefixes()
    print(prefixes)
    for prefix in prefixes:
        _filter_and_save_features(prefix)

    parameters, fixed_assignments, name_to_smiles_all = _build_parameters(
        example_df, param_conditions
    )

    if len(parameters) == 0:
        print("警告: 搜索空间中没有可变参数（所有条件均为固定值），将直接复制最后一行作为推荐并清空因变量。")
        rec_df = pd.DataFrame()
    else:
        campaign = _build_campaign(parameters, targets_spec=target_conditions if target_conditions else None)
        rec_df = campaign.recommend(batch_size=1)
        # Ensure dtype consistency for discrete numerical values
        for col in rec_df.columns:
            if col.endswith("_ratio"):
                try:
                    rec_df[col] = pd.to_numeric(rec_df[col], errors="coerce")
                except Exception:
                    pass

    out_df = _append_recommendation_row(
        example_df, rec_df, fixed_assignments, name_to_smiles_all
    )

    # 推荐后归一化所有 ratio 列
    ratio_cols = [col for col in out_df.columns if col.endswith('_ratio')]
    row_idx = out_df.index[-1]
    total = sum([out_df.at[row_idx, col] for col in ratio_cols if pd.notna(out_df.at[row_idx, col])])
    if total > 0:
        for col in ratio_cols:
            if pd.notna(out_df.at[row_idx, col]):
                out_df.at[row_idx, col] = out_df.at[row_idx, col] / total

    out_path = "example_with_recommendation.csv"
    out_df.to_csv(out_path, encoding="utf-8-sig", index=False)
    print(f"已生成推荐并追加到文件: {out_path}")


if __name__ == "__main__":
    main()


