"""Export partial candidates and complete nested MAVIC tables."""
from importlib.resources import files
from pathlib import Path
import os
import pandas as pd
from .project import STAGES

def export(project, purpose="both"):
    if purpose not in ("both","fixed","candidate"):
        raise ValueError("Choose both, fixed or candidate")
    project.verify_inputs()
    registry = pd.read_csv(files("mavic_extract").joinpath("data/registry.csv"))
    full = registry.query("fixed == 1 and cohort == 'full'").feature.tolist()
    ling = registry.query("fixed == 1 and cohort == 'linguistic'").feature.tolist()
    rows, ledger, missing = [], [], []
    for item in project.manifest:
        row = {"video_id":item["video_id"]}
        records = {s:project.record(item,s) for s in STAGES}
        for field in registry.itertuples():
            r = records.get(field.stage,{})
            detail = r.get("detail",{})
            reason = item["error"] or detail.get("reason","") or detail.get("errors",{}).get(field.producer,"")
            val = item["duration_seconds"] if field.stage == "media" else r.get("values",{}).get(field.feature)
            if field.feature != "avg_shots_per_min" and detail.get("observations",field.minimum_observations) < field.minimum_observations:
                val,reason = None,"Too few observations for this statistic"
            if val is None:
                missing.append(dict(video_id=row["video_id"],feature=field.feature,reason=reason or ("Stage not run" if not r else "Measure undefined or unavailable")))
            row[field.feature]=val
        absent_av = [k for k in full if row[k] is None]
        absent_text = [k for k in ling if row[k] is None]
        ledger.append(dict(video_id=row["video_id"],full=not absent_av,linguistic=not (absent_av or absent_text),missing_av=" ".join(absent_av),missing_linguistic=" ".join(absent_text)))
        rows.append(row)
    frame = pd.DataFrame(rows)
    cohorts = pd.DataFrame(ledger)
    dest = project.folder / "exports"
    dest.mkdir(exist_ok=True)
    def save(df,name):
        tmp = dest / (name+".tmp")
        df.to_csv(tmp,index=False)
        os.replace(tmp,dest/name)
    if purpose in ("candidate","both"):
        for name,indicator in [("candidates.csv",1),("ancillary.csv",0)]:
            save(frame[["video_id",*registry.loc[registry.candidate==indicator,"feature"]]],name)
    if purpose in ("fixed","both"):
        save(frame.loc[cohorts.full,["video_id",*full]],"full_features.csv")
        save(frame.loc[cohorts.linguistic,["video_id",*full,*ling]],"linguistic_features.csv")
    save(cohorts,"cohorts.csv")
    save(pd.DataFrame(missing,columns=["video_id","feature","reason"]),"missing.csv")
    save(registry,"feature_dictionary.csv")
    return cohorts
