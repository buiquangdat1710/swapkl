"""
Ghi thêm dòng vào CSV mà KHÔNG lệch cột.

Lỗi đã gặp: `df.to_csv(f, mode='a', header=not exists)` ghi giá trị theo THỨ TỰ CỘT
của DataFrame mới, chứ không theo tiêu đề đã có trong file. Khi hai lần chạy sinh ra
tập cột khác nhau — ví dụ `--tasks all` cho 18 cột còn `--tasks mmlu` chỉ cho 7 —
các giá trị bị nhét vào 7 vị trí đầu tiên. Kết quả: điểm MMLU nằm trong cột `piqa`,
còn cột `mmlu` thành rỗng. Dữ liệu trông vẫn bình thường nên rất dễ tin nhầm.
"""
import os

import pandas as pd


def append_row(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if not os.path.exists(path):
        df.to_csv(path, index=False)
        return df
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    new = [c for c in df.columns if c not in cols]
    if new:
        old = pd.read_csv(path)
        pd.concat([old, df.reindex(columns=cols + new)], ignore_index=True) \
          .reindex(columns=cols + new).to_csv(path, index=False)
    else:
        df.reindex(columns=cols).to_csv(path, mode='a', header=False, index=False)
    return df


def repair(path: str, value_cols=('mmlu', 'mmlu_weighted'), shifted_into=('piqa', 'piqa_stderr')):
    """Sửa các dòng đã bị lệch: mmlu rỗng nhưng piqa có giá trị -> chuyển ngược lại."""
    d = pd.read_csv(path)
    for v, s in zip(value_cols, shifted_into):
        if v not in d.columns or s not in d.columns:
            continue
        bad = d[v].isna() & d[s].notna() & d.get('avg_zero_shot', pd.Series(index=d.index)).isna()
        if bad.any():
            print(f'  {bad.sum()} dòng: chuyển {s} -> {v}')
            for t, x in zip(d.loc[bad, 'tag'], d.loc[bad, s]):
                print(f'    {t}: {x}')
            d.loc[bad, v] = d.loc[bad, s]
            d.loc[bad, s] = float('nan')
    d.to_csv(path, index=False)
    return d
